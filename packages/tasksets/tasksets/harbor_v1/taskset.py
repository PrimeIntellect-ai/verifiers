"""harbor: run a Harbor (Terminal-Bench) dataset.

`dataset` is a Harbor Hub registry id (e.g. "name", "name@version",
"org/name@ref"), downloaded + cached on first use via the `harbor` CLI
(`uv tool install harbor`). Each task dir ships task.toml + instruction.md
(+ tests/, solution/, environment/). Defaults to the registry `hello-world` task.

The harness runs in a container and edits /app; then the task's verifier
(tests/test.sh) runs in the SAME container and the reward it writes to
/logs/verifier/reward.txt becomes the score. The verification lives on the
taskset (the `solved` reward), so a harbor task runs under ANY harness.

A task's declared [environment].docker_image becomes a first-class `Task.image`
the Environment injects into the runtime. For a task that only ships an
environment/Dockerfile, the taskset reuses a matching image from the authenticated
Prime image list or creates one with the Prime CLI. `ignore_dockerfile` instead runs
the task on the harness runtime's image. A task with no environment also runs on that
image, unless `require_image`.
"""

import io
import subprocess
import tarfile
import tomllib
from pathlib import Path

from pydantic import Field

from tasksets.harbor_v1.images import ensure_prime_images
from verifiers.v1.decorators import reward
from verifiers.v1.errors import ProgramError
from verifiers.v1.runtimes import Runtime
from verifiers.v1.task import Task, TaskResources, TaskTimeout
from verifiers.v1.taskset import Taskset, TasksetConfig
from verifiers.v1.trace import Trace
from verifiers.v1.types import StrictBaseModel

CACHE = Path.home() / ".cache" / "harbor"


class HarborConfig(TasksetConfig):
    dataset: str = "hello-world"
    """A Harbor Hub registry id ("name", "name@version", "org/name@ref"),
    downloaded + cached via the `harbor` CLI."""
    tasks: list[str] | None = None
    """Optional subset of task names to load (None = all)."""
    timeout_multiplier: float = Field(1.0, gt=0)
    """Scale each task's agent and verifier timeouts."""
    resource_multiplier: float = Field(1.0, gt=0)
    """Scale each task's CPU, memory, and disk requests. GPU requests are unchanged."""
    require_image: bool = False
    """For a task with NO declared environment at all (no docker_image, no Dockerfile),
    whether to reject it (True) or run it on the runtime's default image (False)."""
    ignore_dockerfile: bool = False
    """Run a task whose environment is only a `Dockerfile` on the harness runtime's image
    instead of resolving or building a Prime image. Only correct when the harness image
    already has what the task needs."""


class Author(StrictBaseModel):
    name: str | None = None
    email: str | None = None


class HarborTask(Task):
    """A Harbor task. The base fields carry instruction.md (`prompt`), the
    resolved container `image`, the `timeout`/`resources`
    (from task.toml's [harness]/[verifier]/[environment]), and [task].name/description;
    the rest mirror [metadata]."""

    keywords: list[str] = []
    authors: list[Author] = []
    difficulty: str | None = None
    category: str | None = None
    tags: list[str] = []
    task_dir: str = Field(exclude=True)
    """Host path to the task dir; used to stage tests/ to verify, not serialized."""


def dataset_dir(dataset: str) -> Path:
    """Download a Harbor Hub `dataset` to a directory of task dirs via the `harbor`
    CLI, cached on first use."""
    out = CACHE / dataset.replace("/", "_").replace("@", "_")
    if not out.is_dir():
        try:
            subprocess.run(
                ["harbor", "download", dataset, "--export", "-o", str(out)], check=True
            )
        except FileNotFoundError as e:
            raise RuntimeError(
                f"the `harbor` CLI is needed to download {dataset!r}; "
                "install it with `uv tool install harbor`"
            ) from e
    return out


def resolve_image(
    task_dir: Path,
    config: dict,
    require_image: bool,
    ignore_dockerfile: bool = False,
    prime_image: str | None = None,
) -> str | None:
    """Resolve a declared image, a Prime-built Dockerfile image, or the runtime image."""
    declared = config.get("environment", {}).get("docker_image")
    if declared:
        return declared
    if (task_dir / "environment" / "Dockerfile").is_file():
        if ignore_dockerfile:
            return None
        if prime_image:
            return prime_image
        raise RuntimeError(f"{task_dir.name}: Prime image resolution returned no image")
    if require_image:
        raise ValueError(
            f"{task_dir.name}: no [environment].docker_image and require_image=True"
        )
    return None


def parse_resources(env: dict, multiplier: float = 1.0) -> TaskResources:
    """Map a task.toml [environment] block to TaskResources (0 gpus -> unset)."""
    return TaskResources(
        cpu=env["cpus"] * multiplier if env.get("cpus") else None,
        memory=env["memory_mb"] / 1024 * multiplier if env.get("memory_mb") else None,
        gpu=str(env["gpus"]) if env.get("gpus") else None,
        disk=env["storage_mb"] / 1024 * multiplier if env.get("storage_mb") else None,
    )


def parse_task(
    task_dir: Path,
    idx: int,
    harbor_config: HarborConfig,
    prime_image: str | None = None,
) -> HarborTask:
    """Read a harbor task dir (task.toml + instruction.md) into a typed task,
    handling both the [task].authors and legacy [metadata].author_name layouts."""
    config = tomllib.loads((task_dir / "task.toml").read_text())
    task, meta = config.get("task", {}), config.get("metadata", {})
    authors = [Author(**a) for a in task.get("authors", [])]
    if not authors and meta.get("author_name"):
        authors = [Author(name=meta["author_name"], email=meta.get("author_email"))]
    harness_timeout = config.get("agent", {}).get("timeout_sec")
    scoring_timeout = config.get("verifier", {}).get("timeout_sec")
    return HarborTask(
        idx=idx,
        name=task.get("name") or task_dir.name,
        description=task.get("description"),
        prompt=(task_dir / "instruction.md").read_text().strip(),
        image=resolve_image(
            task_dir,
            config,
            harbor_config.require_image,
            ignore_dockerfile=harbor_config.ignore_dockerfile,
            prime_image=prime_image,
        ),
        timeout=TaskTimeout(
            harness=harness_timeout * harbor_config.timeout_multiplier
            if harness_timeout is not None
            else None,
            scoring=scoring_timeout * harbor_config.timeout_multiplier
            if scoring_timeout is not None
            else None,
        ),
        resources=parse_resources(
            config.get("environment", {}), harbor_config.resource_multiplier
        ),
        keywords=task.get("keywords", []),
        authors=authors,
        difficulty=meta.get("difficulty"),
        category=meta.get("category"),
        tags=meta.get("tags", []),
        task_dir=str(task_dir),
    )


def make_tar(directory: Path) -> bytes:
    """Tar a directory's contents (flat) into a gzipped tarball."""
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w:gz") as tar:
        for item in sorted(directory.iterdir()):
            tar.add(item, arcname=item.name)
    return buffer.getvalue()


class HarborTaskset(Taskset[HarborTask, HarborConfig]):
    def load_tasks(self) -> list[HarborTask]:
        root = dataset_dir(self.config.dataset)
        task_dirs = [
            toml_path.parent
            for toml_path in sorted(root.rglob("task.toml"))
            if (toml_path.parent / "instruction.md").is_file()
            and (
                self.config.tasks is None or toml_path.parent.name in self.config.tasks
            )
        ]
        if not task_dirs:
            raise ValueError(f"no harbor tasks found in {root}")
        prime_images = (
            {}
            if self.config.ignore_dockerfile
            else ensure_prime_images(self.config.dataset, task_dirs)
        )
        return [
            parse_task(
                task_dir,
                idx,
                self.config,
                prime_image=prime_images.get(task_dir),
            )
            for idx, task_dir in enumerate(task_dirs)
        ]

    @reward(weight=1.0)
    async def solved(self, task: HarborTask, trace: Trace, runtime: Runtime) -> float:
        # Stage the task's tests into the live runtime, run its harbor verifier, and
        # read back the reward it writes — runtime-opaque (write/run/read hide whether
        # that's the host fs or across a container boundary), so it scores under any harness.
        await runtime.write("/tmp/tests.tgz", make_tar(Path(task.task_dir) / "tests"))
        await runtime.run(
            [
                "sh",
                "-c",
                "mkdir -p /logs/verifier /tests && tar -xzf /tmp/tests.tgz -C /tests",
            ],
            {},
        )
        await runtime.run(["sh", "-c", "cd /tests && bash test.sh"], {})
        try:
            reward = (await runtime.read("/logs/verifier/reward.txt")).decode().strip()
            return float(reward or 0)
        except (ProgramError, OSError, ValueError):
            return 0.0
