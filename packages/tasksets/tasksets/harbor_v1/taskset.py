"""harbor: run a Harbor (Terminal-Bench) dataset.

`dataset` is a Harbor Hub registry id (e.g. "name", "name@version",
"org/name@ref"), downloaded + cached on first use via the `harbor` CLI
(fetched on demand with `uv` when needed). Each task dir ships task.toml +
instruction.md (+ tests/, solution/, environment/). Defaults to the registry
`hello-world` task.

The harness runs in a container and edits /app; then the task's verifier
(tests/test.sh) runs in the SAME container and the reward it writes to
/logs/verifier/reward.txt becomes the score. The verification lives on the
taskset (the `solved` reward), so a harbor task runs under ANY harness.

A task's declared [environment].docker_image becomes a first-class `Task.image`
the Environment injects into the runtime (docker/prime both pull it). A task whose
environment is only a `Dockerfile` has no pullable image — we don't build Dockerfiles
(a locally-built image isn't pullable by a remote sandbox) — so it's rejected unless
`ignore_dockerfile`, which runs it on the harness runtime's image instead. A task with
no environment at all also runs on that image, unless `require_image`.
"""

import io
import logging
import shutil
import subprocess
import sys
import tarfile
import tomllib
from pathlib import Path

from pydantic import Field

from verifiers.v1.decorators import reward
from verifiers.v1.errors import SandboxError
from verifiers.v1.runtimes import Runtime
from verifiers.v1.task import Task, TaskResources, TaskTimeout
from verifiers.v1.taskset import Taskset, TasksetConfig
from verifiers.v1.trace import Trace
from verifiers.v1.types import StrictBaseModel

CACHE = Path.home() / ".cache" / "harbor"
HARBOR_PACKAGE = "harbor==0.14.0"
logger = logging.getLogger("verifiers.v1.tasksets.harbor")


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
    whether to reject it (True) or run it on the runtime's default image (False). A task
    whose environment is a `Dockerfile` is rejected too (building Dockerfiles isn't
    supported), unless `ignore_dockerfile`."""
    ignore_dockerfile: bool = False
    """Run a task whose environment is only a `Dockerfile` on the harness runtime's image
    instead of rejecting it. The Dockerfile is NOT built, so the task scores against the
    harness image rather than its declared environment — only correct when that image already
    has what the task needs (e.g. you've pointed the runtime at the right image)."""


class Author(StrictBaseModel):
    name: str | None = None
    email: str | None = None


class HarborTask(Task):
    """A Harbor task. The base fields carry instruction.md (`prompt`), the
    resolved container `image`, the `harness_timeout`/`scoring_timeout`/`resources`
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
    if out.is_dir():
        return out

    harbor_bin = shutil.which("harbor")
    if harbor_bin is not None:
        command = [harbor_bin]
    else:
        uv_bin = shutil.which("uv")
        if uv_bin is None:
            raise RuntimeError(
                "Harbor datasets require an installed `harbor` CLI or `uv` for automatic installation"
            )
        logger.info("harbor: installing %s with uv", HARBOR_PACKAGE)
        command = [uv_bin, "tool", "run"]
        # Harbor requires Python 3.12, while Verifiers also supports Python 3.11.
        if sys.version_info[:2] == (3, 11):
            command.extend(["--python", "3.12"])
        command.extend(["--from", HARBOR_PACKAGE, "harbor"])
    subprocess.run(
        [*command, "download", dataset, "--export", "-o", str(out)], check=True
    )
    return out


def resolve_image(
    task_dir: Path,
    config: dict,
    require_image: bool,
    ignore_dockerfile: bool = False,
) -> str | None:
    """The task's declared registry image (usable by docker or prime). A pullable
    `[environment].docker_image` is used directly. A task whose environment is a
    `Dockerfile` is rejected — we don't build Dockerfiles, and running it on the default
    image would silently score against the wrong environment (e.g. SWE-bench's `/testbed`
    repo would be missing) — unless `ignore_dockerfile`, which returns None to run it on the
    harness runtime's image. A task with no environment at all runs on that image too, unless
    `require_image`. None means "use the runtime's own image"."""
    declared = config.get("environment", {}).get("docker_image")
    if declared:
        return declared
    if (task_dir / "environment" / "Dockerfile").exists():
        if ignore_dockerfile:
            return None
        raise ValueError(
            f"{task_dir.name}: environment is a Dockerfile, not a pullable "
            "[environment].docker_image — building Dockerfiles isn't supported, so this "
            "task can't run (it would otherwise score against the wrong default image). "
            "Pass --taskset.ignore-dockerfile to run it on the harness runtime's image instead."
        )
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


def parse_task(task_dir: Path, idx: int, harbor_config: HarborConfig) -> HarborTask:
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
            harbor_config.ignore_dockerfile,
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
        return [
            parse_task(task_dir, idx, self.config)
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
        except (SandboxError, OSError, ValueError):
            return 0.0
