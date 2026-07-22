"""Harbor tasksets backed by Harbor Hub packages.

The Harbor CLI downloads and caches each task directory. Its verifier runs in the
same runtime the harness edited, then writes the score to
``/logs/verifier/reward.txt``.

A pullable ``[environment].docker_image`` becomes ``TaskData.image``. Verifiers does
not build Dockerfile-only environments, so those are rejected unless ``ignore_dockerfile``
deliberately uses the harness runtime image. Tasks without an environment also use that
image unless ``require_image`` is set.
"""

import hashlib
import io
import shutil
import subprocess
import sys
import tarfile
import tempfile
import tomllib
from collections.abc import Iterator
from functools import lru_cache
from pathlib import Path

from pydantic import Field

from verifiers.v1.decorators import reward
from verifiers.v1.errors import SandboxError
from verifiers.v1.runtimes import Runtime
from verifiers.v1.task import Task, TaskData, TaskResources, TaskTimeout
from verifiers.v1.taskset import Taskset, TasksetConfig
from verifiers.v1.types import StrictBaseModel

CACHE = Path.home() / ".cache" / "harbor"
HARBOR_INSTALL_HINT = "uv sync --python 3.12 --extra harbor"


class HarborConfig(TasksetConfig):
    dataset: str = "harbor/hello-world"
    """A Harbor Hub package id ("org/name" or "org/name@ref"), where ref is a
    tag, integer revision, or sha256 digest. Legacy registries selected with `repo`,
    `registry_path`, or `registry_url` use a bare dataset name ("name" or "name@version")."""
    repo: str | None = None
    """Optional Harbor `--repo` registry selector, e.g. "org/repo@ref"."""
    registry_path: Path | None = None
    """Optional Harbor `--registry-path` selector. Local unless `repo` is also set."""
    registry_url: str | None = None
    """Optional Harbor `--registry-url` selector for a raw registry.json URL."""
    tasks: list[str] | None = None
    """Optional subset of task names to load (None = all)."""
    ignore_timeouts: bool = True
    """Drop each task's declared agent and verifier timeouts so rollouts run
    unbounded (unless run-level `--timeout.*` limits are set). Task timeouts are
    authored against Harbor's runtime and confound model capability with inference
    speed; set False to apply them anyway."""
    timeout_multiplier: float = Field(1.0, gt=0)
    """Scale each task's agent and verifier timeouts. Only applies with
    `ignore_timeouts=False`."""
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


class HarborData(TaskData):
    """Parsed ``task.toml`` metadata plus the host-side verifier directory.

    Base ``TaskData`` fields hold the prompt, resolved image, timeout, resources,
    name, and description. The remaining fields mirror Harbor metadata.
    """

    keywords: list[str] = []
    authors: list[Author] = []
    difficulty: str | None = None
    category: str | None = None
    tags: list[str] = []
    task_dir: str = Field("", exclude=True)
    """Host path to the task dir; used to stage tests/ to verify, not serialized."""
    verifier_env: dict[str, str] = {}
    """Raw [verifier.env] entries (literals or `${VAR}`/`${VAR:-default}` templates).
    Resolved against the host environment at scoring time, like `harbor run` — so a
    verifier that needs judge API keys or configuration actually receives them."""


class HarborTask(Task[HarborData]):
    """Stage and run Harbor's verifier inside the task's live runtime."""

    @reward(weight=1.0)
    async def solved(self, runtime: Runtime) -> float:
        await runtime.write(
            "/tmp/tests.tgz", make_tar(Path(self.data.task_dir) / "tests")
        )
        await runtime.run(
            [
                "sh",
                "-c",
                "mkdir -p /logs/verifier /tests && tar -xzf /tmp/tests.tgz -C /tests",
            ],
            {},
        )
        await runtime.run(
            ["sh", "-c", "cd /tests && bash test.sh"], verifier_env(self.data)
        )
        try:
            reward = (await runtime.read("/logs/verifier/reward.txt")).decode().strip()
            return float(reward or 0)
        except (SandboxError, OSError, ValueError):
            return 0.0


def harbor_cli() -> str:
    scripts_dir = Path(sys.executable).parent
    harbor_bin = shutil.which("harbor", path=str(scripts_dir))
    if harbor_bin is None:
        raise RuntimeError(
            "Harbor tasksets require the Harbor CLI from the `harbor` extra. "
            f"Install it with: `{HARBOR_INSTALL_HINT}`"
        )
    return harbor_bin


def cache_dir(config: HarborConfig) -> Path:
    selector_parts = [config.dataset]
    if config.repo is not None:
        selector_parts.extend(("repo", config.repo))
    if config.registry_path is not None:
        registry_path = (
            config.registry_path
            if config.repo is not None
            else config.registry_path.expanduser().resolve()
        )
        selector_parts.extend(("registry_path", str(registry_path)))
    if config.registry_url is not None:
        selector_parts.extend(("registry_url", config.registry_url))

    name = config.dataset.replace("/", "_").replace("@", "_")
    if len(selector_parts) > 1:
        digest = hashlib.sha256("\0".join(selector_parts).encode()).hexdigest()[:12]
        name = f"{name}_{digest}"
    return CACHE / name


def download_command(config: HarborConfig, output_dir: Path) -> list[str]:
    command = [
        harbor_cli(),
        "download",
        config.dataset,
        "--export",
        "-o",
        str(output_dir),
    ]
    if config.repo is not None:
        command.extend(["--repo", config.repo])
    if config.registry_path is not None:
        registry_path = (
            config.registry_path
            if config.repo is not None
            else config.registry_path.expanduser()
        )
        command.extend(["--registry-path", str(registry_path)])
    if config.registry_url is not None:
        command.extend(["--registry-url", config.registry_url])
    return command


def dataset_dir(config: HarborConfig) -> Path:
    """Download/cache a Hub or legacy-registry package selected by the config."""
    out = cache_dir(config)
    if out.is_dir():
        return out

    CACHE.mkdir(parents=True, exist_ok=True)
    # Publish only a complete CLI export to the cache.
    with tempfile.TemporaryDirectory(dir=CACHE) as temp:
        export_dir = Path(temp) / "export"
        command = download_command(config, export_dir)
        try:
            subprocess.run(command, check=True, capture_output=True, text=True)
        except subprocess.CalledProcessError as exc:
            message = (
                f"Harbor download failed for {config.dataset!r} with exit code "
                f"{exc.returncode}"
            )
            outputs = [
                output.strip()
                for output in (exc.stdout, exc.stderr)
                if isinstance(output, str) and output.strip()
            ]
            if output := "\n".join(outputs):
                message = f"{message}:\n{output}"
            raise RuntimeError(message) from exc
        try:
            export_dir.rename(out)
        except OSError:
            if out.is_dir():
                return out
            raise
    return out


def resolve_image(
    task_dir: Path,
    config: dict,
    require_image: bool,
    ignore_dockerfile: bool = False,
) -> str | None:
    """Choose a pullable image without silently ignoring a declared Dockerfile.

    ``None`` tells the runtime to keep the harness image. That is the intended
    fallback for tasks with no environment, but would score a Dockerfile task in
    the wrong environment unless the user explicitly opts in.
    """
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
            "Pass --env.taskset.ignore-dockerfile to run it on the harness runtime's image instead."
        )
    if require_image:
        raise ValueError(
            f"{task_dir.name}: no [environment].docker_image and require_image=True"
        )
    return None


def size_to_mb(size: str | int | float) -> float:
    """A Harbor size in MB, from either schema: current integer-MB fields or the
    legacy schema-1.0 size strings ("8G", "512M", "64K")."""
    if not isinstance(size, str):
        return float(size)
    scale = {"G": 1024.0, "M": 1.0, "K": 1 / 1024}.get(size.strip().upper()[-1:])
    if scale is None:
        raise ValueError(
            f"invalid Harbor size {size!r}: expected a number of MB or a "
            "'<number>[G|M|K]' string"
        )
    return float(size.strip()[:-1]) * scale


def parse_resources(env: dict, multiplier: float = 1.0) -> TaskResources:
    # Harbor's current schema reports memory/storage as integer-MB fields; the
    # legacy schema 1.0 used size strings under `memory`/`storage`, which Harbor
    # still migrates (datasets like senior-swe-bench are authored against it).
    # TaskResources stores GB. A zero GPU count means no GPU request rather than
    # the string "0".
    memory = env.get("memory_mb") or env.get("memory")
    disk = env.get("storage_mb") or env.get("storage")
    return TaskResources(
        cpu=env["cpus"] * multiplier if env.get("cpus") else None,
        memory=size_to_mb(memory) / 1024 * multiplier if memory else None,
        gpu=str(env["gpus"]) if env.get("gpus") else None,
        disk=size_to_mb(disk) / 1024 * multiplier if disk else None,
    )


def parse_task(task_dir: Path, idx: int, harbor_config: HarborConfig) -> HarborData:
    config = tomllib.loads((task_dir / "task.toml").read_text())
    task, meta = config.get("task", {}), config.get("metadata", {})
    authors = [Author(**a) for a in task.get("authors", [])]
    # Older registry entries stored one author in [metadata].
    if not authors and meta.get("author_name"):
        authors = [Author(name=meta["author_name"], email=meta.get("author_email"))]
    if harbor_config.ignore_timeouts:
        harness_timeout = scoring_timeout = None
    else:
        harness_timeout = config.get("agent", {}).get("timeout_sec")
        scoring_timeout = config.get("verifier", {}).get("timeout_sec")
    return HarborData(
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
        verifier_env=config.get("verifier", {}).get("env", {}),
    )


def verifier_env(task: HarborData) -> dict[str, str]:
    """Resolve templates at scoring time so host secrets are never serialized."""
    if not task.verifier_env:
        return {}

    # Harbor is an optional dependency, so importing this module must still work
    # for users who do not install the Harbor extra.
    from harbor.utils.env import resolve_env_vars

    return resolve_env_vars(task.verifier_env)


# Downloaded test directories are immutable. Cache only the latest archive to
# bound memory while reusing it across rollouts of the current task.
@lru_cache(maxsize=1)
def make_tar(directory: Path) -> bytes:
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w:gz") as tar:
        for item in sorted(directory.iterdir()):
            tar.add(item, arcname=item.name)
    return buffer.getvalue()


class HarborTaskset(Taskset[HarborTask, HarborConfig]):
    def load(self) -> Iterator[HarborTask]:
        root = dataset_dir(self.config)
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
        for idx, task_dir in enumerate(task_dirs):
            yield HarborTask(parse_task(task_dir, idx, self.config), self.config.task)
