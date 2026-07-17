"""Harbor tasksets backed by Harbor Hub packages.

Shared verifiers run in the agent runtime. Separate verifiers collect only Harbor's
declared artifacts, stop the agent runtime, start a clean verifier runtime, restore the
artifacts at their original paths, and run the verifier there.

A pullable ``[environment].docker_image`` becomes ``TaskData.image``. Verifiers does
not build Dockerfile-only environments, so those are rejected unless ``ignore_dockerfile``
deliberately uses the harness runtime image. Tasks without an environment also use that
image unless ``require_image`` is set.
"""

import asyncio
import hashlib
import io
import shutil
import subprocess
import sys
import tarfile
import tempfile
import tomllib
import uuid
from collections.abc import Iterator
from functools import lru_cache
from pathlib import Path, PurePosixPath
from typing import TYPE_CHECKING

from pydantic import Field

from verifiers.v1.decorators import reward
from verifiers.v1.errors import SandboxError
from verifiers.v1.runtimes import (
    DockerConfig,
    PrimeConfig,
    Runtime,
    RuntimeConfig,
    make_runtime,
)
from verifiers.v1.task import Task, TaskData, TaskResources, TaskTimeout
from verifiers.v1.taskset import Taskset, TasksetConfig
from verifiers.v1.trace import Trace
from verifiers.v1.types import StrictBaseModel

CACHE = Path.home() / ".cache" / "harbor"
HARBOR_INSTALL_HINT = "uv sync --python 3.12 --extra harbor"
MAX_ARTIFACT_BYTES = 256 * 1024 * 1024
MAX_ARTIFACT_FILES = 10_000
MAX_ARTIFACT_MANIFEST_BYTES = 4 * 1024 * 1024

if TYPE_CHECKING:
    from harbor.models.task.config import (
        EnvironmentConfig,
        TaskConfig as HarborTaskConfig,
    )


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


class HarborArtifact(StrictBaseModel):
    source: str


class HarborVerifier(StrictBaseModel):
    separate: bool = False
    image: str | None = None
    workdir: str | None = None
    resources: TaskResources = TaskResources()
    network_access: bool = True
    artifacts: list[HarborArtifact] = []


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
    """Raw [verifier.env] entries, resolved on the host at scoring time."""
    verifier: HarborVerifier = Field(default_factory=HarborVerifier, exclude=True)
    """Resolved verifier mode and runtime data, kept host-side for scoring."""


class HarborTask(Task[HarborData]):
    """Run a Harbor verifier in its shared or separate runtime."""

    def scoring_runtime_config(self, base: RuntimeConfig) -> RuntimeConfig | None:
        verifier = self.data.verifier
        if not verifier.separate:
            return None
        if not isinstance(base, (DockerConfig, PrimeConfig)):
            raise ValueError(
                "separate Harbor verification needs a docker or prime runtime"
            )

        from verifiers.v1.env import resolve_runtime_config

        data = TaskData(
            idx=self.data.idx,
            prompt=None,
            image=verifier.image,
            workdir=verifier.workdir,
            resources=verifier.resources,
        )
        config = resolve_runtime_config(base, data)
        return config.model_copy(update={"network_access": verifier.network_access})

    async def score(
        self,
        trace: Trace,
        runtime: Runtime | None = None,
        scoring_runtime_config: RuntimeConfig | None = None,
    ) -> None:
        if not self.data.verifier.separate:
            await super().score(trace, runtime)
            return
        if runtime is None:
            await super().score(trace)
            return
        if scoring_runtime_config is None:
            raise ValueError(
                "separate Harbor scoring requires a verifier runtime config"
            )

        with tempfile.TemporaryDirectory() as temp:
            files, directories = await self._collect_artifacts(runtime, Path(temp))
            # No verifier process starts until the agent runtime is gone.
            await runtime.stop_confirmed()
            target = make_runtime(
                scoring_runtime_config, name=f"{runtime.name}-verifier"
            )
            try:
                await target.start()
                await self._prepare_verifier(target, files, directories)
                await super().score(trace, target)
            finally:
                await target.stop()

    async def _collect_artifacts(
        self, runtime: Runtime, staging: Path
    ) -> tuple[list[tuple[str, Path]], list[str]]:
        await run_checked(
            runtime,
            ["sh", "-c", "rm -rf /logs/verifier && mkdir -p /logs/verifier"],
            {},
            "Harbor artifact staging setup",
        )
        files: list[tuple[str, Path]] = []
        directories: list[str] = []
        total_bytes = 0
        for artifact in self.data.verifier.artifacts:
            source = artifact.source
            result = await runtime.run(
                [
                    "sh",
                    "-c",
                    'if [ -L "$1" ]; then printf unsupported; '
                    'elif [ ! -e "$1" ]; then printf missing; '
                    'elif [ -f "$1" ]; then printf file; '
                    'elif [ -d "$1" ]; then printf directory; '
                    "else printf unsupported; fi",
                    "sh",
                    source,
                ],
                {},
            )
            if result.exit_code:
                detail = (result.stderr or result.stdout).strip()
                raise SandboxError(
                    f"Harbor artifact probe failed for {source!r}: {detail}"
                )
            kind = result.stdout
            if kind == "missing":
                continue
            if kind not in ("file", "directory"):
                raise SandboxError(
                    f"Harbor artifact {source!r} must be a regular file or directory"
                )

            paths = [source]
            if kind == "directory":
                directories.append(source)
                manifest = f"/logs/verifier/.vf-harbor-{uuid.uuid4().hex}"
                listed = await runtime.run(
                    [
                        "sh",
                        "-c",
                        'find "$1" -mindepth 1 ! -type d ! -type f '
                        '-print -quit > "$3" && [ ! -s "$3" ] && '
                        'find "$1" -type f -print0 > "$2"',
                        "sh",
                        source,
                        manifest,
                        f"{manifest}.bad",
                    ],
                    {},
                )
                try:
                    if listed.exit_code:
                        raise SandboxError(
                            f"Harbor artifact directory {source!r} must contain only "
                            "regular files and directories"
                        )
                    raw = await runtime.read_bounded(
                        manifest, MAX_ARTIFACT_MANIFEST_BYTES
                    )
                finally:
                    await runtime.run(["rm", "-f", manifest, f"{manifest}.bad"], {})
                try:
                    paths = [
                        value.decode()
                        for value in raw.rstrip(b"\0").split(b"\0")
                        if value
                    ]
                except UnicodeDecodeError as exc:
                    raise SandboxError(
                        f"Harbor artifact directory {source!r} has a non-UTF-8 path"
                    ) from exc

            if len(files) + len(paths) > MAX_ARTIFACT_FILES:
                raise SandboxError(
                    f"Harbor artifacts exceed the {MAX_ARTIFACT_FILES:,} file limit"
                )
            root = PurePosixPath(source)
            for path in paths:
                artifact_path = PurePosixPath(path)
                if (
                    not artifact_path.is_absolute()
                    or ".." in artifact_path.parts
                    or not (artifact_path == root or root in artifact_path.parents)
                ):
                    raise SandboxError(
                        f"Harbor artifact directory produced unsafe path {path!r}"
                    )
                data = await runtime.read_bounded(
                    artifact_path.as_posix(), MAX_ARTIFACT_BYTES - total_bytes
                )
                total_bytes += len(data)
                host_path = staging / str(len(files))
                await asyncio.to_thread(host_path.write_bytes, data)
                files.append((artifact_path.as_posix(), host_path))
        return files, directories

    async def _prepare_verifier(
        self,
        runtime: Runtime,
        files: list[tuple[str, Path]],
        directories: list[str],
    ) -> None:
        await run_checked(
            runtime,
            ["rm", "-rf", "/logs/verifier"],
            {},
            "Harbor verifier setup",
        )
        await run_checked(
            runtime,
            ["mkdir", "-p", "/logs/verifier"],
            {},
            "Harbor verifier setup",
        )
        sources = [artifact.source for artifact in self.data.verifier.artifacts]
        if sources:
            await run_checked(
                runtime,
                ["rm", "-rf", *sources],
                {},
                "Harbor artifact target cleanup",
            )
        directories = list(dict.fromkeys(["/logs/artifacts", *directories]))
        await run_checked(
            runtime,
            ["mkdir", "-p", *directories],
            {},
            "Harbor artifact directory restore",
        )
        for path, host_path in files:
            await runtime.write(path, await asyncio.to_thread(host_path.read_bytes))

    @reward(weight=1.0)
    async def solved(self, runtime: Runtime) -> float:
        if not self.data.verifier.separate:
            await runtime.write(
                "/tmp/tests.tgz", make_tar(Path(self.data.task_dir) / "tests")
            )
            await run_checked(
                runtime,
                [
                    "sh",
                    "-c",
                    "rm -rf /logs/verifier /tests && "
                    "mkdir -p /logs/verifier /tests && "
                    "tar -xzf /tmp/tests.tgz -C /tests",
                ],
                {},
                "Harbor verifier setup",
            )
        await run_checked(
            runtime,
            ["test", "-f", "/tests/test.sh"],
            {},
            "Harbor verifier test discovery",
        )
        await runtime.run(
            ["sh", "-c", "cd /tests && bash test.sh"],
            verifier_env(self.data),
        )
        try:
            reward = (await runtime.read("/logs/verifier/reward.txt")).decode().strip()
            return float(reward or 0)
        except (SandboxError, OSError, ValueError):
            return 0.0


async def run_checked(
    runtime: Runtime, argv: list[str], env: dict[str, str], action: str
) -> None:
    result = await runtime.run(argv, env)
    if result.exit_code:
        detail = (result.stderr or result.stdout).strip()
        raise SandboxError(f"{action} failed: {detail}")


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
            "Pass --taskset.ignore-dockerfile to run it on the harness runtime's image instead."
        )
    if require_image:
        raise ValueError(
            f"{task_dir.name}: no [environment].docker_image and require_image=True"
        )
    return None


def parse_resources(env: "EnvironmentConfig", multiplier: float = 1.0) -> TaskResources:
    """Convert Harbor's validated MB resources to Verifiers' GB resources."""
    return TaskResources(
        cpu=env.cpus * multiplier if env.cpus else None,
        memory=env.memory_mb / 1024 * multiplier if env.memory_mb else None,
        gpu=str(env.gpus) if env.gpus else None,
        disk=env.storage_mb / 1024 * multiplier if env.storage_mb else None,
    )


def parse_verifier(
    task_dir: Path, config: "HarborTaskConfig", resource_multiplier: float
) -> tuple[HarborVerifier, dict[str, str]]:
    # Harbor is optional, so importing this module still works without the extra.
    from harbor.models.task.config import NetworkMode, TaskOS
    from harbor.models.task.verifier_mode import (
        VerifierEnvironmentMode,
        resolve_effective_verifier_env_config,
        resolve_task_verifier_mode,
    )

    mode = resolve_task_verifier_mode(config)
    if mode == VerifierEnvironmentMode.SHARED:
        return HarborVerifier(), dict(config.verifier.env)
    if config.verifier.collect:
        raise ValueError(
            f"{task_dir.name}: [[verifier.collect]] needs compose services, which "
            "Verifiers runtimes do not support"
        )
    if config.verifier.user is not None:
        raise ValueError(
            f"{task_dir.name}: [verifier].user is not supported for separate runtimes"
        )

    environment = resolve_effective_verifier_env_config(config, None)
    if environment is None:
        raise ValueError(f"{task_dir.name}: separate verifier environment is missing")
    if config.verifier.environment is None or not environment.docker_image:
        raise ValueError(
            f"{task_dir.name}: separate verification needs a pullable "
            "[verifier.environment].docker_image"
        )
    if environment.os != TaskOS.LINUX:
        raise ValueError(
            f"{task_dir.name}: separate Harbor verification supports Linux images only"
        )
    unsupported = [
        field
        for field in ("healthcheck", "mcp_servers", "skills_dir", "gpu_types", "tpu")
        if getattr(environment, field)
    ]
    if unsupported:
        raise ValueError(
            f"{task_dir.name}: separate verifier environment fields are not supported: "
            + ", ".join(unsupported)
        )

    network_mode = config.verifier.network_mode or environment.network_mode
    if (
        network_mode == NetworkMode.ALLOWLIST
        or config.verifier.allowed_hosts
        or environment.allowed_hosts
    ):
        raise ValueError(
            f"{task_dir.name}: verifier network allowlists are not supported"
        )

    artifacts: list[HarborArtifact] = []
    for artifact in ["/logs/artifacts", *config.artifacts]:
        source = artifact if isinstance(artifact, str) else artifact.source
        if not isinstance(artifact, str) and artifact.service not in (None, "main"):
            raise ValueError(f"{task_dir.name}: sidecar artifacts are not supported")
        if not isinstance(artifact, str) and artifact.destination:
            raise ValueError(
                f"{task_dir.name}: artifact destinations are not supported"
            )
        if not isinstance(artifact, str) and artifact.exclude:
            raise ValueError(
                f"{task_dir.name}: artifact exclude patterns are not supported"
            )
        if not source.startswith("/") or ".." in PurePosixPath(source).parts:
            raise ValueError(
                f"{task_dir.name}: artifact source must be an absolute non-root path, "
                f"got {source!r}"
            )
        path = PurePosixPath(f"/{source.lstrip('/')}")
        source = path.as_posix()
        if path == PurePosixPath("/"):
            raise ValueError(
                f"{task_dir.name}: artifact source must be an absolute non-root path, "
                f"got {source!r}"
            )
        protected = (PurePosixPath("/tests"), PurePosixPath("/logs/verifier"))
        if any(
            path == target or path in target.parents or target in path.parents
            for target in protected
        ):
            raise ValueError(
                f"{task_dir.name}: artifact source {source!r} overlaps verifier-owned files"
            )
        existing = [PurePosixPath(entry.source) for entry in artifacts]
        # Harbor keeps the first declaration when artifact sources overlap.
        if any(
            path == other or path in other.parents or other in path.parents
            for other in existing
        ):
            continue
        artifacts.append(HarborArtifact(source=source))

    return (
        HarborVerifier(
            separate=True,
            image=environment.docker_image,
            workdir=environment.workdir,
            resources=parse_resources(environment, resource_multiplier),
            network_access=network_mode == NetworkMode.PUBLIC,
            artifacts=artifacts,
        ),
        dict(environment.env) | dict(config.verifier.env),
    )


def parse_task(task_dir: Path, idx: int, harbor_config: HarborConfig) -> HarborData:
    from harbor.models.task.config import TaskConfig as HarborTaskConfig

    config = tomllib.loads((task_dir / "task.toml").read_text())
    parsed = HarborTaskConfig.model_validate(config)
    if parsed.steps:
        raise ValueError(f"{task_dir.name}: Harbor multi-step tasks are not supported")
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
    verifier, verifier_environment = parse_verifier(
        task_dir, parsed, harbor_config.resource_multiplier
    )
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
        workdir=parsed.environment.workdir,
        timeout=TaskTimeout(
            harness=harness_timeout * harbor_config.timeout_multiplier
            if harness_timeout is not None
            else None,
            scoring=scoring_timeout * harbor_config.timeout_multiplier
            if scoring_timeout is not None
            else None,
        ),
        resources=parse_resources(
            parsed.environment, harbor_config.resource_multiplier
        ),
        keywords=task.get("keywords", []),
        authors=authors,
        difficulty=meta.get("difficulty"),
        category=meta.get("category"),
        tags=meta.get("tags", []),
        task_dir=str(task_dir),
        verifier_env=verifier_environment,
        verifier=verifier,
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
