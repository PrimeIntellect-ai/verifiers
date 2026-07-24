"""Harbor tasksets backed by Harbor Hub packages.

Shared verifiers grade in the agent runtime. Separate verifiers get a fresh runtime on
the same provider, with declared artifacts restored at their original paths. Verifiers
only supports pullable Harbor images unless ``ignore_dockerfile`` is set.
"""

import hashlib
import io
import json
import logging
import shutil
import subprocess
import sys
import tarfile
import tempfile
import tomllib
from collections.abc import AsyncIterator, Iterator
from contextlib import AbstractAsyncContextManager, asynccontextmanager
from functools import lru_cache
from pathlib import Path, PurePosixPath
from typing import TYPE_CHECKING, Any

from pydantic import Field

from verifiers.v1.decorators import reward
from verifiers.v1.errors import SandboxError
from verifiers.v1.runtimes import (
    DockerConfig,
    PrimeConfig,
    Runtime,
    RuntimeConfig,
    provision_runtime,
)
from verifiers.v1.task import Task, TaskData, TaskResources, TaskTimeout
from verifiers.v1.configs.taskset import TasksetConfig
from verifiers.v1.taskset import Taskset
from verifiers.v1.types import StrictBaseModel

logger = logging.getLogger(__name__)

CACHE = Path.home() / ".cache" / "harbor"
HARBOR_INSTALL_HINT = "uv sync --python 3.12 --extra harbor"
MAX_ARTIFACT_BYTES = 256 * 1024 * 1024
MAX_ARTIFACT_ARCHIVE_BYTES = 2 * MAX_ARTIFACT_BYTES
MAX_ARTIFACT_FILES = 10_000
MAX_ARTIFACT_MANIFEST_BYTES = 4 * 1024 * 1024
MAX_REWARD_BYTES = 1024 * 1024
SYSTEM_ARTIFACT_ROOTS = {
    "/",
    "/bin",
    "/etc",
    "/lib",
    "/lib64",
    "/sbin",
    "/tmp",
    "/usr",
    "/var",
}

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


class Artifact(StrictBaseModel):
    source: str
    exclude: list[str] = []


class VerifierConfig(StrictBaseModel):
    image: str | None = None
    resources: TaskResources = TaskResources()
    workdir: str | None = None
    fresh_copy: bool = False
    network_access: bool = True


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
    task_dir: str = ""
    verifier_env: dict[str, str] = {}
    verifier: VerifierConfig | None = None
    artifacts: list[Artifact] = []


class HarborTask(Task[HarborData]):
    def scoring_runtime(
        self, runtime: Runtime, runtime_policy: RuntimeConfig
    ) -> AbstractAsyncContextManager[Runtime] | None:
        verifier_data = self.data.verifier
        if verifier_data is None:
            return None
        config = verifier_runtime_config(
            runtime,
            runtime_policy,
            verifier_data,
        )
        return self._separate_scoring_runtime(runtime, config)

    @asynccontextmanager
    async def _separate_scoring_runtime(
        self, runtime: Runtime, config: RuntimeConfig
    ) -> AsyncIterator[Runtime]:
        artifacts = await self._collect_artifacts(runtime)
        # No verifier process starts until the agent runtime is confirmed gone.
        await runtime.stop_confirmed()
        async with provision_runtime(
            config, name=f"{runtime.name}-verifier"
        ) as verifier:
            await verifier.prepare_setup()
            await self._prepare_verifier(verifier, artifacts)
            await verifier.prepare_execution([])
            yield verifier

    async def _prepare_verifier(
        self,
        runtime: Runtime,
        artifacts: tuple[bytes, list[str]] | None = None,
    ) -> None:
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
        if artifacts is None:
            return
        archive, roots = artifacts
        archive_path = "/logs/verifier/artifacts.tar"
        await runtime.write(archive_path, archive)
        await run_checked(
            runtime,
            [
                "sh",
                "-c",
                "set -e; archive=$1; workdir=$2; shift 2; "
                'for path do rm -rf -- "$path"; done; '
                'mkdir -p -- "$workdir"; '
                'tar -xf "$archive" -C /; rm -f -- "$archive"',
                "sh",
                archive_path,
                runtime.config.workdir,
                *roots,
            ],
            {},
            "Harbor artifact restore",
        )

    @reward(weight=1.0)
    async def solved(self, runtime: Runtime) -> float | dict[str, float]:
        if self.data.verifier is None:
            await self._prepare_verifier(runtime)
        await run_checked(
            runtime,
            ["test", "-f", "/tests/test.sh"],
            {},
            "Harbor verifier test discovery",
        )
        await runtime.run(
            ["sh", "-c", "cd /tests && bash test.sh"], verifier_env(self.data)
        )
        if self.data.verifier is not None:
            try:
                data = json.loads(
                    await runtime.read("/logs/verifier/reward.json", MAX_REWARD_BYTES)
                )
                if type(data) in (int, float):
                    return float(data)
                if (
                    data
                    and isinstance(data, dict)
                    and all(type(value) in (int, float) for value in data.values())
                ):
                    rewards = {key: float(value) for key, value in data.items()}
                    return rewards.get("reward", rewards)
            except (SandboxError, OSError, TypeError, ValueError):
                pass
        try:
            data = await runtime.read("/logs/verifier/reward.txt", MAX_REWARD_BYTES)
            return float(data.decode().strip() or 0)
        except (SandboxError, OSError, ValueError):
            return 0.0

    async def _collect_artifacts(self, runtime: Runtime) -> tuple[bytes, list[str]]:
        """Run the artifact hook, then create and validate one bounded archive."""
        await run_checked(
            runtime,
            ["sh", "-c", "rm -rf /logs/verifier && mkdir -p /logs/verifier"],
            {},
            "Harbor artifact staging setup",
        )
        script = Path(self.data.task_dir) / "pre_artifacts.sh"
        if script.is_file():
            script_path = "/logs/verifier/pre-artifacts.sh"
            await runtime.write(script_path, script.read_bytes())
            result = await runtime.run(["bash", script_path], {})
            if result.exit_code != 0:
                logger.warning(
                    "%s: pre_artifacts.sh exited %d — grading whatever artifacts exist: %s",
                    self.data.name,
                    result.exit_code,
                    result.stderr.strip(),
                )

        workdir = PurePosixPath(runtime.config.workdir)
        artifacts: list[Artifact] = []
        protected = (PurePosixPath("/tests"), PurePosixPath("/logs/verifier"))
        for artifact in self.data.artifacts:
            source = PurePosixPath(artifact.source)
            if not source.is_absolute():
                source = workdir / source
            source = PurePosixPath(f"/{source.as_posix().lstrip('/')}")
            source_text = source.as_posix()
            if (
                ".." in source.parts
                or source_text in SYSTEM_ARTIFACT_ROOTS
                or any(
                    source.is_relative_to(target) or target.is_relative_to(source)
                    for target in protected
                )
            ):
                raise ValueError(
                    f"task {self.data.name!r} has unsafe artifact source "
                    f"{source_text!r}"
                )
            if any(
                source.is_relative_to(other := PurePosixPath(entry.source))
                or other.is_relative_to(source)
                for entry in artifacts
            ):
                continue
            artifacts.append(artifact.model_copy(update={"source": source_text}))

        archive_path = "/logs/verifier/artifacts.tar"
        await runtime.write(archive_path, b"\0" * tarfile.RECORDSIZE)
        for artifact in artifacts:
            prefix = artifact.source.lstrip("/").replace("\\", "\\\\")
            prefix = prefix.replace("&", "\\&").replace("|", "\\|")
            result = await runtime.run(
                [
                    "sh",
                    "-c",
                    "source=$1; archive=$2; path=$3; transform=$4; shift 4; "
                    'if [ ! -e "$source" ] && [ ! -L "$source" ]; then exit; fi; '
                    'if [ -d "$source" ]; then '
                    'exec tar -rf "$archive" --format=posix --hard-dereference '
                    '"$@" "$transform" -C "$source" -- .; fi; '
                    'exec tar -rf "$archive" --format=posix --hard-dereference '
                    '-C / -- "$path"',
                    "sh",
                    artifact.source,
                    archive_path,
                    artifact.source.lstrip("/"),
                    f"--transform=s|^\\.|{prefix}|",
                    *(f"--exclude={pattern}" for pattern in artifact.exclude),
                ],
                {"LC_ALL": "C"},
            )
            if result.exit_code or result.stderr:
                detail = (result.stderr or result.stdout).strip()
                raise SandboxError(
                    f"Harbor artifact collection for {artifact.source!r} "
                    f"failed: {detail}"
                )
        data = await runtime.read(archive_path, MAX_ARTIFACT_ARCHIVE_BYTES)
        await runtime.run(["rm", "-f", archive_path], {})

        roots = [artifact.source for artifact in artifacts]
        allowed = [PurePosixPath(root.lstrip("/")) for root in roots]
        seen: set[str] = set()
        total_bytes = manifest_bytes = 0
        try:
            with tarfile.open(fileobj=io.BytesIO(data), mode="r:") as archive:
                for member in archive:
                    name = member.name
                    path = PurePosixPath(name)
                    if (
                        not name
                        or path.is_absolute()
                        or ".." in path.parts
                        or path.as_posix() != name
                        or not any(path.is_relative_to(root) for root in allowed)
                        or name in seen
                    ):
                        raise SandboxError(
                            f"Harbor artifact archive contains unsafe path {name!r}"
                        )
                    if not (member.isfile() or member.isdir()):
                        raise SandboxError(
                            f"Harbor artifact {name!r} must be a regular file or directory"
                        )
                    seen.add(name)
                    manifest_bytes += len(name.encode()) + 1
                    if member.isfile():
                        total_bytes += member.size
                    if (
                        manifest_bytes > MAX_ARTIFACT_MANIFEST_BYTES
                        or len(seen) > MAX_ARTIFACT_FILES
                        or total_bytes > MAX_ARTIFACT_BYTES
                    ):
                        raise SandboxError("Harbor artifacts exceed transfer limits")
        except (tarfile.TarError, UnicodeError) as exc:
            raise SandboxError(f"invalid Harbor artifact archive: {exc}") from exc
        return data, roots


def verifier_runtime_config(
    runtime: Runtime,
    runtime_policy: RuntimeConfig,
    verifier: VerifierConfig,
) -> RuntimeConfig:
    """Derive a separate verifier runtime from the agent runtime."""
    config = runtime.config
    if not isinstance(config, (DockerConfig, PrimeConfig)) or type(
        runtime_policy
    ) is not type(config):
        raise ValueError(
            "separate verification needs matching Docker or Prime runtime policies"
        )
    updates: dict[str, Any] = {
        "allow": runtime_policy.allow if verifier.network_access else [],
        "block": runtime_policy.block if verifier.network_access else [],
    }
    if not verifier.fresh_copy:
        updates.update(
            {
                field: getattr(runtime_policy, field)
                for field in ("workdir", "cpu", "memory", "gpu", "disk")
            }
        )
    if verifier.image is not None:
        updates["image"] = verifier.image
    updates.update(verifier.resources.model_dump(exclude_none=True))
    if verifier.workdir is not None:
        updates["workdir"] = verifier.workdir
    return type(config).model_validate({**config.model_dump(), **updates})


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
            "Pass --env.taskset.ignore-dockerfile to run it on the harness runtime's image instead."
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
    task_dir: Path, config: "HarborTaskConfig", harbor_config: HarborConfig
) -> dict[str, Any]:
    """Parse Harbor's shared or separate verifier configuration."""
    from harbor.models.task.artifacts import with_convention_entry
    from harbor.models.task.config import NetworkMode, TaskOS
    from harbor.models.task.verifier_mode import (
        VerifierEnvironmentMode,
        resolve_effective_verifier_env_config,
        resolve_task_verifier_mode,
    )

    mode = resolve_task_verifier_mode(config)
    if mode == VerifierEnvironmentMode.SHARED:
        return {"verifier_env": dict(config.verifier.env)}
    if config.verifier.collect or config.verifier.user is not None:
        raise ValueError(f"{task_dir.name}: unsupported separate verifier fields")

    environment = resolve_effective_verifier_env_config(config, None)
    if environment is None:
        raise ValueError(f"{task_dir.name}: separate verifier environment is missing")
    explicit_environment = config.verifier.environment is not None
    if (
        explicit_environment
        and environment.docker_image is None
        and not harbor_config.ignore_dockerfile
    ):
        raise ValueError(
            f"{task_dir.name}: explicit verifier environments require a docker_image"
        )
    if environment.os != TaskOS.LINUX or any(
        getattr(environment, field)
        for field in ("healthcheck", "mcp_servers", "skills_dir", "gpu_types", "tpu")
    ):
        raise ValueError(f"{task_dir.name}: unsupported separate verifier environment")

    network_mode = config.verifier.network_mode or environment.network_mode
    if (
        network_mode == NetworkMode.ALLOWLIST
        or config.verifier.allowed_hosts
        or environment.allowed_hosts
    ):
        raise ValueError(
            f"{task_dir.name}: verifier network allowlists aren't supported"
        )

    entries = with_convention_entry(
        config.artifacts, convention_source="/logs/artifacts"
    )
    if any(entry.service not in (None, "main") for entry in entries):
        raise ValueError(f"{task_dir.name}: artifact sidecars aren't supported")
    artifacts = [
        Artifact(source=entry.source, exclude=entry.exclude or []) for entry in entries
    ]

    return {
        "artifacts": artifacts,
        "verifier": VerifierConfig(
            image=environment.docker_image if explicit_environment else None,
            resources=(
                parse_resources(environment, harbor_config.resource_multiplier)
                if explicit_environment
                else TaskResources()
            ),
            workdir=environment.workdir if explicit_environment else None,
            fresh_copy=not explicit_environment,
            network_access=network_mode == NetworkMode.PUBLIC,
        ),
        "verifier_env": dict(environment.env) | dict(config.verifier.env),
    }


def parse_task(task_dir: Path, idx: int, harbor_config: HarborConfig) -> HarborData:
    # Harbor is optional, so importing its schema is deferred until a Harbor task loads.
    from harbor.models.task.config import NetworkMode, TaskConfig as HarborTaskConfig

    config = tomllib.loads((task_dir / "task.toml").read_text())
    parsed = HarborTaskConfig.model_validate(config)
    if parsed.steps:
        raise ValueError(f"{task_dir.name}: Harbor multi-step tasks aren't supported")
    network = (
        parsed.agent.explicit_phase_policy() or parsed.environment.resolve_baseline()
    )
    task, meta = config.get("task", {}), config.get("metadata", {})
    authors = [Author(**a) for a in task.get("authors", [])]
    # Older registry entries stored one author in [metadata].
    if not authors and meta.get("author_name"):
        authors = [Author(name=meta["author_name"], email=meta.get("author_email"))]
    if harbor_config.ignore_timeouts:
        harness_timeout = scoring_timeout = None
    else:
        harness_timeout = parsed.agent.timeout_sec
        scoring_timeout = parsed.verifier.timeout_sec
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
        network_allow=(
            ["*"]
            if network.network_mode == NetworkMode.PUBLIC
            else list(network.allowed_hosts)
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
            parsed.environment, harbor_config.resource_multiplier
        ),
        keywords=task.get("keywords", []),
        authors=authors,
        difficulty=meta.get("difficulty"),
        category=meta.get("category"),
        tags=meta.get("tags", []),
        task_dir=str(task_dir),
        **parse_verifier(task_dir, parsed, harbor_config),
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
