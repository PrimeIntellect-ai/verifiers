"""Harbor tasksets backed by Harbor Hub packages.

Shared verifiers run in the agent runtime. Separate verifiers collect only Harbor's
declared artifacts, stop the agent runtime, start a clean verifier runtime, restore the
artifacts at their original paths, and run the verifier there.

A pullable ``[environment].docker_image`` becomes ``TaskData.image``. Verifiers does
not build Dockerfile-only environments, so those are rejected unless ``ignore_dockerfile``
deliberately uses the harness runtime image. Tasks without an environment also use that
image unless ``require_image`` is set.
"""

import contextlib
import gzip
import hashlib
import io
import json
import math
import shlex
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
from typing import BinaryIO

from pydantic import Field

from verifiers.v1.decorators import reward
from verifiers.v1.errors import SandboxError
from verifiers.v1.runtimes import (
    Runtime,
    RuntimeConfig,
    SubprocessConfig,
    make_runtime,
)
from verifiers.v1.task import Task, TaskData, TaskResources, TaskTimeout
from verifiers.v1.taskset import Taskset, TasksetConfig
from verifiers.v1.trace import Trace
from verifiers.v1.types import StrictBaseModel

CACHE = Path.home() / ".cache" / "harbor"
HARBOR_INSTALL_HINT = "uv sync --python 3.12 --extra harbor"
MAX_ARTIFACT_ARCHIVE_BYTES = 256 * 1024 * 1024
MAX_ARTIFACT_CONTENT_BYTES = 1024 * 1024 * 1024
MAX_ARTIFACT_MEMBERS = 100_000
MAX_ARTIFACT_TAR_BYTES = MAX_ARTIFACT_CONTENT_BYTES + MAX_ARTIFACT_ARCHIVE_BYTES
MAX_ARTIFACT_HEADERS = MAX_ARTIFACT_MEMBERS * 2 + 8
MAX_ARTIFACT_METADATA_BYTES = 1024 * 1024
MAX_PAX_RECORDS = 4096


class _LimitedTarReader:
    """Keep tarfile from turning one hostile metadata header into a huge allocation."""

    def __init__(self, file: BinaryIO) -> None:
        self.file = file

    def read(self, size: int = -1) -> bytes:
        if size < 0 or size > MAX_ARTIFACT_METADATA_BYTES + tarfile.BLOCKSIZE:
            raise SandboxError("Harbor artifact archive has oversized metadata")
        return self.file.read(size)

    def seek(self, offset: int, whence: int = io.SEEK_SET) -> int:
        return self.file.seek(offset, whence)

    def tell(self) -> int:
        return self.file.tell()

    def __getattr__(self, name: str) -> object:
        return getattr(self.file, name)


def _validate_pax_payload(payload: bytes) -> None:
    """Reject structural overrides and bound tarfile's per-record allocations."""
    position = 0
    records = 0
    while position < len(payload) and payload[position]:
        separator = payload.find(b" ", position)
        if separator < 0:
            raise SandboxError("Harbor artifact archive has invalid PAX metadata")
        try:
            length = int(payload[position:separator])
        except ValueError as exc:
            raise SandboxError(
                "Harbor artifact archive has invalid PAX metadata"
            ) from exc
        end = position + length
        if length < 5 or end > len(payload) or payload[end - 1] != 0x0A:
            raise SandboxError("Harbor artifact archive has invalid PAX metadata")
        key, equals, value = payload[separator + 1 : end - 1].partition(b"=")
        if not key or not equals:
            raise SandboxError("Harbor artifact archive has invalid PAX metadata")
        records += 1
        if records > MAX_PAX_RECORDS:
            raise SandboxError("Harbor artifact archive has too many PAX records")
        if key == b"size":
            raise SandboxError(
                "Harbor artifact archive uses an unsupported PAX size override"
            )
        if (
            key.startswith(b"GNU.sparse.")
            or key == b"SCHILY.realsize"
            or (key == b"SCHILY.filetype" and value == b"sparse")
        ):
            raise SandboxError("Harbor artifact archive uses unsupported sparse files")
        position = end


def _validate_tar_archive(file: BinaryIO) -> None:
    """Bound raw extensions before tarfile recursively interprets them."""
    file.seek(0, io.SEEK_END)
    archive_size = file.tell()
    file.seek(0)
    headers = 0
    consecutive_extensions = 0
    extensions = {
        tarfile.XHDTYPE,
        tarfile.XGLTYPE,
        tarfile.SOLARIS_XHDTYPE,
        tarfile.GNUTYPE_LONGNAME,
        tarfile.GNUTYPE_LONGLINK,
    }
    while file.tell() < archive_size:
        header = file.read(tarfile.BLOCKSIZE)
        if len(header) != tarfile.BLOCKSIZE:
            raise SandboxError("Harbor artifact archive has a truncated header")
        if not any(header):
            break
        try:
            member = tarfile.TarInfo.frombuf(header, "utf-8", "surrogateescape")
        except tarfile.TarError as exc:
            raise SandboxError("Harbor artifact archive has an invalid header") from exc
        if member.size < 0:
            raise SandboxError("Harbor artifact archive has a negative entry size")
        headers += 1
        if headers > MAX_ARTIFACT_HEADERS:
            raise SandboxError("Harbor artifact archive has too many headers")
        if member.type == tarfile.GNUTYPE_SPARSE:
            raise SandboxError("Harbor artifact archive uses unsupported sparse files")

        stored_size = (
            member.size
            if member.type in extensions
            or member.isreg()
            or member.type not in tarfile.SUPPORTED_TYPES
            else 0
        )
        data_end = file.tell() + stored_size
        padded_end = data_end + (-stored_size % tarfile.BLOCKSIZE)
        if padded_end > archive_size:
            raise SandboxError("Harbor artifact archive has a truncated entry")
        if member.type in extensions:
            consecutive_extensions += 1
            if consecutive_extensions > 8:
                raise SandboxError(
                    "Harbor artifact archive has too many metadata headers"
                )
            if member.size > MAX_ARTIFACT_METADATA_BYTES:
                raise SandboxError("Harbor artifact archive has oversized metadata")
            payload = file.read(member.size)
            if member.type in {
                tarfile.XHDTYPE,
                tarfile.XGLTYPE,
                tarfile.SOLARIS_XHDTYPE,
            }:
                _validate_pax_payload(payload)
        else:
            consecutive_extensions = 0
        file.seek(padded_end)
    file.seek(0)


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


class HarborArtifact(StrictBaseModel):
    source: str


class HarborVerifier(StrictBaseModel):
    separate: bool = False
    image: str | None = None
    workdir: str | None = None
    resources: TaskResources = TaskResources()
    network_access: bool = True
    env: dict[str, str] = {}
    artifacts: list[HarborArtifact] = []
    upload_tests: bool = False


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
    verifier: HarborVerifier = Field(default_factory=HarborVerifier, exclude=True)
    """Resolved verifier mode and runtime data, kept host-side for scoring."""


class HarborTask(Task[HarborData]):
    """Run a Harbor verifier in its shared or separate runtime."""

    def scoring_runtime_config(self, base: RuntimeConfig) -> RuntimeConfig | None:
        verifier = self.data.verifier
        if not verifier.separate:
            return None

        from verifiers.v1.env import resolve_runtime_config

        data = (
            self.data
            if verifier.upload_tests
            else TaskData(
                idx=self.data.idx,
                prompt=None,
                image=verifier.image,
                workdir=verifier.workdir,
                resources=verifier.resources,
            )
        )
        config = resolve_runtime_config(base, data)
        if isinstance(config, SubprocessConfig):
            raise ValueError(
                "separate Harbor verification needs a docker, prime, or modal runtime"
            )
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

        artifacts, paths = await self._collect_artifacts(runtime)
        # The agent runtime must be gone before any verifier process starts.
        await runtime.stop_confirmed()
        target = make_runtime(scoring_runtime_config, name=f"{runtime.name}-verifier")
        try:
            await target.start()
            await self._prepare_verifier(
                target,
                artifacts,
                paths,
                getattr(scoring_runtime_config, "workdir", "/"),
            )
            await super().score(trace, target)
        finally:
            await target.stop()

    async def _collect_artifacts(self, runtime: Runtime) -> tuple[bytes, list[str]]:
        buffer = tempfile.SpooledTemporaryFile(max_size=8 * 1024 * 1024)
        seen: set[str] = set()
        paths: list[str] = []
        content_size = 0
        member_count = 0
        try:
            await run_checked(
                runtime,
                [
                    "sh",
                    "-c",
                    "rm -rf /logs/verifier && mkdir -p /logs/verifier",
                ],
                {},
                "Harbor artifact staging setup",
            )
            with tarfile.open(fileobj=buffer, mode="w:gz") as safe:
                for artifact in self.data.verifier.artifacts:
                    exists = await runtime.run(["test", "-e", artifact.source], {})
                    if exists.exit_code == 1:
                        continue
                    if exists.exit_code:
                        detail = (exists.stderr or exists.stdout).strip()
                        raise SandboxError(
                            f"Harbor artifact probe failed for {artifact.source!r}: {detail}"
                        )

                    archive = (
                        f"/logs/verifier/.vf-harbor-artifacts-{uuid.uuid4().hex}.tgz"
                    )
                    packed = await runtime.run(
                        [
                            "tar",
                            "-chzf",
                            archive,
                            "--exclude",
                            f"./{archive.lstrip('/')}",
                            "-C",
                            "/",
                            f"./{artifact.source.lstrip('/')}",
                        ],
                        {},
                    )
                    if packed.exit_code:
                        with contextlib.suppress(Exception):
                            await runtime.run(["rm", "-f", archive], {})
                        exists = await runtime.run(["test", "-e", artifact.source], {})
                        if exists.exit_code == 1:
                            continue
                        detail = (packed.stderr or packed.stdout).strip()
                        raise SandboxError(
                            f"Harbor artifact collection failed for "
                            f"{artifact.source!r}: {detail}"
                        )
                    try:
                        unsafe = await runtime.read(
                            archive, max_bytes=MAX_ARTIFACT_ARCHIVE_BYTES
                        )
                    finally:
                        with contextlib.suppress(Exception):
                            await runtime.run(["rm", "-f", archive], {})

                    root = PurePosixPath(artifact.source.lstrip("/"))
                    # The agent controls these bytes. Flatten links and retain only
                    # regular files/directories rooted under the declared artifact.
                    with tempfile.SpooledTemporaryFile(
                        max_size=8 * 1024 * 1024
                    ) as unpacked:
                        with gzip.GzipFile(fileobj=io.BytesIO(unsafe)) as compressed:
                            tar_size = 0
                            while chunk := compressed.read(1024 * 1024):
                                tar_size += len(chunk)
                                if tar_size > MAX_ARTIFACT_TAR_BYTES:
                                    raise SandboxError(
                                        "Harbor artifact archive exceeds the "
                                        "decompression limit"
                                    )
                                unpacked.write(chunk)
                        unpacked.seek(0)
                        _validate_tar_archive(unpacked)
                        with tarfile.open(
                            fileobj=_LimitedTarReader(unpacked), mode="r:"
                        ) as source:
                            for member in source:
                                path = PurePosixPath(member.name)
                                if (
                                    member.size < 0
                                    or path.is_absolute()
                                    or ".." in path.parts
                                    or not (path == root or root in path.parents)
                                    or not (
                                        member.isfile()
                                        or member.islnk()
                                        or member.isdir()
                                    )
                                ):
                                    raise SandboxError(
                                        "Harbor artifact collection produced unsafe "
                                        f"entry {member.name!r}"
                                    )
                                member_count += 1
                                content_size += member.size
                                if member_count > MAX_ARTIFACT_MEMBERS:
                                    raise SandboxError(
                                        "Harbor artifacts exceed the "
                                        f"{MAX_ARTIFACT_MEMBERS:,} entry limit"
                                    )
                                if content_size > MAX_ARTIFACT_CONTENT_BYTES:
                                    raise SandboxError(
                                        "Harbor artifacts exceed the "
                                        f"{MAX_ARTIFACT_CONTENT_BYTES // 1024**3} GiB "
                                        "uncompressed limit"
                                    )
                                if path.as_posix() in seen:
                                    continue
                                seen.add(path.as_posix())
                                clean = tarfile.TarInfo(path.as_posix())
                                clean.mode = member.mode & 0o777
                                clean.mtime = member.mtime
                                if member.isdir():
                                    clean.type = tarfile.DIRTYPE
                                    safe.addfile(clean)
                                    continue
                                file = source.extractfile(member)
                                if file is None:
                                    raise SandboxError(
                                        "Harbor artifact collection could not read "
                                        f"{member.name!r}"
                                    )
                                if member.islnk():
                                    with tempfile.SpooledTemporaryFile(
                                        max_size=8 * 1024 * 1024
                                    ) as linked:
                                        linked_size = 0
                                        while chunk := file.read(1024 * 1024):
                                            linked_size += len(chunk)
                                            if (
                                                content_size + linked_size
                                                > MAX_ARTIFACT_CONTENT_BYTES
                                            ):
                                                raise SandboxError(
                                                    "Harbor artifacts exceed the "
                                                    f"{MAX_ARTIFACT_CONTENT_BYTES // 1024**3} "
                                                    "GiB uncompressed limit"
                                                )
                                            linked.write(chunk)
                                        content_size += linked_size
                                        clean.size = linked_size
                                        linked.seek(0)
                                        safe.addfile(clean, linked)
                                    continue
                                clean.size = member.size
                                safe.addfile(clean, file)
                    paths.append(artifact.source)
        except (OSError, tarfile.TarError) as exc:
            buffer.close()
            raise SandboxError(
                "Harbor artifact collection produced an invalid archive"
            ) from exc
        except BaseException:
            buffer.close()
            raise
        try:
            buffer.seek(0, io.SEEK_END)
            if buffer.tell() > MAX_ARTIFACT_ARCHIVE_BYTES:
                raise SandboxError(
                    "Harbor artifacts exceed the "
                    f"{MAX_ARTIFACT_ARCHIVE_BYTES // 1024**2} MiB archive limit"
                )
            buffer.seek(0)
            return buffer.read(), paths
        finally:
            buffer.close()

    async def _prepare_verifier(
        self, runtime: Runtime, artifacts: bytes, paths: list[str], workdir: str
    ) -> None:
        await run_checked(
            runtime,
            [
                "sh",
                "-c",
                "rm -rf /logs/verifier && mkdir -p /logs/verifier /logs/artifacts",
            ],
            {},
            "Harbor verifier setup",
        )
        archive = f"/logs/verifier/.vf-harbor-artifacts-{uuid.uuid4().hex}.tgz"
        if paths:
            command = "rm -rf " + " ".join(shlex.quote(path) for path in paths)
            command += f" && mkdir -p {shlex.quote(workdir)}"
            await run_checked(
                runtime,
                ["sh", "-c", command],
                {},
                "Harbor artifact target cleanup",
            )
        await runtime.write(archive, artifacts)
        await run_checked(
            runtime,
            [
                "sh",
                "-c",
                f"tar -xzf {shlex.quote(archive)} -C / && rm -f {shlex.quote(archive)}",
            ],
            {},
            "Harbor artifact restore",
        )
        if self.data.verifier.upload_tests:
            tests = f"/logs/verifier/.vf-harbor-tests-{uuid.uuid4().hex}.tgz"
            await runtime.write(tests, make_tar(Path(self.data.task_dir) / "tests"))
            await run_checked(
                runtime,
                [
                    "sh",
                    "-c",
                    "rm -rf /tests && mkdir -p /tests && "
                    f"tar -xzf {shlex.quote(tests)} -C /tests && "
                    f"rm -f {shlex.quote(tests)}",
                ],
                {},
                "Harbor test upload",
            )

    @reward(weight=1.0)
    async def solved(self, runtime: Runtime) -> float | dict[str, float]:
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
            [
                "sh",
                "-c",
                "cd /tests && bash test.sh > /logs/verifier/test-stdout.txt 2>&1",
            ],
            verifier_env(self.data),
        )
        # Harbor gives structured rewards precedence when both files exist.
        try:
            raw = (await runtime.read("/logs/verifier/reward.json")).decode()
        except UnicodeDecodeError:
            return 0.0
        except (SandboxError, OSError):
            raw = None
        if raw is not None:
            try:
                values = json.loads(raw)
                if not isinstance(values, dict):
                    return 0.0
                rewards = {
                    str(key): float(value)
                    for key, value in values.items()
                    if isinstance(value, (int, float)) and not isinstance(value, bool)
                }
                return (
                    rewards
                    if len(rewards) == len(values)
                    and all(math.isfinite(value) for value in rewards.values())
                    else 0.0
                )
            except (TypeError, ValueError, OverflowError):
                return 0.0
        try:
            raw = (await runtime.read("/logs/verifier/reward.txt")).decode().strip()
            value = float(raw)
            return value if math.isfinite(value) else 0.0
        except (SandboxError, OSError, ValueError, OverflowError):
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


def parse_verifier(
    task_dir: Path, config: dict, resource_multiplier: float
) -> HarborVerifier:
    environment = config.get("environment", {})
    verifier = config.get("verifier", {})
    mode = verifier.get("environment_mode")
    verifier_environment = verifier.get("environment")
    if mode not in (None, "shared", "separate"):
        raise ValueError(f"{task_dir.name}: invalid verifier environment_mode {mode!r}")
    if mode == "shared" and verifier_environment is not None:
        raise ValueError(
            f"{task_dir.name}: [verifier.environment] is incompatible with "
            "environment_mode='shared'"
        )

    separate = mode == "separate" or verifier_environment is not None
    if not separate:
        return HarborVerifier(env=verifier.get("env", {}))
    if verifier.get("collect"):
        raise ValueError(
            f"{task_dir.name}: [[verifier.collect]] needs compose services, which "
            "Verifiers runtimes do not support"
        )
    if verifier.get("user") is not None:
        raise ValueError(
            f"{task_dir.name}: [verifier].user is not supported for separate runtimes"
        )

    explicit_environment = verifier_environment is not None
    verifier_environment = verifier_environment if explicit_environment else environment
    if verifier_environment.get("os", "linux").lower() != "linux":
        raise ValueError(
            f"{task_dir.name}: separate Harbor verification supports Linux images only"
        )
    if explicit_environment and not verifier_environment.get("docker_image"):
        raise ValueError(
            f"{task_dir.name}: [verifier.environment] needs a pullable docker_image; "
            "building tests/Dockerfile is not supported"
        )
    if not explicit_environment and (task_dir / "tests" / "Dockerfile").exists():
        raise ValueError(
            f"{task_dir.name}: tests/Dockerfile needs a pullable "
            "[verifier.environment].docker_image; building verifier Dockerfiles "
            "is not supported"
        )
    unsupported = [
        field
        for field in ("healthcheck", "mcp_servers", "skills_dir", "gpu_types", "tpu")
        if verifier_environment.get(field)
    ]
    if unsupported:
        raise ValueError(
            f"{task_dir.name}: separate verifier environment fields are not supported: "
            + ", ".join(unsupported)
        )

    network_mode = verifier.get("network_mode")
    if network_mode is None:
        network_mode = verifier_environment.get("network_mode")
    if network_mode is None:
        network_mode = (
            "public"
            if verifier_environment.get("allow_internet", True)
            else "no-network"
        )
    if network_mode == "allowlist":
        raise ValueError(
            f"{task_dir.name}: verifier network_mode='allowlist' is not supported"
        )
    if verifier.get("allowed_hosts") is not None or verifier_environment.get(
        "allowed_hosts"
    ):
        raise ValueError(
            f"{task_dir.name}: verifier network allowlists are not supported"
        )
    if network_mode not in ("public", "no-network"):
        raise ValueError(
            f"{task_dir.name}: invalid verifier network_mode {network_mode!r}"
        )

    raw_artifacts = config.get("artifacts", [])
    if not isinstance(raw_artifacts, list):
        raise ValueError(f"{task_dir.name}: artifacts must be a list")
    raw_artifacts = ["/logs/artifacts", *raw_artifacts]
    artifacts: list[HarborArtifact] = []
    for raw in raw_artifacts:
        artifact = {"source": raw} if isinstance(raw, str) else raw
        if not isinstance(artifact, dict) or not isinstance(
            artifact.get("source"), str
        ):
            raise ValueError(f"{task_dir.name}: invalid artifact entry {raw!r}")
        if artifact.get("service") not in (None, "main"):
            raise ValueError(f"{task_dir.name}: sidecar artifacts are not supported")
        if artifact.get("destination"):
            raise ValueError(
                f"{task_dir.name}: artifact destinations are not supported"
            )
        if artifact.get("exclude"):
            raise ValueError(
                f"{task_dir.name}: artifact exclude patterns are not supported"
            )
        source = artifact["source"]
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

    env = verifier_environment.get("env", {}) | verifier.get("env", {})
    return HarborVerifier(
        separate=True,
        image=verifier_environment.get("docker_image"),
        workdir=verifier_environment.get("workdir"),
        resources=parse_resources(verifier_environment, resource_multiplier),
        network_access=network_mode == "public",
        env=env,
        artifacts=artifacts,
        upload_tests=not explicit_environment,
    )


def parse_task(task_dir: Path, idx: int, harbor_config: HarborConfig) -> HarborData:
    config = tomllib.loads((task_dir / "task.toml").read_text())
    if config.get("steps"):
        raise ValueError(f"{task_dir.name}: Harbor multi-step tasks are not supported")
    task, meta = config.get("task", {}), config.get("metadata", {})
    authors = [Author(**a) for a in task.get("authors", [])]
    # Older registry entries stored one author in [metadata].
    if not authors and meta.get("author_name"):
        authors = [Author(name=meta["author_name"], email=meta.get("author_email"))]
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
        workdir=config.get("environment", {}).get("workdir"),
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
        verifier=parse_verifier(task_dir, config, harbor_config.resource_multiplier),
    )


def verifier_env(task: HarborData) -> dict[str, str]:
    """Resolve templates at scoring time so host secrets are never serialized."""
    if not task.verifier.env:
        return {}

    # Harbor is an optional dependency, so importing this module must still work
    # for users who do not install the Harbor extra.
    from harbor.utils.env import resolve_env_vars

    return resolve_env_vars(task.verifier.env)


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
