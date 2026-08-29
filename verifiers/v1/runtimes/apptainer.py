"""Local Apptainer instance runtime."""

import asyncio
import contextlib
import json
import logging
import os
import re
import shutil
import signal
import subprocess
import tempfile
import threading
import uuid
from pathlib import Path, PurePosixPath
from typing import Literal, Self

from pydantic import Field, model_validator

from verifiers.v1.configs.runtime import NetworkPolicyConfig
from verifiers.v1.errors import SandboxError
from verifiers.v1.runtimes.attached import (
    AttachedProcess,
    open_attached_process,
)
from verifiers.v1.runtimes.base import (
    BaseRuntimeInfo,
    ProgramResult,
    Runtime,
)
from verifiers.v1.utils.aio import run_shielded

logger = logging.getLogger(__name__)

_CLIENT_ENV_SUFFIXES = {
    "AUTH_FILE",
    "AUTHFILE",
    "CACHEDIR",
    "CONFIGDIR",
    "DISABLE_CACHE",
    "DOCKER_HOST",
    "DOCKER_PASSWORD",
    "DOCKER_USERNAME",
    "DOWNLOAD_BUFFER_SIZE",
    "DOWNLOAD_CONCURRENCY",
    "DOWNLOAD_PART_SIZE",
    "LIBRARY",
    "TMPDIR",
}
_SINGLE_COLON_TRANSPORTS = {
    "dir",
    "docker-archive",
    "docker-daemon",
    "oci",
    "oci-archive",
}


class ApptainerConfig(NetworkPolicyConfig):
    type: Literal["apptainer"] = "apptainer"
    image: str = Field(default="python:3.11-slim", min_length=1)
    workdir: str = "/app"
    cpu: float | None = Field(default=None, ge=0.01)
    """CPU limit in cores. The host must support delegated cgroups v2."""
    memory: float | None = Field(default=None, gt=0)
    """Memory limit in GB. The host must support delegated cgroups v2."""
    gpu: Literal["nvidia:all", "rocm:all"] | None = None
    """Expose all accessible GPUs from one vendor; model/count isolation is unsupported."""
    disk: None = None
    """Apptainer has no portable unprivileged per-instance disk limit."""

    @model_validator(mode="after")
    def validate_apptainer(self) -> Self:
        if self.network_restricted:
            raise ValueError(
                "Apptainer shares the host network and supports only unrestricted "
                "networking; CNI isolation requires host administrator configuration"
            )
        path = PurePosixPath(self.workdir)
        if (
            not path.is_absolute()
            or path == path.parent
            or ".." in path.parts
            or ":" in self.workdir
            or "," in self.workdir
        ):
            raise ValueError(
                "Apptainer workdir must be a non-root absolute container path without "
                "'..', ':' or ','"
            )
        if not self.image.strip():
            raise ValueError("Apptainer image must not be blank")
        return self


class ApptainerRuntimeInfo(ApptainerConfig, BaseRuntimeInfo):
    pass


async def apptainer(
    *args: str,
    env: dict[str, str] | None = None,
    input: bytes | None = None,
    client_started: asyncio.Event | None = None,
) -> ProgramResult:
    process = await asyncio.create_subprocess_exec(
        "apptainer",
        *args,
        env=env,
        stdin=asyncio.subprocess.PIPE if input is not None else None,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        start_new_session=True,
    )
    if client_started is not None:
        client_started.set()
    try:
        stdout, stderr = await process.communicate(input=input)
    except BaseException:
        if process.returncode is None:
            with contextlib.suppress(ProcessLookupError, PermissionError):
                os.killpg(process.pid, signal.SIGKILL)
        await run_shielded(process.communicate())
        raise
    return ProgramResult(
        exit_code=process.returncode or 0,
        stdout=stdout.decode(errors="replace"),
        stderr=stderr.decode(errors="replace"),
    )


def image_reference(image: str) -> str:
    """Map Docker shorthand to its URI while preserving Apptainer-native references."""
    transport, separator, _ = image.partition(":")
    if "://" in image or (separator and transport in _SINGLE_COLON_TRANSPORTS):
        return image
    path = Path(image).expanduser()
    name = image.rsplit("/", 1)[-1]
    if image.startswith(("/", "./", "../", "~")) or (
        path.suffix.lower() in {".sif", ".sqsh", ".img"}
        and ":" not in name
        and path.is_file()
    ):
        return str(path.resolve())
    return f"docker://{image}"


class ApptainerRuntime(Runtime):
    def __init__(self, config: ApptainerConfig, name: str | None = None) -> None:
        super().__init__(name)
        # Task resources are applied with model_copy(), so revalidate here before
        # provisioning to reject unsupported disk and GPU requests.
        self.config = ApptainerConfig.model_validate(config.model_dump())
        self.info = ApptainerRuntimeInfo(**self.config.model_dump())
        # Instance names are user-global, so cleanup owns a private identifier rather
        # than the caller-supplied logical runtime name.
        self._instance = f"vf-{uuid.uuid4().hex}"
        self._tempdir: Path | None = None
        self._workspace: Path | None = None
        self._instance_started = False
        self._cleaned = False
        self._cleanup_lock = threading.Lock()

    def _host_env(self, env: dict[str, str] | None = None) -> dict[str, str]:
        # Preserve registry credentials and cache placement, but not ambient options
        # that can change mounts, namespaces, privileges, resources, or guest env.
        host_env: dict[str, str] = {}
        for key, value in os.environ.items():
            if key.startswith(("APPTAINERENV_", "SINGULARITYENV_")):
                continue
            for prefix in ("APPTAINER_", "SINGULARITY_"):
                if key.startswith(prefix):
                    if key.removeprefix(prefix) in _CLIENT_ENV_SUFFIXES:
                        host_env[key] = value
                    break
            else:
                host_env[key] = value
        if env is not None:
            host_env.update(
                {
                    f"APPTAINERENV_{key}": value
                    for key, value in self.process_env(env).items()
                }
            )
        return host_env

    def _exec_args(self, argv: list[str]) -> list[str]:
        return [
            "exec",
            "--cleanenv",
            "--no-eval",
            "--cwd",
            self.config.workdir,
            f"instance://{self._instance}",
            *argv,
        ]

    async def _runtime_exec(self, command: list[str]) -> ProgramResult:
        return await apptainer(*self._exec_args(command), env=self._host_env())

    async def _cleanup_async(self) -> None:
        await run_shielded(asyncio.to_thread(self.cleanup))

    async def start(self) -> None:
        try:
            version = await apptainer("version", env=self._host_env())
        except FileNotFoundError as e:
            raise RuntimeError(
                "apptainer runtime selected but the `apptainer` CLI is not installed"
            ) from e
        if version.exit_code != 0:
            detail = (version.stderr or version.stdout).strip()
            raise RuntimeError(
                f"apptainer runtime selected but Apptainer is unavailable: {detail}"
            )
        match = re.search(r"(?<!\d)(\d+)\.(\d+)(?:\.\d+)?", version.stdout)
        if match is None or tuple(map(int, match.groups()[:2])) < (1, 5):
            found = version.stdout.strip() or "an unknown version"
            raise RuntimeError(f"Apptainer 1.5 or newer is required; found {found}")

        self._tempdir = Path(tempfile.mkdtemp(prefix="vf-apptainer-", dir="/tmp"))
        self._workspace = self._tempdir / "workspace"
        self._workspace.mkdir()
        session_dir = self._tempdir / "session"
        session_dir.mkdir()
        image = image_reference(self.config.image)
        transport, separator, _ = image.partition(":")
        if "://" in image or (separator and transport in _SINGLE_COLON_TRANSPORTS):
            local_image = self._tempdir / "image.sif"
            operation = "pull" if "://" in image else "build"
            try:
                converted = await apptainer(
                    operation,
                    str(local_image),
                    image,
                    env=self._host_env(),
                )
            except BaseException:
                await self._cleanup_async()
                raise
            if converted.exit_code != 0:
                await self._cleanup_async()
                detail = (converted.stderr or converted.stdout).strip()
                raise SandboxError(f"apptainer image {operation} failed: {detail}")
            image = str(local_image)

        try:
            inspected = await apptainer(
                "inspect", "--startscript", image, env=self._host_env()
            )
        except BaseException:
            await self._cleanup_async()
            raise
        if inspected.exit_code != 0:
            await self._cleanup_async()
            detail = (inspected.stderr or inspected.stdout).strip()
            raise SandboxError(f"apptainer image inspection failed: {detail}")
        if inspected.stdout.strip():
            await self._cleanup_async()
            raise SandboxError(
                "Apptainer images with a startscript are unsupported because "
                "instance start would execute it outside the Runtime process lifecycle"
            )

        try:
            listed = await apptainer(
                "instance", "list", "--json", self._instance, env=self._host_env()
            )
        except BaseException:
            await self._cleanup_async()
            raise
        if listed.exit_code != 0:
            await self._cleanup_async()
            raise SandboxError(
                f"Apptainer instance inspection failed: {listed.stderr.strip()}"
            )
        try:
            instances = json.loads(listed.stdout)["instances"]
            collision = any(
                instance["instance"] == self._instance for instance in instances
            )
        except (KeyError, TypeError, ValueError) as error:
            await self._cleanup_async()
            raise SandboxError("Apptainer returned an invalid instance list") from error
        if collision:
            await self._cleanup_async()
            raise SandboxError(
                f"Apptainer instance name {self._instance!r} is already in use"
            )

        limits: list[str] = []
        if self.config.cpu is not None:
            limits += ["--cpus", str(self.config.cpu)]
        if self.config.memory is not None:
            limits += ["--memory", f"{self.config.memory}G"]
        gpu = {
            "nvidia:all": ["--nv"],
            "rocm:all": ["--rocm"],
            None: [],
        }[self.config.gpu]
        client_started = asyncio.Event()
        try:
            started = await apptainer(
                "instance",
                "start",
                "--cleanenv",
                "--containall",
                "--no-mount",
                "cwd,hostfs,bind-paths",
                "--no-eval",
                "--no-umask",
                "--workdir",
                str(session_dir),
                "--writable-tmpfs",
                "--bind",
                f"{self._workspace}:{self.config.workdir}:rw",
                *limits,
                *gpu,
                image,
                self._instance,
                env=self._host_env(),
                client_started=client_started,
            )
        except BaseException:
            # The local client can be cancelled after the instance was created but
            # before it reports success, so teardown must attempt the named stop.
            self._instance_started = client_started.is_set()
            with contextlib.suppress(Exception):
                await self._cleanup_async()
            raise
        self._instance_started = True
        if started.exit_code != 0:
            await self._cleanup_async()
            raise SandboxError(
                f"apptainer instance start failed: {started.stderr.strip()}"
            )
        self.info.id = self._instance

    async def run(self, argv: list[str], env: dict[str, str]) -> ProgramResult:
        return await apptainer(
            *self._exec_args(argv),
            env=self._host_env(env),
        )

    async def open_process(
        self, argv: list[str], env: dict[str, str]
    ) -> AttachedProcess:
        return await open_attached_process(
            argv,
            command=["apptainer", *self._exec_args([])],
            runtime_exec=self._runtime_exec,
            runtime_name="apptainer",
            host_env=self._host_env(env),
        )

    async def run_background(
        self, argv: list[str], env: dict[str, str], log: str
    ) -> None:
        result = await apptainer(
            *self._exec_args(
                [
                    "sh",
                    "-c",
                    'log=$1; shift; nohup "$@" > "$log" 2>&1 < /dev/null &',
                    "vf-background",
                    log,
                    *argv,
                ]
            ),
            env=self._host_env(env),
        )
        if result.exit_code != 0:
            raise SandboxError(
                f"apptainer background process failed: {result.stderr.strip()}"
            )

    async def _read(self, path: str) -> bytes:
        process = await asyncio.create_subprocess_exec(
            "apptainer",
            *self._exec_args(["cat", "--", path]),
            env=self._host_env(),
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        try:
            stdout, stderr = await process.communicate()
        except BaseException:
            if process.returncode is None:
                with contextlib.suppress(ProcessLookupError):
                    process.kill()
            await run_shielded(process.communicate())
            raise
        if process.returncode != 0:
            raise SandboxError(
                f"read {path!r}: {stderr.decode(errors='replace').strip()}"
            )
        return stdout

    async def write(self, path: str, data: bytes) -> None:
        parent = str(PurePosixPath(path).parent)
        result = await apptainer(
            *self._exec_args(
                [
                    "sh",
                    "-c",
                    'mkdir -p "$1" && cat > "$2"',
                    "vf-write",
                    parent,
                    path,
                ]
            ),
            env=self._host_env(),
            input=data,
        )
        if result.exit_code != 0:
            raise SandboxError(f"write {path!r}: {result.stderr.strip()}")

    def cleanup(self) -> None:
        with self._cleanup_lock:
            if self._cleaned:
                return
            if self._instance_started:
                logger.debug("apptainer: stopping instance %s", self._instance)
                stop_error: BaseException | None = None
                try:
                    stopped = subprocess.run(
                        ["apptainer", "instance", "stop", "--force", self._instance],
                        env=self._host_env(),
                        stdout=subprocess.DEVNULL,
                        stderr=subprocess.DEVNULL,
                        timeout=30,
                        check=False,
                    )
                except (OSError, subprocess.SubprocessError) as error:
                    stop_error = error
                else:
                    if stopped.returncode != 0:
                        stop_error = RuntimeError(f"exit code {stopped.returncode}")
                if stop_error is not None:
                    try:
                        listed = subprocess.run(
                            [
                                "apptainer",
                                "instance",
                                "list",
                                "--json",
                                self._instance,
                            ],
                            env=self._host_env(),
                            capture_output=True,
                            text=True,
                            timeout=30,
                            check=False,
                        )
                        instances = json.loads(listed.stdout)["instances"]
                        absent = listed.returncode == 0 and not any(
                            instance["instance"] == self._instance
                            for instance in instances
                        )
                    except (
                        OSError,
                        subprocess.SubprocessError,
                        KeyError,
                        TypeError,
                        ValueError,
                    ):
                        absent = False
                    if not absent:
                        raise RuntimeError(
                            f"failed to stop Apptainer instance {self._instance}"
                        ) from stop_error
                self._instance_started = False
            if self._tempdir is not None and self._tempdir.exists():
                try:
                    os.chmod(self._tempdir, 0o700)
                    for root, directories, _ in os.walk(self._tempdir):
                        for directory in directories:
                            with contextlib.suppress(OSError, NotImplementedError):
                                os.chmod(
                                    Path(root) / directory,
                                    0o700,
                                    follow_symlinks=False,
                                )
                    shutil.rmtree(self._tempdir)
                except OSError as error:
                    raise RuntimeError(
                        f"failed to remove Apptainer workspace {self._tempdir}"
                    ) from error
            self._cleaned = True
