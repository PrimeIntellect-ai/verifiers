"""Local Docker runtime. On Linux the container shares the host network, so host
services and container ports are reachable on loopback. Elsewhere Docker runs in a
VM: the container reaches host loopback through `host.docker.internal` and publishes
one service port. Docker enforces no network policy. The same backend also execs into
Compose services on a provider VM through that VM's runtime."""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import os
import re
import shlex
import signal
import subprocess
import sys
import uuid
from collections.abc import AsyncIterator
from ipaddress import ip_address
from pathlib import PurePosixPath
from typing import TYPE_CHECKING, ClassVar, Literal
from urllib.parse import urlsplit

from pydantic_config import BaseConfig

from verifiers.v1.errors import SandboxError
from verifiers.v1.runtimes.backends.subprocess import SubprocessProcess
from verifiers.v1.runtimes.base import (
    SERVICE_PORT,
    BaseRuntimeInfo,
    ProgramResult,
    RuntimeBackend,
    RuntimeProcess,
    parse_gpu,
)
from verifiers.v1.runtimes.runtime import Runtime
from verifiers.v1.utils.aio import run_shielded
from verifiers.v1.utils.scope import run_scope

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from verifiers.v1.runtimes.backends.prime import PrimeConfig

_HOST_NETWORK = sys.platform == "linux"
_HOST_ALIAS = "host.docker.internal"


class DockerConfig(BaseConfig):
    type: Literal["docker"] = "docker"
    image: str = "python:3.11-slim"
    workdir: str | None = None
    """Working directory override; None uses the task's workdir, or /app."""
    # TaskData.resources uses these units; non-default runtime config values take precedence.
    cpu: float | None = None
    """Pin the container to this many CPU cores. None = unlimited."""
    memory: float | None = None
    """Hard memory limit in GB. None = unlimited."""
    gpu: str | None = None
    """GPU spec, e.g. "A100" or "2". Docker exposes that many GPUs (needs the nvidia
    container toolkit) and checks requested types with nvidia-smi. None = none."""
    disk: float | None = None
    """Advisory disk request in GB. Docker has no portable per-container size limit, so this is accepted (so a task can declare it without a warning) but not
    enforced."""


class DockerRuntimeInfo(DockerConfig, BaseRuntimeInfo):
    pass


def _is_loopback(host: str) -> bool:
    if host == "localhost" or host.endswith(".localhost"):
        return True
    try:
        return ip_address(host).is_loopback
    except ValueError:
        return False


async def _communicate(
    *argv: str, input: bytes | None = None, env: dict[str, str] | None = None
) -> tuple[int, bytes, bytes]:
    """Run a host command; explicit env replaces inheritance, cancellation kills it."""
    proc = await asyncio.create_subprocess_exec(
        *argv,
        env=env,
        stdin=asyncio.subprocess.PIPE
        if input is not None
        else asyncio.subprocess.DEVNULL,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        start_new_session=True,
    )
    try:
        stdout, stderr = await proc.communicate(input)
    except BaseException:
        with contextlib.suppress(ProcessLookupError):
            os.killpg(proc.pid, signal.SIGKILL)
        await run_shielded(proc.communicate())
        raise
    return proc.returncode or 0, stdout, stderr


async def cli(
    *argv: str, input: bytes | None = None, env: dict[str, str] | None = None
) -> ProgramResult:
    code, stdout, stderr = await _communicate(*argv, input=input, env=env)
    return ProgramResult(
        code, stdout.decode(errors="replace"), stderr.decode(errors="replace")
    )


class ContainerProcess(RuntimeProcess):
    """A process attached through the CLI client. The client does not forward signals,
    so the container runtime delivers them directly to the inner process group."""

    def __init__(
        self,
        process: RuntimeProcess,
        runtime: Runtime,
        pid: int,
        pidfile: str,
    ) -> None:
        self._process = process
        self._runtime = runtime.backend.bind(env={}).with_user(runtime.user)
        self._pid = pid
        self._pidfile = pidfile
        self.stdout, self.stderr = process.stdout, process.stderr

    async def write(self, data: bytes) -> None:
        await self._process.write(data)

    async def wait(self) -> int:
        try:
            return await self._process.wait()
        finally:
            # Removing bookkeeping files must not replace the process result.
            with contextlib.suppress(Exception):
                await run_shielded(self._runtime.run(["rm", "-f", self._pidfile], {}))

    async def poll(self) -> int | None:
        return await self._process.poll()

    async def terminate(self) -> None:
        await self._signal("TERM")

    async def kill(self) -> None:
        await self._signal("KILL")

    async def _signal(self, signal: str) -> None:
        if await self.poll() is not None:
            return
        result = await self._runtime.run(
            [
                "sh",
                "-c",
                'kill -"$1" "-$2" 2>/dev/null || kill -"$1" "$2" 2>/dev/null',
                "vf-signal",
                signal,
                str(self._pid),
            ],
            {},
        )
        if result.exit_code != 0 and await self.poll() is None:
            raise SandboxError(
                f"container process signal failed: {result.stderr.strip()}"
            )


async def _abort_process_startup(
    proc: RuntimeProcess, runtime: Runtime, pidfile: str
) -> str:
    """Kill a partially opened container process and reap its CLI client."""
    runtime = runtime.backend.bind(env={}).with_user(runtime.user)
    # The target normally writes its PID immediately, but cancellation can win
    # that race. Wait briefly for the file before signalling the process group.
    cleanup = (
        'i=0; while [ "$i" -lt 20 ]; do '
        'if [ -s "$1" ]; then pid=$(cat "$1"); '
        'kill -KILL "-$pid" 2>/dev/null || kill -KILL "$pid" 2>/dev/null || true; '
        'rm -f "$1"; exit 0; fi; '
        "i=$((i + 1)); sleep 0.05; done; exit 1"
    )
    try:
        with contextlib.suppress(Exception):
            # An exited CLI client may have left its inner process running.
            await asyncio.wait_for(
                runtime.run(
                    [
                        "sh",
                        "-c",
                        cleanup,
                        "vf-process-cleanup",
                        pidfile,
                    ],
                    {},
                ),
                timeout=5,
            )
    finally:
        with contextlib.suppress(Exception):
            await asyncio.wait_for(proc.kill(), 5)
        stderr = b""
        with contextlib.suppress(Exception):
            async with asyncio.timeout(5):
                async for _ in proc.stdout:
                    pass
                stderr = b"".join([chunk async for chunk in proc.stderr])
                await proc.wait()
    return stderr.decode(errors="replace").strip()


class DockerBackend(RuntimeBackend):
    """A container reached through the docker CLI: every operation is an `exec`."""

    SUPPORTS_TASK_IMAGES: ClassVar[bool] = True
    SUPPORTS_USER_SWITCHING: ClassVar[bool] = True
    SUPPORTS_LIVE_PROCESSES: ClassVar[bool] = True

    def __init__(
        self,
        config: DockerConfig | PrimeConfig,
        name: str | None = None,
        *,
        host_backend: RuntimeBackend | None = None,
    ) -> None:
        super().__init__(name)
        self.config = config
        self._host_backend = host_backend
        self.is_local = host_backend is None or host_backend.is_local
        self.info = (
            host_backend.info.model_copy(
                update={"image": self.config.image, "workdir": self.config.workdir}
            )
            if host_backend
            else DockerRuntimeInfo(**self.config.model_dump())
        )
        self._container: str | None = None  # our `--name` (used for exec/rm)
        self._service_url: str | None = None
        self._image_env: dict[str, str] = {}
        self._stopped = False

    @classmethod
    @contextlib.asynccontextmanager
    async def attach(
        cls,
        config: DockerConfig | PrimeConfig,
        container: str,
        *,
        host_backend: RuntimeBackend,
    ) -> AsyncIterator[DockerBackend]:
        """Borrow a container on a provider VM; its caller owns creation and removal."""
        backend = cls(config, host_backend=host_backend)
        backend._container = container
        backend.info.borrowed = True  # cleanup never removes a borrowed container
        try:
            code, data, stderr = await backend._communicate_host(
                "docker", "inspect", "--format", "{{json .Config}}", container
            )
            if code:
                raise SandboxError(
                    f"Container inspection failed: {stderr.decode(errors='replace')}"
                )
            info = json.loads(data)
            backend._image_env = dict(
                entry.split("=", 1) for entry in info["Env"] or []
            )
            backend.config = config.model_copy(
                update={"image": info["Image"], "workdir": info["WorkingDir"] or "/"}
            )
            backend.info.image = backend.config.image
            backend.info.workdir = backend.config.workdir
            yield backend
        finally:
            await backend.stop()

    @property
    def published_port(self) -> int | None:
        if self._host_backend is None and _HOST_NETWORK:
            return None
        return SERVICE_PORT

    async def start(self) -> None:
        # Compose keeps its authored workdir; only standalone containers default to /app.
        self.config = self.config.model_copy(
            update={"workdir": self.config.workdir or "/app"}
        )
        self.info.workdir = self.config.workdir
        try:
            version = await cli("docker", "version")
        except FileNotFoundError as e:
            raise RuntimeError(
                "docker runtime selected but the `docker` CLI is not installed"
            ) from e
        if version.exit_code != 0:
            detail = (version.stderr or version.stdout).strip()
            hint = ""
            if "permission denied" in detail.lower():
                hint = (
                    "\nYour user isn't in the `docker` group. Either run the command "
                    'under `sg docker -c "..."`, or add yourself with '
                    "`sudo usermod -aG docker $USER` and start a new login shell."
                )
            raise RuntimeError(
                f"docker runtime selected but the engine is not reachable: {detail}{hint}"
            )
        # Engine names are global; cleanup must never target a caller's existing container.
        self._container = f"vf-{uuid.uuid4().hex}"
        if _HOST_NETWORK:
            options = ["--network", "host"]
        else:
            options = ["--publish", f"127.0.0.1::{SERVICE_PORT}"]
            options += ["--add-host", f"{_HOST_ALIAS}:host-gateway"]
        if self.config.cpu is not None:
            options += ["--cpus", str(self.config.cpu)]
        if self.config.memory is not None:
            options += ["--memory", f"{self.config.memory}g"]
        gpu_type, gpu_count = parse_gpu(self.config.gpu)
        if gpu_count:
            options += ["--gpus", str(gpu_count)]
        env_args = [
            arg
            for key, value in self.env.items()
            for arg in ("--env", f"{key}={value}")
        ]
        run = await cli(
            "docker",
            "run",
            "--label",
            f"verifiers.run={run_scope()}",
            "--detach",
            *options,
            *env_args,
            "--entrypoint",
            "sleep",
            "--name",
            self._container,
            self.config.image,
            "infinity",
        )
        if run.exit_code != 0:
            raise SandboxError(f"docker run failed: {run.stderr.strip()}")
        self.info.id = run.stdout.strip()[:12]  # `run -d` prints the container id
        if gpu_type and gpu_count:
            # Check the devices Docker actually exposed, including on remote daemons.
            gpus = await cli(
                "docker",
                "exec",
                self._container,
                "nvidia-smi",
                "--query-gpu=name",
                "--format=csv,noheader",
            )
            if gpus.exit_code != 0:
                raise SandboxError(
                    f"Cannot verify Docker GPU type: {(gpus.stderr or gpus.stdout).strip()}"
                )
            names = gpus.stdout.strip().splitlines()
            if len(names) != gpu_count or any(
                not re.search(
                    rf"(?:^|[\s-]){re.escape(gpu_type)}(?:$|[\s-])", name, re.IGNORECASE
                )
                for name in names
            ):
                raise SandboxError(
                    f"Requested {self.config.gpu!r}, but Docker exposed: {', '.join(names) or 'no GPUs'}"
                )
        inspected = await cli(
            "docker", "inspect", "--format", "{{json .Config}}", self._container
        )
        if inspected.exit_code != 0:
            raise SandboxError(
                f"docker environment inspection failed: {inspected.stderr.strip()}"
            )
        container = json.loads(inspected.stdout)
        self._image_env = dict(entry.split("=", 1) for entry in container["Env"] or [])
        # Create a missing workdir, owned by the image's execution user.
        made = await cli(
            "docker",
            "exec",
            "--user",
            "0",
            self._container,
            "sh",
            "-c",
            'if [ ! -d "$1" ]; then mkdir -p "$1" && chown "$2" "$1"; fi',
            "vf-workdir",
            self.config.workdir,
            container["User"] or "0",
        )
        if made.exit_code != 0:
            raise SandboxError(f"docker workdir setup failed: {made.stderr.strip()}")
        if not _HOST_NETWORK:
            published = await cli(
                "docker", "port", self._container, f"{SERVICE_PORT}/tcp"
            )
            host_port = (published.stdout.split() or [""])[0].rpartition(":")[2]
            if published.exit_code != 0 or not host_port.isdigit():
                detail = (published.stderr or published.stdout).strip()
                raise SandboxError(f"docker did not publish the service port: {detail}")
            self._service_url = f"http://127.0.0.1:{host_port}"
        logger.info(
            "docker: started container %s (image=%s)",
            self._container,
            self.config.image,
        )

    def host_url(self, url: str) -> str:
        if self._host_backend is not None:
            return self._host_backend.host_url(url)
        parts = urlsplit(url)
        if _HOST_NETWORK or not _is_loopback((parts.hostname or "").lower()):
            return url
        userinfo, at, _ = parts.netloc.rpartition("@")
        port = f":{parts.port}" if parts.port else ""
        return parts._replace(netloc=f"{userinfo}{at}{_HOST_ALIAS}{port}").geturl()

    async def expose(self, port: int) -> str:
        if self._host_backend is not None:
            return await self._host_backend.expose(port)
        if _HOST_NETWORK:
            return f"http://127.0.0.1:{port}"
        if port != SERVICE_PORT or self._service_url is None:
            raise SandboxError(f"docker publishes only port {SERVICE_PORT}, not {port}")
        return self._service_url

    async def prepare_execution(self, routes: list[str] | None) -> None:
        if self._host_backend is not None:
            await self._host_backend.prepare_execution(routes)

    def _exec(
        self, env: dict[str, str], *, stdin: bool = False, user: str | None = None
    ) -> list[str]:
        assert self._container is not None
        # Callback URLs must bypass any proxy the image or task configures.
        values = {**self._image_env, **env}
        exclusions = dict.fromkeys(
            entry.strip()
            for key in ("NO_PROXY", "no_proxy")
            for entry in values.get(key, "").split(",")
            if entry.strip()
        )
        exclusions.update(dict.fromkeys(("localhost", "127.0.0.1", _HOST_ALIAS)))
        env = {
            **env,
            "NO_PROXY": ",".join(exclusions),
            "no_proxy": ",".join(exclusions),
        }
        return [
            "docker",
            "exec",
            *(("-i",) if stdin else ()),
            *(arg for key, value in env.items() for arg in ("--env", f"{key}={value}")),
            *(("--user", user) if user is not None else ()),
            "--workdir",
            self.config.workdir,
            self._container,
        ]

    async def _run_background(
        self, runtime: Runtime, argv: list[str], env: dict[str, str], log: str
    ) -> None:
        # The engine owns the background server as a detached exec process.
        command = self._exec(runtime.process_env(env), user=runtime.user)
        command.insert(2, "--detach")
        script = f"exec {shlex.join(argv)} > {shlex.quote(log)} 2>&1 < /dev/null"
        result = await self._run_host(*command, "sh", "-c", script)
        if result.exit_code != 0:
            raise SandboxError(
                f"docker background process failed: {result.stderr.strip()}"
            )

    def cleanup(self) -> None:
        if self.info.borrowed or self._container is None or self._stopped:
            return
        self._stopped = (
            True  # idempotency guard; keep `_container` so the name still shows
        )
        logger.debug("docker: removing container %s", self._container)
        with contextlib.suppress(Exception):
            subprocess.run(
                ["docker", "rm", "--force", self._container],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                timeout=30,
                check=False,
            )

    async def _run_host(self, *argv: str) -> ProgramResult:
        if self._host_backend is None:
            return await cli(*argv)
        return await self._host_backend.bind().run(list(argv), {})

    async def _communicate_host(
        self, *argv: str, input: bytes | None = None
    ) -> tuple[int, bytes, bytes]:
        if self._host_backend is None:
            return await _communicate(*argv, input=input)
        # Stage bytes through the provider filesystem, never its text command logs.
        temporary = f"/tmp/vf-io-{uuid.uuid4().hex}"
        command = f"{shlex.join(argv)} > {temporary}.out"
        try:
            if input is not None:
                await self._host_backend.bind().write(f"{temporary}.in", input)
                command += f" < {temporary}.in"
            result = await self._run_host("sh", "-c", command)
            data = await self._host_backend.bind().read(f"{temporary}.out")
            return result.exit_code, data, result.stderr.encode()
        finally:
            # Temporary-file cleanup is best-effort after command completion/failure.
            with contextlib.suppress(Exception):
                await run_shielded(
                    self._run_host("rm", "-f", f"{temporary}.in", f"{temporary}.out")
                )

    async def _run(
        self, runtime: Runtime, argv: list[str], env: dict[str, str]
    ) -> ProgramResult:
        return await self._run_host(
            *self._exec(runtime.process_env(env), user=runtime.user), *argv
        )

    async def _open_process(
        self, runtime: Runtime, argv: list[str], env: dict[str, str]
    ) -> RuntimeProcess:
        pidfile = f"/tmp/vf-process-{uuid.uuid4().hex}.pid"
        # Give the target its own process group when `setsid -w` is available so
        # terminate()/kill() reap its descendants while the CLI client remains
        # attached if setsid needs to fork. The inner shell records the
        # post-setsid PID before exec preserves it as the target PID.
        wrapper = (
            "if setsid -w true >/dev/null 2>&1; then "
            'exec setsid -w sh -c \'echo $$ > "$1"; shift; exec "$@"\' '
            'vf-process "$@"; '
            'fi; echo $$ > "$1"; shift; exec "$@"'
        )
        command = [
            *self._exec(runtime.process_env(env), user=runtime.user, stdin=True),
            "sh",
            "-c",
            wrapper,
            "vf-process",
            pidfile,
            *argv,
        ]
        if self._host_backend is None:
            proc = SubprocessProcess(
                await asyncio.create_subprocess_exec(
                    *command,
                    stdin=asyncio.subprocess.PIPE,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                    start_new_session=True,
                )
            )
        else:
            proc = await self._host_backend.bind().open_process(command, {})
        loop = asyncio.get_running_loop()
        deadline = loop.time() + (30 if self._host_backend is not None else 5)
        try:
            async with asyncio.timeout_at(deadline):
                while True:
                    # An exited client gets one final PID read: short commands may
                    # have written their PID and finished before the first poll.
                    returncode = await proc.poll()
                    ready = await self._run_host(
                        *self._exec({}, user=runtime.user),
                        "cat",
                        pidfile,
                    )
                    if ready.exit_code == 0 and ready.stdout.strip().isdigit():
                        return ContainerProcess(
                            proc, runtime, int(ready.stdout.strip()), pidfile
                        )
                    if returncode is not None:
                        break
                    await asyncio.sleep(0.05)
        except TimeoutError as error:
            stderr = await run_shielded(_abort_process_startup(proc, runtime, pidfile))
            raise SandboxError(
                f"container live process failed to start: {stderr or 'PID unavailable'}"
            ) from error
        except BaseException:
            await run_shielded(_abort_process_startup(proc, runtime, pidfile))
            raise
        stderr = await run_shielded(_abort_process_startup(proc, runtime, pidfile))
        raise SandboxError(
            f"container client exited with code {returncode} before reporting its PID: "
            f"{stderr or ready.stderr.strip()}"
        )

    async def _read(
        self, runtime: Runtime, path: str, max_bytes: int | None = None
    ) -> bytes:
        argv = ["cat"] if max_bytes is None else ["head", "-c", str(max_bytes)]
        code, data, stderr = await self._communicate_host(
            *self._exec({}, user=runtime.user), *argv, "--", path
        )
        if code != 0:
            raise SandboxError(
                f"read {path!r}: {stderr.decode(errors='replace').strip()}"
            )
        return data

    async def _write(self, runtime: Runtime, path: str, data: bytes) -> None:
        parent = shlex.quote(str(PurePosixPath(path).parent))
        code, _, stderr = await self._communicate_host(
            *self._exec({}, user=runtime.user, stdin=True),
            "sh",
            "-c",
            f"mkdir -p {parent} && cat > {shlex.quote(path)}",
            input=data,
        )
        if code != 0:
            raise SandboxError(
                f"write {path!r}: {stderr.decode(errors='replace').strip()}"
            )
