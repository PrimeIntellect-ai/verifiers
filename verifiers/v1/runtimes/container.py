"""Container operations through a CLI on the local machine or an owned runtime."""

import asyncio
import contextlib
import os
import shlex
import signal
import uuid
from pathlib import PurePosixPath
from typing import TYPE_CHECKING

from pydantic_config import BaseConfig

from verifiers.v1.errors import SandboxError
from verifiers.v1.runtimes.base import ProgramResult, Runtime, RuntimeProcess
from verifiers.v1.runtimes.subprocess import SubprocessProcess
from verifiers.v1.utils.aio import run_shielded

if TYPE_CHECKING:
    from verifiers.v1.runtimes.modal import ModalConfig
    from verifiers.v1.runtimes.prime import PrimeConfig


class ContainerConfig(BaseConfig):
    image: str = "python:3.11-slim"
    workdir: str = "/app"
    # TaskData.resources uses these units; non-default runtime config values take precedence.
    cpu: float | None = None
    """Pin the container to this many CPU cores. None = unlimited."""
    memory: float | None = None
    """Hard memory limit in GB. None = unlimited."""
    gpu: str | None = None
    """GPU spec, e.g. "A100" or "2". Docker exposes that many GPUs (needs the nvidia
    container toolkit); Podman selects that many NVIDIA CDI devices. Apptainer exposes
    all accessible NVIDIA GPUs, so its count is advisory. None = none."""
    disk: float | None = None
    """Advisory disk request in GB. Local containers have no portable per-container size
    limit, so this is accepted (so a task can declare it without a warning) but not
    enforced."""


async def _communicate(
    *argv: str, input: bytes | None = None, env: dict[str, str] | None = None
) -> tuple[int, bytes, bytes]:
    """Run a host command to completion; a cancelled await kills it first."""
    proc = await asyncio.create_subprocess_exec(
        *argv,
        env=None if env is None else {**os.environ, **env},
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
        self, process: RuntimeProcess, runtime: "ContainerRuntime", pid: int
    ) -> None:
        self._process = process
        self._runtime = runtime
        self._pid = pid
        self._returncode: int | None = None
        self.stdout, self.stderr = process.stdout, process.stderr

    async def write(self, data: bytes) -> None:
        await self._process.write(data)

    async def wait(self) -> int:
        self._returncode = await self._process.wait()
        return self._returncode

    async def terminate(self) -> None:
        await self._signal("TERM")

    async def kill(self) -> None:
        await self._signal("KILL")

    async def _signal(self, signal: str) -> None:
        if self._returncode is not None:
            return
        result = await self._runtime._run_host(
            *self._runtime._exec({}),
            "sh",
            "-c",
            'kill -"$1" "-$2" 2>/dev/null || kill -"$1" "$2" 2>/dev/null || ! kill -0 "$2" 2>/dev/null',
            "vf-signal",
            signal,
            str(self._pid),
        )
        if result.exit_code != 0:
            raise SandboxError(
                f"container process signal failed: {result.stderr.strip()}"
            )


async def _abort_process_startup(
    proc: RuntimeProcess, runtime: "ContainerRuntime", pidfile: str
) -> str:
    """Kill a partially opened container process and reap its CLI client."""
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
            await asyncio.wait_for(
                runtime._run_host(
                    *runtime._exec({}),
                    "sh",
                    "-c",
                    cleanup,
                    "vf-process-cleanup",
                    pidfile,
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


class ContainerRuntime(Runtime):
    """A container reached through its CLI: every operation is an `exec` into it.
    Subclasses provision the container (`start` / `cleanup`) and describe the exec."""

    config: "ContainerConfig | PrimeConfig | ModalConfig"
    _host: Runtime | None = None

    async def _run_host(
        self, *argv: str, env: dict[str, str] | None = None
    ) -> ProgramResult:
        if self._host is None:
            return await cli(*argv, env=env)
        return await self._host.run(list(argv), env or {})

    async def _communicate_host(
        self, *argv: str, input: bytes | None = None
    ) -> tuple[int, bytes, bytes]:
        if self._host is None:
            return await _communicate(*argv, input=input)
        # Stage bytes through the provider filesystem, never its text command logs.
        temporary = f"/tmp/vf-io-{uuid.uuid4().hex}"
        command = f"{shlex.join(argv)} > {temporary}.out"
        try:
            if input is not None:
                await self._host.write(f"{temporary}.in", input)
                command += f" < {temporary}.in"
            result = await self._run_host("sh", "-c", command)
            data = await self._host.read(f"{temporary}.out")
            return result.exit_code, data, result.stderr.encode()
        finally:
            await run_shielded(
                self._run_host("rm", "-f", f"{temporary}.in", f"{temporary}.out")
            )

    def _exec(self, env: dict[str, str], *, stdin: bool = False) -> list[str]:
        """Host argv that runs a command inside the container, in the workdir, with
        `env` in its environment; `stdin` keeps the caller's stdin attached."""
        raise NotImplementedError

    async def run(self, argv: list[str], env: dict[str, str]) -> ProgramResult:
        return await self._run_host(*self._exec(self.process_env(env)), *argv)

    async def open_process(
        self, argv: list[str], env: dict[str, str]
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
            *self._exec(self.process_env(env), stdin=True),
            "sh",
            "-c",
            wrapper,
            "vf-process",
            pidfile,
            *argv,
        ]
        if self._host is None:
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
            proc = await self._host.open_process(command, {})
        loop = asyncio.get_running_loop()
        deadline = loop.time() + (30 if self._host is not None else 5)
        try:
            async with asyncio.timeout_at(deadline):
                while True:
                    ready = await self._run_host(*self._exec({}), "cat", pidfile)
                    if ready.exit_code == 0 and ready.stdout.strip().isdigit():
                        return ContainerProcess(proc, self, int(ready.stdout.strip()))
                    await asyncio.sleep(0.05)
        except TimeoutError as error:
            stderr = await run_shielded(_abort_process_startup(proc, self, pidfile))
            raise SandboxError(
                f"container live process failed to start: {stderr or 'PID unavailable'}"
            ) from error
        except BaseException:
            await run_shielded(_abort_process_startup(proc, self, pidfile))
            raise

    async def run_background(
        self, argv: list[str], env: dict[str, str], log: str
    ) -> None:
        # Backgrounded inside the container, so it outlives this exec and lives until
        # the container is removed in stop().
        script = f"{shlex.join(argv)} > {shlex.quote(log)} 2>&1 < /dev/null &"
        result = await self._run_host(
            *self._exec(self.process_env(env)), "sh", "-c", script
        )
        if result.exit_code != 0:
            raise SandboxError(
                f"container background process failed: {result.stderr.strip()}"
            )

    async def _read(self, path: str, max_bytes: int | None = None) -> bytes:
        argv = ["cat"] if max_bytes is None else ["head", "-c", str(max_bytes)]
        code, data, stderr = await self._communicate_host(
            *self._exec({}), *argv, "--", path
        )
        if code != 0:
            raise SandboxError(
                f"read {path!r}: {stderr.decode(errors='replace').strip()}"
            )
        return data

    async def write(self, path: str, data: bytes) -> None:
        parent = shlex.quote(str(PurePosixPath(path).parent))
        code, _, stderr = await self._communicate_host(
            *self._exec({}, stdin=True),
            "sh",
            "-c",
            f"mkdir -p {parent} && cat > {shlex.quote(path)}",
            input=data,
        )
        if code != 0:
            raise SandboxError(
                f"write {path!r}: {stderr.decode(errors='replace').strip()}"
            )
