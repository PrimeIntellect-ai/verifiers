"""Shared machinery for local container runtimes driven through a CLI `exec`."""

import asyncio
import contextlib
import shlex
import uuid
from pathlib import PurePosixPath

from pydantic_config import BaseConfig

from verifiers.v1.errors import SandboxError
from verifiers.v1.runtimes.base import ProgramResult, Runtime, RuntimeProcess
from verifiers.v1.runtimes.subprocess import read_stream
from verifiers.v1.utils.aio import run_shielded


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
    container toolkit); Podman and Apptainer expose every GPU on the host, so the count
    is advisory there. None = none."""
    disk: float | None = None
    """Advisory disk request in GB. Local containers have no portable per-container size
    limit, so this is accepted (so a task can declare it without a warning) but not
    enforced."""


async def _communicate(
    *argv: str, input: bytes | None = None
) -> tuple[int, bytes, bytes]:
    """Run a host command to completion; a cancelled await kills it first."""
    proc = await asyncio.create_subprocess_exec(
        *argv,
        stdin=asyncio.subprocess.PIPE
        if input is not None
        else asyncio.subprocess.DEVNULL,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    try:
        stdout, stderr = await proc.communicate(input)
    except BaseException:
        if proc.returncode is None:
            with contextlib.suppress(ProcessLookupError):
                proc.kill()
        await run_shielded(proc.communicate())
        raise
    return proc.returncode or 0, stdout, stderr


async def cli(*argv: str, input: bytes | None = None) -> ProgramResult:
    code, stdout, stderr = await _communicate(*argv, input=input)
    return ProgramResult(
        code, stdout.decode(errors="replace"), stderr.decode(errors="replace")
    )


class ContainerProcess(RuntimeProcess):
    """A process attached through the CLI client. The client does not forward signals,
    so `exec` (the argv prefix that runs a command inside the container) delivers them."""

    def __init__(
        self, process: asyncio.subprocess.Process, exec: list[str], pid: int
    ) -> None:
        self._process = process
        self._exec = exec
        self._pid = pid
        assert process.stdin is not None
        assert process.stdout is not None
        assert process.stderr is not None
        self._stdin = process.stdin
        self.stdout = read_stream(process.stdout)
        self.stderr = read_stream(process.stderr)

    async def write(self, data: bytes) -> None:
        self._stdin.write(data)
        await self._stdin.drain()

    async def wait(self) -> int:
        return await self._process.wait()

    async def terminate(self) -> None:
        await self._signal("TERM")

    async def kill(self) -> None:
        await self._signal("KILL")

    async def _signal(self, signal: str) -> None:
        if self._process.returncode is not None:
            return
        result = await cli(
            *self._exec,
            "sh",
            "-c",
            'kill -"$1" "-$2" 2>/dev/null || kill -"$1" "$2"',
            "vf-signal",
            signal,
            str(self._pid),
        )
        if result.exit_code != 0 and self._process.returncode is None:
            raise SandboxError(
                f"container process signal failed: {result.stderr.strip()}"
            )


async def _abort_process_startup(
    proc: asyncio.subprocess.Process, exec: list[str], pidfile: str
) -> str:
    """Kill a partially opened container process and reap its local CLI client."""
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
                cli(*exec, "sh", "-c", cleanup, "vf-process-cleanup", pidfile),
                timeout=2,
            )
    finally:
        if proc.returncode is None:
            with contextlib.suppress(ProcessLookupError):
                proc.kill()
        _, stderr = await proc.communicate()
    return stderr.decode(errors="replace").strip()


class ContainerRuntime(Runtime):
    """A local container reached through its CLI: every operation is an `exec` into it.
    Subclasses provision the container (`start` / `cleanup`) and describe the exec."""

    config: ContainerConfig

    async def _exec(self, env: dict[str, str], *, stdin: bool = False) -> list[str]:
        """Host argv that runs a command inside the container, in the workdir, with
        `env` in its environment; `stdin` keeps the caller's stdin attached. Async so
        a runtime can finish per-exec preparation first."""
        raise NotImplementedError

    async def run(self, argv: list[str], env: dict[str, str]) -> ProgramResult:
        return await cli(*await self._exec(self.process_env(env)), *argv)

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
        control = await self._exec({})
        proc = await asyncio.create_subprocess_exec(
            *await self._exec(self.process_env(env), stdin=True),
            "sh",
            "-c",
            wrapper,
            "vf-process",
            pidfile,
            *argv,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        loop = asyncio.get_running_loop()
        deadline = loop.time() + 5
        exited = False
        try:
            while True:
                ready = await cli(*control, "cat", pidfile)
                if ready.exit_code == 0 and ready.stdout.strip().isdigit():
                    return ContainerProcess(proc, control, int(ready.stdout.strip()))
                # A target that already exited still left its pidfile: poll once more.
                if exited or loop.time() >= deadline:
                    break
                exited = proc.returncode is not None
                await asyncio.sleep(0.05)
        except BaseException:
            await run_shielded(_abort_process_startup(proc, control, pidfile))
            raise

        stderr = await run_shielded(_abort_process_startup(proc, control, pidfile))
        detail = stderr or ready.stderr.strip()
        raise SandboxError(
            f"container live process failed to start: {detail or 'PID unavailable'}"
        )

    async def run_background(
        self, argv: list[str], env: dict[str, str], log: str
    ) -> None:
        # Backgrounded inside the container, so it outlives this exec and lives until
        # the container is removed in stop().
        script = f"{shlex.join(argv)} > {shlex.quote(log)} 2>&1 < /dev/null &"
        result = await cli(*await self._exec(self.process_env(env)), "sh", "-c", script)
        if result.exit_code != 0:
            raise SandboxError(
                f"container background process failed: {result.stderr.strip()}"
            )

    async def _read(self, path: str) -> bytes:
        code, data, stderr = await _communicate(*await self._exec({}), "cat", path)
        if code != 0:
            raise SandboxError(
                f"read {path!r}: {stderr.decode(errors='replace').strip()}"
            )
        return data

    async def write(self, path: str, data: bytes) -> None:
        parent = shlex.quote(str(PurePosixPath(path).parent))
        result = await cli(
            *await self._exec({}, stdin=True),
            "sh",
            "-c",
            f"mkdir -p {parent} && cat > {shlex.quote(path)}",
            input=data,
        )
        if result.exit_code != 0:
            raise SandboxError(f"write {path!r}: {result.stderr.strip()}")
