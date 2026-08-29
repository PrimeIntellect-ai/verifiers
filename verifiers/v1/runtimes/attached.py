"""Attached local CLI processes that need signalling inside a container."""

import asyncio
import contextlib
import tempfile
import uuid
from collections.abc import AsyncIterator, Awaitable, Callable
from typing import BinaryIO

from verifiers.v1.errors import SandboxError
from verifiers.v1.runtimes.base import ProgramResult, RuntimeProcess
from verifiers.v1.utils.aio import run_shielded

RuntimeExec = Callable[[list[str]], Awaitable[ProgramResult]]
_START_TIMEOUT = 5
_CONTROL_TIMEOUT = 2

_PROCESS_WRAPPER = (
    "if ! setsid -w true >/dev/null 2>&1; then "
    "echo 'attached processes require setsid -w' >&2; exit 125; fi; "
    'exec setsid -w sh -c \'echo $$ > "$1"; shift; exec "$@"\' '
    'vf-process "$@"'
)
_SIGNAL = 'kill -"$1" "-$2" 2>/dev/null || kill -"$1" "$2"'
_FINISH = 'kill -KILL "-$1" 2>/dev/null || true; rm -f "$2"'
_ABORT = (
    'i=0; while [ "$i" -lt 20 ]; do '
    'if [ -s "$1" ]; then pid=$(cat "$1"); '
    'kill -KILL "-$pid" 2>/dev/null || kill -KILL "$pid" 2>/dev/null || true; '
    'rm -f "$1"; exit 0; fi; '
    "i=$((i + 1)); sleep 0.05; done; exit 1"
)


class _ProcessStream(AsyncIterator[bytes]):
    def __init__(self, reader: asyncio.StreamReader) -> None:
        self.reader = reader
        self.captured: BinaryIO | None = None
        self.capture_finished = False
        self._read_lock = asyncio.Lock()

    def __aiter__(self) -> "_ProcessStream":
        return self

    async def __anext__(self) -> bytes:
        async with self._read_lock:
            if self.captured is not None:
                if chunk := self.captured.read(64 * 1024):
                    return chunk
                self.captured.close()
                self.captured = None
            if self.capture_finished:
                raise StopAsyncIteration
            chunk = await self.reader.read(64 * 1024)
            if not chunk:
                raise StopAsyncIteration
            return chunk

    async def capture(self) -> None:
        async with self._read_lock:
            if self.captured is None:
                # It lives until the caller consumes output after wait().
                self.captured = tempfile.SpooledTemporaryFile(  # noqa: SIM115
                    max_size=1 << 20
                )
            unread = self.captured.tell()
            self.captured.seek(0, 2)
            finished = False
            try:
                while chunk := await self.reader.read(64 * 1024):
                    self.captured.write(chunk)
                finished = True
            finally:
                self.capture_finished = finished
                self.captured.seek(unread)


class AttachedProcess(RuntimeProcess):
    def __init__(
        self,
        process: asyncio.subprocess.Process,
        pid: int,
        pidfile: str,
        runtime_exec: RuntimeExec,
        runtime_name: str,
    ) -> None:
        self._process = process
        self._pid = pid
        self._pidfile = pidfile
        self._runtime_exec = runtime_exec
        self._runtime_name = runtime_name
        assert process.stdin is not None
        assert process.stdout is not None
        assert process.stderr is not None
        self._stdin = process.stdin
        self.stdout = _ProcessStream(process.stdout)
        self.stderr = _ProcessStream(process.stderr)

    async def write(self, data: bytes) -> None:
        self._stdin.write(data)
        await self._stdin.drain()

    async def wait(self) -> int:
        # Let concurrently scheduled stream consumers begin reading first.
        await asyncio.sleep(0)
        capture_streams = [self.stdout, self.stderr]
        captures = [asyncio.create_task(stream.capture()) for stream in capture_streams]
        try:
            exit_code = await self._process.wait()
            # A finished leader can leave descendants holding its streamed pipes open.
            reaped = False
            with contextlib.suppress(Exception):
                cleanup = await asyncio.wait_for(
                    self._runtime_exec(
                        [
                            "sh",
                            "-c",
                            _FINISH,
                            "vf-process-finish",
                            str(self._pid),
                            self._pidfile,
                        ]
                    ),
                    _CONTROL_TIMEOUT,
                )
                reaped = cleanup.exit_code == 0
            if reaped:
                await asyncio.gather(*captures)
            else:
                await asyncio.sleep(0)
        finally:
            for capture in captures:
                capture.cancel()
            await asyncio.gather(*captures, return_exceptions=True)
        return exit_code

    async def terminate(self) -> None:
        await self._signal("TERM")

    async def kill(self) -> None:
        await self._signal("KILL")

    def close_stdin(self) -> None:
        self._stdin.close()

    async def _signal(self, signal: str) -> None:
        if self._process.returncode is not None:
            return
        result = await self._runtime_exec(
            ["sh", "-c", _SIGNAL, "vf-signal", signal, str(self._pid)]
        )
        if result.exit_code != 0 and self._process.returncode is None:
            raise SandboxError(
                f"{self._runtime_name} exec process signal failed: "
                f"{result.stderr.strip()}"
            )


async def _abort_process(
    process: asyncio.subprocess.Process,
    runtime_exec: RuntimeExec,
    pidfile: str,
) -> str:
    """Kill a partially opened target and reap its local CLI process."""
    try:
        with contextlib.suppress(Exception):
            await asyncio.wait_for(
                runtime_exec(["sh", "-c", _ABORT, "vf-process-cleanup", pidfile]),
                timeout=_CONTROL_TIMEOUT,
            )
    finally:
        if process.returncode is None:
            with contextlib.suppress(ProcessLookupError):
                process.kill()
        _, stderr = await process.communicate()
    return stderr.decode(errors="replace").strip()


async def open_attached_process(
    argv: list[str],
    *,
    command: list[str],
    runtime_exec: RuntimeExec,
    runtime_name: str,
    host_env: dict[str, str] | None = None,
) -> AttachedProcess:
    """Open a streamed CLI process and resolve its target PID inside the runtime."""
    pidfile = f"/tmp/vf-process-{uuid.uuid4().hex}.pid"
    process = await asyncio.create_subprocess_exec(
        *command,
        "sh",
        "-c",
        _PROCESS_WRAPPER,
        "vf-process",
        pidfile,
        *argv,
        env=host_env,
        stdin=asyncio.subprocess.PIPE,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    ready = ProgramResult(1, "", "PID unavailable")
    exited = False
    try:
        async with asyncio.timeout(_START_TIMEOUT):
            while True:
                ready = await runtime_exec(["cat", pidfile])
                if ready.exit_code == 0 and ready.stdout.strip().isdigit():
                    return AttachedProcess(
                        process,
                        int(ready.stdout.strip()),
                        pidfile,
                        runtime_exec,
                        runtime_name,
                    )
                if process.returncode is not None:
                    if exited:
                        break
                    exited = True
                    continue
                await asyncio.sleep(0.05)
    except TimeoutError:
        pass
    except BaseException:
        await run_shielded(_abort_process(process, runtime_exec, pidfile))
        raise

    stderr = await run_shielded(_abort_process(process, runtime_exec, pidfile))
    detail = stderr or ready.stderr.strip()
    raise SandboxError(
        f"{runtime_name} live process failed to start: {detail or 'PID unavailable'}"
    )


async def run_attached_process(process: AttachedProcess) -> ProgramResult:
    """Run an attached process to completion, killing its target on cancellation."""
    process.close_stdin()

    async def read_all(stream: AsyncIterator[bytes]) -> bytes:
        return b"".join([chunk async for chunk in stream])

    try:
        stdout, stderr, exit_code = await asyncio.gather(
            read_all(process.stdout), read_all(process.stderr), process.wait()
        )
    except BaseException:
        with contextlib.suppress(Exception):
            await run_shielded(
                _abort_process(
                    process._process,
                    process._runtime_exec,
                    process._pidfile,
                )
            )
        raise
    return ProgramResult(
        exit_code=exit_code,
        stdout=stdout.decode(errors="replace"),
        stderr=stderr.decode(errors="replace"),
    )
