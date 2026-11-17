"""Host CLI transport for a runtime provider, such as the Docker client."""

import asyncio
import contextlib
import os
import signal
from collections.abc import AsyncIterator

from verifiers.v1.runtimes.base import RuntimeProcess


async def read_stream(reader: asyncio.StreamReader) -> AsyncIterator[bytes]:
    while chunk := await reader.read(64 * 1024):
        yield chunk


def signal_process(
    process: asyncio.subprocess.Process, signal_: signal.Signals
) -> None:
    if process.returncode is not None:
        return
    # Launch CLI clients in a new session so signalling reaps their process tree.
    with contextlib.suppress(ProcessLookupError, PermissionError):
        os.killpg(os.getpgid(process.pid), signal_)


class CLIProcess(RuntimeProcess):
    def __init__(self, process: asyncio.subprocess.Process) -> None:
        self._process = process
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

    async def poll(self) -> int | None:
        return self._process.returncode

    async def terminate(self) -> None:
        signal_process(self._process, signal.SIGTERM)

    async def kill(self) -> None:
        signal_process(self._process, signal.SIGKILL)
