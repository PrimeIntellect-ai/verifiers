"""Interactive service processes without coupling callers to a provider SDK."""

from __future__ import annotations

import asyncio
import contextlib
import shlex
import uuid
from collections.abc import Awaitable, Callable

from harbor.environments.base import ExecResult

from verifiers.v1.runtimes.base import RuntimeProcess
from verifiers.v1.utils.aio import run_shielded


class ServiceProcess(RuntimeProcess):
    """Signals target the service process group, not its remote CLI client."""

    def __init__(self, process: RuntimeProcess, execute, pid: int, path: str):
        self._process, self._execute, self._pid, self._path = (
            process,
            execute,
            pid,
            path,
        )
        self.stdout, self.stderr = process.stdout, process.stderr

    async def write(self, data: bytes) -> None:
        await self._process.write(data)

    async def poll(self) -> int | None:
        return await self._process.poll()

    async def wait(self) -> int:
        try:
            return await self._process.wait()
        finally:
            with contextlib.suppress(Exception):
                await run_shielded(
                    asyncio.wait_for(
                        self._execute(f"rm -f {shlex.quote(self._path)}"), 10
                    )
                )

    async def terminate(self) -> None:
        await self._signal("TERM")

    async def kill(self) -> None:
        await self._signal("KILL")

    async def _signal(self, signal: str) -> None:
        if await self.poll() is not None:
            return
        result = await self._execute(
            f"kill -{signal} -{self._pid} 2>/dev/null || kill -{signal} {self._pid} 2>/dev/null"
        )
        if result.return_code and await self.poll() is None:
            raise RuntimeError(f"Cannot signal service process: {result.stderr}")


async def open_service_process(
    command: str,
    *,
    launch: Callable[[str], Awaitable[RuntimeProcess]],
    execute: Callable[[str], Awaitable[ExecResult]],
) -> RuntimeProcess:
    path = f"/tmp/harbor-process-{uuid.uuid4().hex}.pid"
    inner = f"echo $$ > {shlex.quote(path)}; exec sh -c {shlex.quote(command)}"
    wrapped = (
        "if setsid -w true >/dev/null 2>&1; then "
        f"exec setsid -w sh -c {shlex.quote(inner)}; "
        f"else exec sh -c {shlex.quote(inner)}; fi"
    )
    process = await launch(wrapped)
    try:
        async with asyncio.timeout(30):
            while True:
                result = await execute(f"cat {shlex.quote(path)}")
                if result.return_code == 0 and (result.stdout or "").strip().isdigit():
                    return ServiceProcess(
                        process, execute, int((result.stdout or "").strip()), path
                    )
                if await process.poll() is not None:
                    raise RuntimeError(
                        "Service process exited before reporting its PID"
                    )
                await asyncio.sleep(0.05)
    except BaseException:

        async def cleanup():
            with contextlib.suppress(Exception):
                async with asyncio.timeout(10):
                    await execute(
                        'i=0; while [ "$i" -lt 20 ]; do '
                        f"if [ -s {path} ]; then p=$(cat {path}); "
                        'kill -KILL "-$p" 2>/dev/null || kill -KILL "$p" 2>/dev/null || true; '
                        f"rm -f {path}; exit 0; fi; "
                        "i=$((i + 1)); sleep 0.05; done; exit 1"
                    )
            with contextlib.suppress(Exception):
                async with asyncio.timeout(10):
                    await process.kill()

        await run_shielded(cleanup())
        raise
