"""Multiplex bounded JSON calls over a RuntimeProcess's reliable byte streams."""

import asyncio
import contextlib
import uuid

from verifiers.v1.errors import SandboxError
from verifiers.v1.runtimes.base import RuntimeProcess
from verifiers.v1.runtimes.rpc_wire import encode, frames


class ProcessRPC:
    def __init__(self, process: RuntimeProcess):
        self.process = process
        self._pending: dict[str, asyncio.Future] = {}
        self._write_lock = asyncio.Lock()
        self._error: BaseException | None = None
        self._stderr = bytearray()
        self._tasks = [
            asyncio.create_task(self._receive()),
            asyncio.create_task(self._drain()),
        ]

    async def _receive(self) -> None:
        try:
            async for message in frames(self.process.stdout):
                future = self._pending.get(message["id"])
                if future is not None and not future.done():
                    if "error" in message:
                        future.set_exception(SandboxError(message["error"]))
                    else:
                        future.set_result(message["result"])
            raise SandboxError(
                f"runtime RPC closed - {self._stderr.decode(errors='replace')}"
            )
        except BaseException as error:  # noqa: BLE001 — fail every pending call on EOF/cancellation
            self._error = error
            for future in self._pending.values():
                if not future.done():
                    future.set_exception(error)

    async def _drain(self) -> None:
        async for chunk in self.process.stderr:
            self._stderr.extend(chunk)
            del self._stderr[:-4000]

    async def call(self, operation: str, **values):
        if self._error is not None:
            raise self._error
        request_id = uuid.uuid4().hex
        future = asyncio.get_running_loop().create_future()
        self._pending[request_id] = future
        try:
            async with self._write_lock:
                await self.process.write(
                    encode({**values, "id": request_id, "operation": operation})
                )
            return await future
        finally:
            self._pending.pop(request_id, None)

    async def close(self, *, kill: bool = True) -> None:
        try:
            if kill:
                with contextlib.suppress(Exception):
                    await self.process.kill()
        finally:
            for task in self._tasks:
                task.cancel()
            await asyncio.gather(*self._tasks, return_exceptions=True)
