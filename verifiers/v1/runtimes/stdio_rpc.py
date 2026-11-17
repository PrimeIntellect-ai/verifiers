"""Multiplex bounded JSON calls over a RuntimeProcess's reliable byte streams."""

import asyncio
import contextlib
import json
import uuid

from verifiers.v1.errors import SandboxError
from verifiers.v1.runtimes.base import RuntimeProcess
from verifiers.v1.runtimes.rpc_wire import MAX_FRAME, encode


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

    async def _drain(self) -> None:
        async for chunk in self.process.stderr:
            self._stderr.extend(chunk)
            del self._stderr[:-4000]

    async def _receive(self) -> None:
        buffer = bytearray()
        try:
            async for chunk in self.process.stdout:
                buffer.extend(chunk)
                while len(buffer) >= 4:
                    size = int.from_bytes(buffer[:4], "big")
                    if size > MAX_FRAME:
                        raise SandboxError("runtime RPC frame exceeds 32 MiB")
                    if len(buffer) < size + 4:
                        break
                    message = json.loads(buffer[4 : size + 4])
                    del buffer[: size + 4]
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

    async def call(self, operation: str, **values):
        if self._error is not None:
            raise self._error
        request_id = uuid.uuid4().hex
        future = asyncio.get_running_loop().create_future()
        self._pending[request_id] = future
        try:
            async with self._write_lock:
                await self.process.write(
                    encode({"id": request_id, "operation": operation, **values})
                )
            return await future
        finally:
            self._pending.pop(request_id, None)

    async def close(self) -> None:
        with contextlib.suppress(Exception):
            await self.process.kill()
        for task in self._tasks:
            task.cancel()
        await asyncio.gather(*self._tasks, return_exceptions=True)
