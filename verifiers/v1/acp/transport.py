"""Adapt runtime process bytes to the ACP SDK's NDJSON transport."""

import asyncio
from typing import Any, cast

from acp._transport import NdjsonTransport
from acp.task import MessageSender, TaskSupervisor

from verifiers.v1.runtimes import RuntimeProcess


class ProcessTransport(NdjsonTransport):
    """The SDK owns framing and serialized writes; the runtime owns the process."""

    def __init__(self, process: RuntimeProcess) -> None:
        self._process = process
        self._pending = b""
        self._tasks = TaskSupervisor(source="verifiers.acp")
        # The SDK reassembles lines after LimitOverrunError, so StreamReader's
        # buffer limit does not cap ACP message size.
        reader = asyncio.StreamReader()
        self._tasks.create(self._read(reader))
        # MessageSender needs write/drain; the runtime performs the actual I/O.
        super().__init__(reader, MessageSender(cast(Any, self), self._tasks))

    async def _read(self, reader: asyncio.StreamReader) -> None:
        try:
            async for chunk in self._process.stdout:
                reader.feed_data(chunk)
        except Exception as error:  # noqa: BLE001 - deliver runtime failures to the SDK
            reader.set_exception(error)
        else:
            reader.feed_eof()

    def write(self, data: bytes) -> None:
        self._pending += data

    async def drain(self) -> None:
        data, self._pending = self._pending, b""
        await self._process.write(data)

    async def close(self) -> None:
        try:
            await super().close()
        finally:
            await self._tasks.shutdown()
