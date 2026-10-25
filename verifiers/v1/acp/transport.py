"""Adapt runtime process bytes to the ACP SDK's NDJSON transport."""

from typing import Any, cast

from acp._transport import NdjsonTransport
from acp.task import MessageSender, TaskSupervisor

from verifiers.v1.runtimes import RuntimeProcess

MAX_PACKET_BYTES = 128 * 1024 * 1024


class ProcessTransport(NdjsonTransport):
    """Pull stdout only when the SDK asks, bounding each packet before decoding it."""

    def __init__(self, process: RuntimeProcess) -> None:
        self._process = process
        self._source = process.stdout.__aiter__()
        self._buffer = bytearray()
        self._pending = b""
        self._tasks = TaskSupervisor(source="verifiers.acp")
        # RuntimeProcess supplies bytes instead of an asyncio reader/writer. The SDK
        # still owns JSON decoding and serialized writes; _read_line supplies framing.
        super().__init__(cast(Any, None), MessageSender(cast(Any, self), self._tasks))

    async def _read_line(self) -> bytes:
        while True:
            end = self._buffer.find(b"\n")
            size = end if end >= 0 else len(self._buffer)
            if size > MAX_PACKET_BYTES:
                raise ValueError(f"ACP message is too large: {size} bytes")
            if end >= 0:
                line = bytes(self._buffer[: end + 1])
                del self._buffer[: end + 1]
                return line
            try:
                self._buffer.extend(await anext(self._source))
            except StopAsyncIteration:
                line = bytes(self._buffer)
                self._buffer.clear()
                return line

    def write(self, data: bytes) -> None:
        if len(data) > MAX_PACKET_BYTES:
            raise ValueError(f"ACP message is too large: {len(data)} bytes")
        self._pending = data

    async def drain(self) -> None:
        data, self._pending = self._pending, b""
        await self._process.write(data)

    async def close(self) -> None:
        try:
            await super().close()
        finally:
            await self._tasks.shutdown()
            self._buffer.clear()
