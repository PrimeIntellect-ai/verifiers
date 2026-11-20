"""Stdlib-only framing shared with sandbox workers."""

import asyncio
import json

MAX_FRAME = 32 * 1024 * 1024


def encode(message: dict) -> bytes:
    data = json.dumps(message).encode()
    if len(data) > MAX_FRAME:
        raise RuntimeError("runtime RPC frame exceeds 32 MiB")
    return len(data).to_bytes(4, "big") + data


async def serve(handler) -> None:
    """Serve framed requests on stdin; cancellation closes every in-flight call."""
    import sys

    reader = asyncio.StreamReader(limit=MAX_FRAME + 4)
    protocol = asyncio.StreamReaderProtocol(reader)
    await asyncio.get_running_loop().connect_read_pipe(
        lambda: protocol, sys.stdin.buffer
    )

    async def chunks():
        while chunk := await reader.read(65536):
            yield chunk

    tasks: set[asyncio.Task] = set()
    lock = asyncio.Lock()

    async def reply(message):
        try:
            result = {"result": await handler(message)}
            packet = encode({"id": message["id"], **result})
        except Exception as error:  # noqa: BLE001 — serialize worker failures to the caller
            packet = encode(
                {"id": message["id"], "error": f"{type(error).__name__}: {error}"}
            )
        async with lock:
            sys.stdout.buffer.write(packet)
            sys.stdout.buffer.flush()

    try:
        async for message in frames(chunks()):
            task = asyncio.create_task(reply(message))
            tasks.add(task)
            task.add_done_callback(tasks.discard)
    finally:
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)


async def frames(chunks):
    """Decode complete, bounded JSON frames across arbitrary chunk boundaries."""
    buffer = bytearray()
    async for chunk in chunks:
        buffer.extend(chunk)
        while len(buffer) >= 4:
            size = int.from_bytes(buffer[:4], "big")
            if size > MAX_FRAME:
                raise RuntimeError("runtime RPC frame exceeds 32 MiB")
            if len(buffer) < size + 4:
                break
            message = json.loads(buffer[4 : 4 + size])
            del buffer[: 4 + size]
            yield message
    if buffer:
        raise RuntimeError("incomplete runtime RPC frame")
