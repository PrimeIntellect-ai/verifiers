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
    tasks: set[asyncio.Task] = set()

    async def reply(message):
        try:
            result = {"result": await handler(message)}
            packet = encode({"id": message["id"], **result})
        except Exception as error:  # noqa: BLE001 — serialize worker failures to the caller
            packet = encode(
                {"id": message["id"], "error": f"{type(error).__name__}: {error}"}
            )
        # One event-loop thread owns writes, so frames cannot interleave.
        sys.stdout.buffer.write(packet)
        sys.stdout.buffer.flush()

    try:
        while True:
            try:
                size = int.from_bytes(await reader.readexactly(4), "big")
            except asyncio.IncompleteReadError:
                break
            if size > MAX_FRAME:
                raise RuntimeError("runtime RPC frame exceeds 32 MiB")
            message = json.loads(await reader.readexactly(size))
            task = asyncio.create_task(reply(message))
            tasks.add(task)
            task.add_done_callback(tasks.discard)
    finally:
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
