"""`EnvClient`: its receive loop hands each run its own deltas in order before the reply,
whatever else is queued, and `close()` drains the fire-and-forget cancels a cancelled
run leaves behind."""

import asyncio
import contextlib
import subprocess
import sys
import time
from collections.abc import AsyncIterator

import numpy as np
import pytest
import zmq
import zmq.asyncio

from verifiers.v1.serve.client import EnvClient
from verifiers.v1.serve.delta import pack
from verifiers.v1.serve.types import HealthRequest, HealthResponse

# `_request` parks a cancelled run's fire-and-forget cancel in `_cancel_tasks` with a
# `discard` done-callback. One loop turn later the sends have finished but the callbacks
# are still queued behind us, so `close()` meets a set of finished tasks.
CLOSE_WITH_FINISHED_CANCELS = """
import asyncio, uuid
from verifiers.v1.serve.client import EnvClient

async def main():
    client = EnvClient("tcp://127.0.0.1:1")
    for _ in range(3):
        task = asyncio.get_running_loop().create_task(client._send_cancel(uuid.uuid4().hex))
        client._cancel_tasks.add(task)
        task.add_done_callback(client._cancel_tasks.discard)
    await asyncio.sleep(0)
    assert all(task.done() for task in client._cancel_tasks) and client._cancel_tasks
    await client.close()

asyncio.run(main())
"""


def test_close_drains_finished_cancels():
    """`close()` returns once every fire-and-forget cancel has run, also when all of them
    finished before it looked: the state Ctrl-C leaves behind in served mode. A child
    process bounds the check, since a `close()` that never yields to the loop never lets
    an in-loop timeout fire either."""
    subprocess.run(
        [sys.executable, "-c", CLOSE_WITH_FINISHED_CANCELS], check=True, timeout=30
    )


REPLY = pack(HealthResponse().model_dump(mode="json"))


@contextlib.asynccontextmanager
async def fake_server() -> AsyncIterator[tuple[zmq.asyncio.Socket, EnvClient]]:
    """A bare ROUTER speaking the serve protocol, and a client connected to it. The
    test scripts the frames: `[identity, request_id, b"delta" | b"reply", data]`."""
    ctx = zmq.asyncio.Context()
    router = ctx.socket(zmq.ROUTER)
    router.setsockopt(zmq.LINGER, 0)
    port = router.bind_to_random_port("tcp://127.0.0.1")
    client = EnvClient(f"tcp://127.0.0.1:{port}")
    try:
        yield router, client
    finally:
        await client.close()
        router.close()
        ctx.term()


async def until(condition, timeout: float = 10.0) -> None:
    deadline = time.monotonic() + timeout
    while not condition():
        assert time.monotonic() < deadline, "timed out"
        await asyncio.sleep(0.01)


@pytest.mark.asyncio
async def test_interleaved_runs_see_their_deltas_in_order_before_the_reply(caplog):
    """Runs streaming at once each see their own deltas in order, and a run's reply
    lands only after its last delta was applied. A malformed frame and an undecodable
    delta queued among them are logged and skipped without losing their neighbours."""
    n_runs, n_deltas = 20, 50
    async with fake_server() as (router, client):
        seen: list[list[int]] = [[] for _ in range(n_runs)]

        async def run(i: int) -> list[int]:
            await client._request(
                HealthRequest(),
                HealthResponse,
                on_delta=lambda delta: seen[i].append(delta["seq"]),
            )
            return list(seen[i])  # what the run had when its reply landed

        runs = [asyncio.create_task(run(i)) for i in range(n_runs)]
        requests = [await router.recv_multipart() for _ in range(n_runs)]
        for seq in range(n_deltas):
            for identity, request_id, *_ in requests:
                await router.send_multipart(
                    [identity, request_id, b"delta", pack({"seq": seq})]
                )
            if seq == n_deltas // 2:
                identity, request_id, *_ = requests[0]
                await router.send_multipart([identity, b"not-the-protocol"])
                await router.send_multipart([identity, request_id, b"delta", b"\xc1"])
        for identity, request_id, *_ in requests:
            await router.send_multipart([identity, request_id, b"reply", REPLY])
        assert await asyncio.gather(*runs) == [list(range(n_deltas))] * n_runs
    assert caplog.text.count("dropping a malformed env-server frame") == 2


@pytest.mark.asyncio
async def test_a_cancelled_run_stops_seeing_its_stream():
    """Deltas that arrive after a run was cancelled are dropped; a run streaming
    beside it still sees all of its own."""
    async with fake_server() as (router, client):
        seen_a: list[int] = []
        seen_b: list[int] = []
        a = asyncio.create_task(
            client._request(
                HealthRequest(),
                HealthResponse,
                on_delta=lambda d: seen_a.append(d["seq"]),
            )
        )
        identity, request_a, *_ = await router.recv_multipart()
        b = asyncio.create_task(
            client._request(
                HealthRequest(),
                HealthResponse,
                on_delta=lambda d: seen_b.append(d["seq"]),
            )
        )
        _, request_b, *_ = await router.recv_multipart()
        for request_id in (request_a, request_b):
            await router.send_multipart(
                [identity, request_id, b"delta", pack({"seq": 0})]
            )
        await until(lambda: seen_a == [0] and seen_b == [0])
        a.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await a
        for seq in range(1, 10):
            for request_id in (request_a, request_b):
                await router.send_multipart(
                    [identity, request_id, b"delta", pack({"seq": seq})]
                )
        for request_id in (request_a, request_b):
            await router.send_multipart([identity, request_id, b"reply", REPLY])
        await b
        assert seen_a == [0]
        assert seen_b == list(range(10))


@pytest.mark.asyncio
async def test_large_deltas_arrive_intact_and_in_order():
    """Deltas of a few MB each (a long turn's token arrays) reach the run whole and in
    order."""
    tokens = np.arange(500_000, dtype=np.int32)
    async with fake_server() as (router, client):
        seen: list[dict] = []
        run = asyncio.create_task(
            client._request(HealthRequest(), HealthResponse, on_delta=seen.append)
        )
        identity, request_id, *_ = await router.recv_multipart()
        for seq in range(5):
            delta = pack({"seq": seq, "token_ids": tokens + seq})
            await router.send_multipart([identity, request_id, b"delta", delta])
        await router.send_multipart([identity, request_id, b"reply", REPLY])
        await run
        assert [delta["seq"] for delta in seen] == list(range(5))
        for seq, delta in enumerate(seen):
            data = np.frombuffer(delta["token_ids"]["data"], dtype=np.int32)
            assert np.array_equal(data, tokens + seq)


@pytest.mark.asyncio
async def test_a_deep_backlog_does_not_starve_the_loop():
    """While thousands of queued deltas drain, the rest of the event loop keeps
    getting turns."""
    n_deltas = 5_000
    async with fake_server() as (router, client):
        ticks = 0
        ticks_at: list[int] = []

        async def ticker() -> None:
            nonlocal ticks
            while True:
                ticks += 1
                await asyncio.sleep(0)

        def on_delta(delta: dict) -> None:
            if delta["seq"] in (0, n_deltas - 1):
                ticks_at.append(ticks)

        run = asyncio.create_task(
            client._request(HealthRequest(), HealthResponse, on_delta=on_delta)
        )
        identity, request_id, *_ = await router.recv_multipart()
        for seq in range(n_deltas):
            await router.send_multipart(
                [identity, request_id, b"delta", pack({"seq": seq})]
            )
        await router.send_multipart([identity, request_id, b"reply", REPLY])
        # Hold the loop so the whole stream is queued on the client's socket.
        time.sleep(0.5)  # noqa: ASYNC251 - blocking the loop is the point
        ticking = asyncio.create_task(ticker())
        await run
        ticking.cancel()
        first, last = ticks_at
        assert last - first >= 3
