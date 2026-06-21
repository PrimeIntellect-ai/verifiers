import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import msgpack
import pytest
import zmq

from verifiers.v1.serve.pool import EnvServerPool


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "payload",
    [b"\xc1", msgpack.packb({}), msgpack.packb({"n": "eight"})],
)
async def test_malformed_group_payload_falls_back_to_one(payload: bytes) -> None:
    pool = object.__new__(EnvServerPool)
    pool.elastic = True
    pool.max_workers = 2
    pool.multiplex = 128
    pool.address = "test://pool"
    pool.workers = []
    pool._shutdown = MagicMock()

    frontend = MagicMock()
    frontend.recv_multipart = AsyncMock(
        return_value=[b"client", b"request", b"run_group", payload]
    )
    frontend.send_multipart = AsyncMock()
    pool.frontend = frontend

    dealer = MagicMock()
    dealer.send_multipart = AsyncMock()
    worker = {"dealer": dealer, "active": 0}
    pool._spawn_worker = MagicMock(side_effect=lambda: pool.workers.append(worker))

    poller = MagicMock()
    poller.poll = AsyncMock(
        side_effect=[{frontend: zmq.POLLIN}, asyncio.CancelledError()]
    )
    with patch("verifiers.v1.serve.pool.zmq.asyncio.Poller", return_value=poller):
        await pool.run()

    assert worker["active"] == 1
    dealer.send_multipart.assert_awaited_once_with([b"request", b"run_group", payload])
