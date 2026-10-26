"""The pool broker routes a cancel to the worker that ran the target run, even after
that run replied: the worker may still hold the run's kept training arrays."""

import asyncio
from types import SimpleNamespace

import msgpack
import zmq

from verifiers.v1.serve.pool import _CANCEL_MISS, EnvServerPool


class FakeWorkerPool(EnvServerPool):
    """A pool whose one worker is an in-process ROUTER socket the test drives."""

    def _spawn_worker(self) -> None:
        dealer = self.ctx.socket(zmq.DEALER)
        dealer.setsockopt(zmq.LINGER, 0)
        dealer.connect("inproc://worker")
        self.workers.append(
            {
                "process": SimpleNamespace(),
                "dealer": dealer,
                "pipe": SimpleNamespace(),
                "active": 0,
                "index": 0,
            }
        )
        if self._poller is not None:
            self._poller.register(dealer, zmq.POLLIN)


def test_a_late_cancel_reaches_the_worker_that_replied():
    async def run() -> None:
        pool = FakeWorkerPool(
            {}, max_workers=1, address="tcp://127.0.0.1:0", elastic=False
        )
        worker = pool.ctx.socket(zmq.ROUTER)
        worker.setsockopt(zmq.LINGER, 0)
        worker.bind("inproc://worker")
        client = pool.ctx.socket(zmq.DEALER)
        client.setsockopt(zmq.LINGER, 0)
        client.connect(pool.address)
        broker = asyncio.create_task(pool.run())

        async def recv(socket) -> list[bytes]:
            return await asyncio.wait_for(socket.recv_multipart(), 5)

        try:
            await client.send_multipart([b"run-1", b"run", b""])
            identity, request_id, method, _ = await recv(worker)
            assert (request_id, method) == (b"run-1", b"run")
            await worker.send_multipart([identity, b"run-1", b"reply", b"done"])
            assert await recv(client) == [b"run-1", b"reply", b"done"]

            cancel = msgpack.packb({"request_id": "run-1"})
            await client.send_multipart([b"cancel-1", b"cancel", cancel])
            identity, request_id, method, payload = await recv(worker)
            assert (request_id, method, payload) == (b"cancel-1", b"cancel", cancel)

            await client.send_multipart(
                [b"cancel-2", b"cancel", msgpack.packb({"request_id": "never-ran"})]
            )
            assert await recv(client) == [b"cancel-2", b"reply", _CANCEL_MISS]
        finally:
            worker.close()
            client.close()
            broker.cancel()
            await asyncio.gather(broker, return_exceptions=True)

    asyncio.run(run())
