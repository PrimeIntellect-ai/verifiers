import asyncio
import threading

from verifiers.utils import threaded_sandbox_client as threaded_module


def test_teardown_closes_per_thread_async_clients(monkeypatch):
    ping_barrier = threading.Barrier(2)

    class FakeAsyncSandboxClient:
        instances = []

        def __init__(self, **_kwargs):
            self.closed = False
            FakeAsyncSandboxClient.instances.append(self)

        async def ping(self):
            ping_barrier.wait(timeout=2)
            return "pong"

        async def aclose(self):
            self.closed = True

    monkeypatch.setattr(
        threaded_module,
        "AsyncSandboxClient",
        FakeAsyncSandboxClient,
    )

    async def scenario():
        client = threaded_module.ThreadedAsyncSandboxClient(max_workers=2)
        try:
            assert await asyncio.gather(client.ping(), client.ping()) == [
                "pong",
                "pong",
            ]
        finally:
            client.teardown(wait=True)

    asyncio.run(scenario())

    assert len(FakeAsyncSandboxClient.instances) == 2
    assert all(client.closed for client in FakeAsyncSandboxClient.instances)
