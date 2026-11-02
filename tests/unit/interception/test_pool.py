"""Interception pool selection must not reuse a slot whose tunnel died."""

import asyncio
import gc
import weakref
from types import SimpleNamespace

import pytest

from verifiers.v1.errors import InterceptionError
from verifiers.v1.interception.pool import (
    ElasticInterceptionPool,
    ElasticInterceptionPoolConfig,
)
from verifiers.v1.interception.server import InterceptionServer


class Slot:
    def __init__(self, healthy, load=0):
        self.healthy = healthy
        self.load = load
        self.stops = 0

    async def stop(self):
        self.stops += 1


@pytest.mark.asyncio
async def test_elastic_server_skips_unhealthy_slots():
    pool = ElasticInterceptionPool(ElasticInterceptionPoolConfig(multiplex=1))
    healthy = Slot(healthy=True)
    dead = Slot(healthy=False)
    pool.servers = [dead, healthy]
    assert await pool._server() is healthy
    assert dead.stops == 1
    assert pool.servers == [healthy]


@pytest.mark.asyncio
async def test_elastic_acquire_rejects_unhealthy_server_before_mutation(monkeypatch):
    pool = ElasticInterceptionPool(ElasticInterceptionPoolConfig(multiplex=1))
    server = InterceptionServer()
    server.healthy = False
    server.base_url = "https://dead.example"

    async def dead_server():
        return server

    monkeypatch.setattr(pool, "_server", dead_server)
    monkeypatch.setattr(
        server, "_client", lambda config: pytest.fail("client assigned")
    )
    session = SimpleNamespace(
        ctx=SimpleNamespace(client=object()),
        trace=SimpleNamespace(id="trace-test"),
        client=None,
    )
    with pytest.raises(InterceptionError, match="unavailable"):
        async with pool.acquire(session):
            pytest.fail("unhealthy slot acquired")
    assert session.client is None
    assert server.sessions == {}


@pytest.mark.asyncio
async def test_elastic_quarantined_slot_drains_before_cleanup():
    pool = ElasticInterceptionPool(ElasticInterceptionPoolConfig(multiplex=1))
    draining = Slot(healthy=False, load=1)
    healthy = Slot(healthy=True)
    pool.servers = [draining, healthy]
    assert await pool._server() is healthy
    assert draining.stops == 0
    assert pool.servers == [draining, healthy]
    draining.load = 0
    await pool._prune()
    assert draining.stops == 1
    assert pool.servers == [healthy]


@pytest.mark.asyncio
async def test_last_acquire_exit_cleans_quarantined_slot(monkeypatch):
    pool = ElasticInterceptionPool(ElasticInterceptionPoolConfig(multiplex=2))
    server = Slot(healthy=True)
    server.base_url = "https://example.invalid"

    def register(session):
        server.load += 1
        return "model", "state"

    def unregister(model, state):
        server.load -= 1

    monkeypatch.setattr(server, "register", register, raising=False)
    monkeypatch.setattr(server, "unregister", unregister, raising=False)
    pool.servers = [server]
    async with pool.acquire(object()):
        async with pool.acquire(object()):
            server.healthy = False
        assert server.stops == 0
        assert server.load == 1
    assert server.stops == 1
    assert pool.servers == []


@pytest.mark.asyncio
@pytest.mark.parametrize("startup_healthy", [True, False])
async def test_retired_servers_are_not_retained_by_pool(monkeypatch, startup_healthy):
    retired = []

    class StartedSlot(Slot):
        def __init__(self, *args, **kwargs):
            super().__init__(healthy=startup_healthy)
            self.base_url = "https://example.invalid"
            retired.append(weakref.ref(self))

        async def __aenter__(self):
            return self

        async def __aexit__(self, *exc):
            await self.stop()

    monkeypatch.setattr(
        "verifiers.v1.interception.pool.InterceptionServer", StartedSlot
    )
    pool = ElasticInterceptionPool()
    for _ in range(3):
        if startup_healthy:
            server = await pool._server()
            server.healthy = False
            await pool._prune()
            assert server.stops == 1
            del server
        else:
            with pytest.raises(InterceptionError, match="startup health checks"):
                await pool._server()
        assert pool.servers == []
    gc.collect()
    assert all(reference() is None for reference in retired)
    await pool.stop()


@pytest.mark.asyncio
async def test_pool_shutdown_stops_all_members_after_one_cleanup_fails(monkeypatch):
    pool = ElasticInterceptionPool()
    first, broken, last = Slot(True), Slot(True), Slot(True)

    async def failed_stop():
        broken.stops += 1
        raise RuntimeError("cleanup failed")

    monkeypatch.setattr(broken, "stop", failed_stop)
    pool.servers = [first, broken, last]
    with pytest.raises(RuntimeError, match="cleanup failed"):
        await pool.stop()
    assert [server.stops for server in (first, broken, last)] == [1, 1, 1]
    assert pool.servers == []


@pytest.mark.asyncio
async def test_cancelled_prune_keeps_members_owned_until_shutdown(monkeypatch):
    pool = ElasticInterceptionPool()
    first, second = Slot(False), Slot(False)
    pool.servers = [first, second]
    stopping = asyncio.Event()
    finish = asyncio.Event()

    async def blocked_stop():
        first.stops += 1
        stopping.set()
        await finish.wait()

    monkeypatch.setattr(first, "stop", blocked_stop)
    pruning = asyncio.create_task(pool._prune())
    await asyncio.wait_for(stopping.wait(), 1)
    pruning.cancel()
    with pytest.raises(asyncio.CancelledError):
        await pruning
    assert pool.servers == [first, second]
    assert second.stops == 0
    finish.set()
    await pool.stop()
    assert [first.stops, second.stops] == [2, 1]
    assert pool.servers == []
