"""Interception pool selection must not reuse a slot whose tunnel died."""

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
