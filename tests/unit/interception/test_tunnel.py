"""A prime tunnel is kept serving by the SDK through the service's health route. A tunnel
that can never work again retires its server: its rollouts fail and the pools skip it."""

import asyncio
import logging
from contextlib import asynccontextmanager
from types import SimpleNamespace

import httpx
import pytest
from prime_tunnel import TunnelGoneError

from verifiers.v1.configs.interception import (
    InterceptionServerConfig,
    StaticInterceptionPoolConfig,
)
from verifiers.v1.errors import TunnelError
from verifiers.v1.interception.pool import (
    ElasticInterceptionPool,
    StaticInterceptionPool,
)
from verifiers.v1.interception.server import InterceptionServer
from verifiers.v1.interception.tunnel import prime
from verifiers.v1.interception.tunnel.base import HEALTH_PATH


class FakeClient:
    gone = False

    def __init__(self, local_port):
        self.health_paths = []
        self.stops = 0
        type(self).instance = self

    async def start(self):
        return "https://t.example/"

    async def keep_serving(self, health_path):
        self.health_paths.append(health_path)
        if self.gone:
            raise TunnelGoneError("inactive")
        await asyncio.Future()

    def sync_stop(self):
        self.stops += 1


@asynccontextmanager
async def unlimited():
    yield


@pytest.fixture
def tunnel(monkeypatch):
    monkeypatch.setattr(prime, "ensure_prime_auth", lambda: None)
    monkeypatch.setattr(prime, "tunnel_limiter", unlimited)
    monkeypatch.setattr("prime_tunnel.Tunnel", FakeClient)
    return prime.PrimeTunnel()


async def test_held_tunnel_is_kept_serving_and_stopped_on_exit(tunnel):
    async with tunnel.expose(1234) as url:
        assert url == "https://t.example"
        await asyncio.sleep(0)
        assert FakeClient.instance.health_paths == [HEALTH_PATH]
        assert FakeClient.instance.stops == 0
    assert FakeClient.instance.stops == 1


async def test_gone_tunnel_reports_once_and_is_still_stopped_on_exit(
    tunnel, monkeypatch
):
    monkeypatch.setattr(FakeClient, "gone", True)
    reports = []
    tunnel.on_gone = lambda: reports.append(True)
    async with tunnel.expose(1234):
        await asyncio.sleep(0)
        await asyncio.sleep(0)
        assert reports == [True]
    assert FakeClient.instance.stops == 1


async def test_sdk_restarts_reach_the_run_logs(caplog):
    with caplog.at_level(logging.WARNING, logger=prime.logger.name):
        logging.getLogger("prime_tunnel.tunnel").warning(
            "Tunnel %s: frpc restarted", "https://t"
        )
    assert "tunnel https://t: frpc restarted" in caplog.messages


async def test_server_answers_the_health_route():
    async with InterceptionServer() as server, httpx.AsyncClient() as client:
        response = await client.get(f"{server.base_url}{HEALTH_PATH}")
    assert response.status_code == 200


def test_gone_tunnel_fails_the_servers_rollouts_and_refuses_new_ones():
    server = InterceptionServer()
    released = []
    session = SimpleNamespace(error=None, release=lambda: released.append(True))
    server.sessions["secret"] = session

    server._handle_tunnel_gone()

    assert isinstance(session.error, TunnelError)
    assert released == [True]
    with pytest.raises(TunnelError):
        server.register(SimpleNamespace())


async def test_elastic_pool_skips_a_retired_server():
    pool = ElasticInterceptionPool()
    retired = SimpleNamespace(healthy=False, load=0)
    live = SimpleNamespace(healthy=True, load=0)
    pool.servers = [retired, live]
    assert await pool._server() is live


async def test_static_pool_with_only_retired_servers_fails_the_rollout():
    pool = StaticInterceptionPool(
        StaticInterceptionPoolConfig(servers=[InterceptionServerConfig()])
    )
    pool.servers[0].healthy = False
    with pytest.raises(TunnelError):
        async with pool.acquire(SimpleNamespace()):
            pass
