from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from verifiers.v1.errors import (
    HarnessError,
    TunnelUnavailableError,
    is_tunnel_unavailable_detail,
)
from verifiers.v1.harness import Harness
from verifiers.v1.interception.pool import (
    ElasticInterceptionPool,
    ElasticInterceptionPoolConfig,
)
from verifiers.v1.interception.server import InterceptionServer
from verifiers.v1.interception.tunnel.prime import PrimeTunnel
from verifiers.v1.runtimes import ProgramResult


@pytest.mark.asyncio
async def test_prime_tunnel_process_alive_but_registration_absent(monkeypatch):
    monkeypatch.setattr(
        "verifiers.v1.interception.tunnel.prime.ensure_prime_auth", lambda: None
    )
    tunnel = PrimeTunnel()
    tunnel._client = SimpleNamespace(
        is_running=True,
        check_registered=AsyncMock(return_value=False),
    )

    assert await tunnel.healthy() is False


@pytest.mark.asyncio
async def test_prime_tunnel_endpoint_404_then_recovers(monkeypatch):
    monkeypatch.setattr(
        "verifiers.v1.interception.tunnel.prime.ensure_prime_auth", lambda: None
    )
    tunnel = PrimeTunnel()
    tunnel._client = SimpleNamespace(
        is_running=True,
        url="https://stable.tunnel.example",
        check_registered=AsyncMock(return_value=True),
    )
    probe = MagicMock()
    probe.__aenter__ = AsyncMock(return_value=probe)
    probe.__aexit__ = AsyncMock(return_value=None)
    probe.get = AsyncMock(
        side_effect=[
            SimpleNamespace(status_code=404),
            SimpleNamespace(status_code=401),
        ]
    )
    monkeypatch.setattr(
        "verifiers.v1.interception.tunnel.prime.httpx.AsyncClient",
        lambda **_kwargs: probe,
    )

    assert await tunnel.healthy() is False
    assert await tunnel.healthy() is True


@pytest.mark.asyncio
async def test_tunnel_registration_recovers_without_replacing_active_endpoint():
    server = InterceptionServer(requires_tunnel=False)
    server.base_url = "https://stable.tunnel.example"
    server.tunnel = MagicMock()
    server.tunnel.healthy = AsyncMock(side_effect=[False, True])
    server.tunnel.reconnect = AsyncMock()
    server.sessions["active"] = MagicMock()

    await server.refresh_health()
    assert server.healthy is False

    await server.refresh_health()
    assert server.healthy is True
    assert server.base_url == "https://stable.tunnel.example"
    server.tunnel.reconnect.assert_not_awaited()


@pytest.mark.asyncio
async def test_idle_unhealthy_tunnel_is_reregistered_with_new_url():
    server = InterceptionServer(requires_tunnel=False)
    server.base_url = "https://stale.tunnel.example"
    server.port = 8123
    server.tunnel = MagicMock()
    server.tunnel.healthy = AsyncMock(return_value=False)
    server.tunnel.reconnect = AsyncMock(
        return_value="https://replacement.tunnel.example"
    )

    await server.refresh_health()

    assert server.healthy is True
    assert server.base_url == "https://replacement.tunnel.example"
    server.tunnel.reconnect.assert_awaited_once_with(8123)


@pytest.mark.asyncio
async def test_elastic_pool_skips_quarantined_server():
    pool = ElasticInterceptionPool(ElasticInterceptionPoolConfig(multiplex=32))
    stale = InterceptionServer(requires_tunnel=False)
    stale.base_url = "https://stale.tunnel.example"
    healthy = InterceptionServer(requires_tunnel=False)
    healthy.base_url = "https://healthy.tunnel.example"
    pool.servers = [stale, healthy]

    pool.quarantine(stale.base_url, "fault injection")

    assert await pool._server() is healthy
    assert stale.healthy is False


@pytest.mark.parametrize(
    "detail",
    [
        "<html><h1>Tunnel not found or no longer active</h1></html>",
        "POST https://abc.prime-tunnel.example: HTTP 504 Gateway Timeout",
    ],
)
def test_prime_tunnel_gateway_failures_are_classified(detail):
    assert is_tunnel_unavailable_detail(detail)


def test_unrelated_gateway_failure_is_not_classified_as_tunnel():
    assert not is_tunnel_unavailable_detail("provider returned HTTP 504")


@pytest.mark.asyncio
async def test_harness_tunnel_404_is_typed_before_runtime_probe():
    harness = MagicMock()
    harness.config.id = "terminus-2"
    runtime = MagicMock()
    runtime.alive = AsyncMock()
    trace = MagicMock(stop_condition=None)
    result = ProgramResult(
        exit_code=1,
        stdout="",
        stderr="<html>Tunnel not found or no longer active</html>",
    )

    with pytest.raises(TunnelUnavailableError):
        await Harness._check_result(harness, trace, runtime, result)

    runtime.alive.assert_not_awaited()


@pytest.mark.asyncio
async def test_deterministic_harness_error_remains_harness_error():
    harness = MagicMock()
    harness.config.id = "terminus-2"
    runtime = MagicMock()
    runtime.alive = AsyncMock(return_value=True)
    trace = MagicMock(stop_condition=None)
    result = ProgramResult(exit_code=2, stdout="", stderr="invalid task input")

    with pytest.raises(HarnessError, match="invalid task input"):
        await Harness._check_result(harness, trace, runtime, result)
