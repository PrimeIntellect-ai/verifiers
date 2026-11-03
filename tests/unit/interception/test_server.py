import asyncio
import logging
from contextlib import asynccontextmanager
from types import SimpleNamespace
from unittest.mock import AsyncMock

import httpx
import pytest

from verifiers.v1.errors import InterceptionError, TunnelError
from verifiers.v1.interception import server as server_module
from verifiers.v1.interception.server import InterceptionServer


class FakeTunnel:
    bind_host = "127.0.0.1"
    bind_port = 0
    tunnel_id = "tunnel-test"

    def __init__(self):
        self.exposures = 0
        self.closed = False

    @asynccontextmanager
    async def expose(self, port):
        self.exposures += 1
        try:
            yield f"http://localhost:{port}"
        finally:
            self.closed = True


async def test_local_and_public_health_probes_report_server_and_tunnel(caplog):
    server = InterceptionServer()
    tunnel = FakeTunnel()
    server.tunnel = tunnel
    with caplog.at_level(logging.INFO, logger="verifiers.v1.interception.server"):
        async with server:
            async with httpx.AsyncClient() as client:
                response = await client.get(f"{server.base_url}/healthz")
                app_health = await client.get(
                    f"{server.base_url}/__interception/healthz"
                )
            assert response.status_code == 200
            assert response.json() == {
                "status": "ok",
                "server_id": server.server_id,
                "tunnel_id": "tunnel-test",
            }
            assert app_health.status_code == 200
            assert app_health.json()["server_id"] == server.server_id
    assert "scope=local" in caplog.text
    assert "scope=public" in caplog.text
    assert "probe=local" in caplog.text
    assert "probe=public" in caplog.text
    assert "matched=True" in caplog.text
    assert f"server_id={server.server_id}" in caplog.text
    assert "tunnel_id=tunnel-test" in caplog.text


async def test_unmatched_not_found_is_logged_without_payload_or_recreation(caplog):
    server = InterceptionServer()
    tunnel = FakeTunnel()
    server.tunnel = tunnel
    async with server:
        caplog.clear()
        with caplog.at_level(logging.INFO, logger="verifiers.v1.interception.server"):
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    f"{server.base_url}/v1/wrong-path?key=url-secret",
                    content=b"prompt-secret",
                    headers={
                        "Authorization": "Bearer auth-secret",
                        "X-ACP-Model-Request-ID": "request-1",
                    },
                )
    assert response.status_code == 404
    assert "status=404 route=unmatched" in caplog.text
    assert "request_id='request-1'" in caplog.text
    assert tunnel.exposures == 1
    for secret in ("prompt-secret", "auth-secret", "url-secret"):
        assert secret not in caplog.text


class FakeSession:
    def __init__(self, trace_id="trace-test"):
        self.ctx = SimpleNamespace(client=object())
        self.trace = SimpleNamespace(id=trace_id)
        self.client = None
        self.error = None
        self.released = False

    def release(self):
        self.released = True


async def test_bad_health_probes_return_false():
    responder = InterceptionServer()
    observer = InterceptionServer()
    async with responder:
        # A reachable route for another interception server is not a healthy route
        # for this one, even when its payload is otherwise valid.
        assert not await observer._probe_health("local", responder.base_url)

        observer.server_id = responder.server_id
        responder.healthy = False
        assert not await observer._probe_health("local", responder.base_url)
        base_url = responder.base_url

    assert not await observer._probe_health("local", base_url)


async def test_failed_startup_probe_rejects_registration_before_mutation(monkeypatch):
    server = InterceptionServer()
    server.tunnel = FakeTunnel()
    watch = AsyncMock()
    monkeypatch.setattr(server, "_watch_health", watch)
    monkeypatch.setattr(server_module, "STARTUP_PROBE_RETRY_DELAY_SECONDS", 0)

    async def failed_probe(scope, base_url):
        return False

    monkeypatch.setattr(server, "_probe_health", failed_probe)
    monkeypatch.setattr(
        server, "_client", lambda config: pytest.fail("client assigned")
    )
    session = FakeSession()
    async with server:
        assert not server.healthy
        with pytest.raises(InterceptionError, match="unavailable"):
            async with server.acquire(session):
                pytest.fail("unhealthy slot acquired")
    assert session.client is None
    assert server.sessions == {}
    watch.assert_not_called()


async def test_startup_public_probe_retries_transient_failures(monkeypatch):
    server = InterceptionServer()
    server.tunnel = FakeTunnel()
    monkeypatch.setattr(server, "_watch_health", AsyncMock())
    monkeypatch.setattr(server_module, "STARTUP_PROBE_RETRY_DELAY_SECONDS", 0)
    calls = []

    async def probe(scope, base_url):
        calls.append(scope)
        # The fresh tunnel's public edge refuses the first two connects.
        return scope == "local" or calls.count("public") >= 3

    monkeypatch.setattr(server, "_probe_health", probe)
    async with server:
        assert server.healthy
        assert calls == ["local", "public", "public", "public"]


async def test_startup_probe_gives_up_after_configured_attempts(monkeypatch):
    server = InterceptionServer()
    server.tunnel = FakeTunnel()
    monkeypatch.setattr(server, "_watch_health", AsyncMock())
    monkeypatch.setattr(server_module, "STARTUP_PROBE_RETRY_DELAY_SECONDS", 0)
    calls = []

    async def probe(scope, base_url):
        calls.append(scope)
        return scope == "local"

    monkeypatch.setattr(server, "_probe_health", probe)
    async with server:
        assert not server.healthy
        assert calls == ["local"] + ["public"] * server_module.STARTUP_PROBE_ATTEMPTS


async def test_startup_probe_skips_public_when_local_fails(monkeypatch):
    server = InterceptionServer()
    server.tunnel = FakeTunnel()
    monkeypatch.setattr(server, "_watch_health", AsyncMock())
    monkeypatch.setattr(server_module, "STARTUP_PROBE_RETRY_DELAY_SECONDS", 0)
    calls = []

    async def probe(scope, base_url):
        calls.append(scope)
        return False

    monkeypatch.setattr(server, "_probe_health", probe)
    async with server:
        assert not server.healthy
        assert calls == ["local"] * server_module.STARTUP_PROBE_ATTEMPTS


async def test_startup_probe_stops_retrying_after_tunnel_loss(monkeypatch):
    server = InterceptionServer()
    server.tunnel = FakeTunnel()
    monkeypatch.setattr(server, "_watch_health", AsyncMock())
    monkeypatch.setattr(server_module, "STARTUP_PROBE_RETRY_DELAY_SECONDS", 0)
    calls = []

    async def probe(scope, base_url):
        calls.append(scope)
        if scope == "public":
            server.healthy = False  # terminal tunnel-gone verdict during the probe
            return False
        return True

    monkeypatch.setattr(server, "_probe_health", probe)
    async with server:
        assert not server.healthy
        assert calls == ["local", "public"]


def test_health_probe_allows_five_second_connect():
    assert server_module.HEALTH_PROBE_CONNECT_TIMEOUT_SECONDS == 5.0
    assert server_module.STARTUP_PROBE_ATTEMPTS == 3


@pytest.fixture
def monitored_server(monkeypatch):
    server = InterceptionServer()
    server.tunnel = FakeTunnel()
    monkeypatch.setattr(server_module, "HEALTH_CHECK_INTERVAL_SECONDS", 0)
    checks = asyncio.Queue()
    completed = asyncio.Queue()
    startup = True
    public_healthy = True

    async def probe(scope, base_url):
        nonlocal startup, public_healthy
        if startup:
            if scope == "public":
                startup = False
            return True
        if scope == "local":
            local_healthy, public_healthy = await checks.get()
            return local_healthy
        completed.put_nowait(None)
        return public_healthy

    async def check(local_healthy=True, public_healthy=True):
        checks.put_nowait((local_healthy, public_healthy))
        await completed.get()

    monkeypatch.setattr(server, "_probe_health", probe)
    return server, check


async def test_transient_health_failure_resets_consecutive_failures(monitored_server):
    server, check = monitored_server
    async with server:
        await check(public_healthy=False)
        assert server.healthy
        await check()
        await check(public_healthy=False)
        assert server.healthy
        await check()
        assert server.healthy


async def test_persistent_public_health_failure_quarantines_without_releasing(
    monitored_server, monkeypatch
):
    server, check = monitored_server
    monkeypatch.setattr(server, "_client", lambda config: object())
    session = FakeSession()
    async with server:
        async with server.acquire(session):
            await check(public_healthy=False)
            assert server.healthy
            await check(public_healthy=False)
            assert not server.healthy
            assert server._health_watch is not None
            assert server._health_watch.done()
            assert not session.released
            assert session.error is None
            assert server.load == 1
            assert not server.tunnel.closed
            replacement = FakeSession("replacement")
            with pytest.raises(InterceptionError, match="unavailable"):
                server.register(replacement)
            assert replacement.client is None
            async with httpx.AsyncClient() as client:
                response = await client.get(
                    f"http://127.0.0.1:{server.port}/__interception/healthz"
                )
            assert response.status_code == 503
            assert response.json()["server_id"] == server.server_id
        assert session.released


async def test_stop_cancels_and_awaits_health_watch(monkeypatch):
    server = InterceptionServer()
    server.tunnel = FakeTunnel()
    monkeypatch.setattr(server, "_probe_health", AsyncMock(return_value=True))
    started = asyncio.Event()
    cancelled = asyncio.Event()

    async def watch():
        started.set()
        try:
            await asyncio.Event().wait()
        finally:
            cancelled.set()

    monkeypatch.setattr(server, "_watch_health", watch)
    async with server:
        await started.wait()
        assert server._health_watch is not None
        task = server._health_watch
    assert cancelled.is_set()
    assert task.cancelled()
    assert server.tunnel.closed


async def test_local_server_does_not_start_health_watch(monkeypatch):
    server = InterceptionServer()
    monkeypatch.setattr(server, "_probe_health", AsyncMock(return_value=True))
    watch = AsyncMock()
    monkeypatch.setattr(server, "_watch_health", watch)
    async with server:
        assert server.healthy
    watch.assert_not_called()


async def test_tunnel_gone_fails_session_and_direct_reacquisition(monkeypatch):
    server = InterceptionServer()
    monkeypatch.setattr(server, "_client", lambda config: object())
    session = FakeSession()
    async with server:
        async with server.acquire(session):
            await server._handle_tunnel_gone()
            assert session.released
            assert isinstance(session.error, TunnelError)
            assert str(session.error) == "interception tunnel lost"

        replacement = FakeSession("replacement")
        with pytest.raises(InterceptionError, match="unavailable"):
            async with server.acquire(replacement):
                pytest.fail("dead URL reacquired")
        assert replacement.client is None
        assert replacement.error is None
        assert server.sessions == {}
