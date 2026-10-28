import logging
from contextlib import asynccontextmanager
from types import SimpleNamespace

import httpx
import pytest

from verifiers.v1.errors import InterceptionError, TunnelError
from verifiers.v1.interception.server import InterceptionServer


class FakeTunnel:
    bind_host = "127.0.0.1"
    bind_port = 0
    tunnel_id = "tunnel-test"

    def __init__(self):
        self.exposures = 0

    @asynccontextmanager
    async def expose(self, port):
        self.exposures += 1
        yield f"http://localhost:{port}"


async def test_local_and_public_health_probes_report_server_and_tunnel(caplog):
    server = InterceptionServer()
    tunnel = FakeTunnel()
    server.tunnel = tunnel
    with caplog.at_level(logging.INFO, logger="verifiers.v1.interception.server"):
        async with server:
            async with httpx.AsyncClient() as client:
                response = await client.get(f"{server.base_url}/healthz")
            assert response.status_code == 200
            assert response.json() == {
                "status": "ok",
                "server_id": server.server_id,
                "tunnel_id": "tunnel-test",
            }
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
