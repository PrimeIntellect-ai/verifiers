"""Application-route containment with fixture HTTP, no sockets or provider calls."""

import asyncio
import json
from contextlib import asynccontextmanager
from types import SimpleNamespace

import httpx
import pytest
from aiohttp.test_utils import make_mocked_request

from verifiers.v1.errors import InterceptionError
from verifiers.v1.interception import server as server_module
from verifiers.v1.interception.server import InterceptionServer
from verifiers.v1.interception.tunnel.base import Tunnel


class Session:
    def __init__(self, name):
        self.ctx = SimpleNamespace(client=object())
        self.trace = SimpleNamespace(id=name)
        self.released = False
        self.error = None

    def release(self):
        self.released = True


@pytest.fixture
def network(monkeypatch):
    record = SimpleNamespace(tunnels=[], runners=[], requests=[], override=None)
    by_port = {}

    class FakeTunnel(Tunnel):
        def __init__(self):
            super().__init__()
            self.closed = False
            record.tunnels.append(self)

        @asynccontextmanager
        async def expose(self, port):
            try:
                yield f"https://fixture-{port}.invalid"
            finally:
                self.closed = True

    class Runner:
        def __init__(self, app):
            self.app = app
            self.closed = False
            record.runners.append(self)

        async def setup(self):
            pass

        async def cleanup(self):
            self.closed = True

    class Site:
        def __init__(self, runner, host, port):
            self.port = 10000 + len(by_port)
            by_port[self.port] = runner
            self._server = SimpleNamespace(
                sockets=[SimpleNamespace(getsockname=lambda: (host, self.port))]
            )

        async def start(self):
            pass

    class HTTP:
        def __init__(self, *, timeout):
            assert timeout == httpx.Timeout(5.0, connect=2.0)

        async def __aenter__(self):
            return self

        async def __aexit__(self, *exc):
            pass

        async def get(self, url):
            record.requests.append(url)
            assert url.endswith("/__interception/healthz")
            port = (
                int(url.split("fixture-", 1)[1].split(".", 1)[0])
                if "fixture-" in url
                else httpx.URL(url).port
            )
            runner = by_port[port]
            request = make_mocked_request("GET", "/__interception/healthz")
            match = await runner.app.router.resolve(request)
            response = await match.handler(request)
            reply = httpx.Response(response.status, content=response.body)
            return record.override(url, reply) if record.override else reply

    monkeypatch.setattr(server_module, "make_tunnel", lambda config: FakeTunnel())
    monkeypatch.setattr(server_module.web, "AppRunner", Runner)
    monkeypatch.setattr(server_module.web, "TCPSite", Site)
    monkeypatch.setattr(server_module.httpx, "AsyncClient", HTTP)
    monkeypatch.setattr(InterceptionServer, "_client", lambda self, config: object())
    return record


async def test_tunneled_start_validates_local_and_public_identity(network):
    async with InterceptionServer(requires_tunnel=True) as server:
        assert len(network.requests) == 2
        assert network.requests == [
            f"http://127.0.0.1:{server.port}/__interception/healthz",
            f"{server.base_url}/__interception/healthz",
        ]
        assert server.healthy and server._health_watch is not None
    assert server._health_watch.done()
    assert network.runners[0].closed and network.tunnels[0].closed


async def test_local_only_start_does_not_probe_or_watch(network):
    async with InterceptionServer() as server:
        assert network.requests == []
        assert server._health_watch is None


@pytest.mark.parametrize("scope", ["local", "public"])
@pytest.mark.parametrize("failure", ["plain-ok", "server", "tunnel", "status", "list"])
async def test_start_rejects_wrong_route_or_identity_and_cleans_up(
    network, scope, failure
):
    def override(url, reply):
        if ("fixture-" in url) != (scope == "public"):
            return reply
        if failure == "plain-ok":
            return httpx.Response(200, text="OK")
        if failure == "list":
            return httpx.Response(200, json=[])
        body = reply.json()
        body[
            {"server": "server_id", "tunnel": "tunnel_id", "status": "status"}[failure]
        ] = "wrong"
        return httpx.Response(200, json=body)

    network.override = override
    server = InterceptionServer(requires_tunnel=True)
    with pytest.raises(InterceptionError, match="startup health"):
        async with server:
            pytest.fail("an invalid application identity was admitted")
    assert not server.healthy and server._health_watch is None
    assert network.tunnels[0].closed and network.runners[0].closed


async def test_health_endpoint_reports_quarantine_and_rejects_new_registrations(
    network,
):
    async with InterceptionServer(requires_tunnel=True) as server:
        session = Session("existing")
        keys = server.register(session)
        server.healthy = False
        reply = await server.handle_health(
            make_mocked_request("GET", "/__interception/healthz")
        )
        assert reply.status == 503
        assert json.loads(reply.body)["status"] == "unhealthy"
        with pytest.raises(InterceptionError, match="unavailable"):
            server.register(Session("new"))
        assert server.load == 1 and not session.released and session.error is None
        server.unregister(*keys)
        assert session.released


async def test_cancelled_server_acquire_unregisters_normally(network):
    async with InterceptionServer(requires_tunnel=True) as server:
        entered = asyncio.Event()
        session = Session("cancelled")

        async def run():
            async with server.acquire(session):
                entered.set()
                await asyncio.Event().wait()

        task = asyncio.create_task(run())
        await entered.wait()
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert server.load == 0 and session.released
        assert session.error is None
