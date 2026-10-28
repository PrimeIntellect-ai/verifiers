"""Host servers a remote harness reaches share one host tunnel: each gets its own route
through it, sessions that overlap or follow each other reuse it, and nothing stays open
once the routes are gone. A Prime tunnel the process still holds when it exits is
stopped then."""

import asyncio
import contextlib

import aiohttp
import pytest
from aiohttp import web

from verifiers.v1.interception.tunnel import prime, shared
from verifiers.v1.interception.tunnel.base import Tunnel
from verifiers.v1.mcp import launch
from verifiers.v1.runtimes import SubprocessConfig, provision_runtime


class LoopbackTunnel(Tunnel):
    """A tunnel that publishes the host port as itself, recording what it opened."""

    def __init__(self, record: dict) -> None:
        super().__init__()
        self.record = record

    @contextlib.asynccontextmanager
    async def expose(self, port):
        self.record["opened"].append(port)
        try:
            yield f"http://127.0.0.1:{port}"
        finally:
            self.record["closed"].append(port)


@pytest.fixture
def tunnels():
    record = {"opened": [], "closed": []}
    return record, shared.SharedHostTunnel(lambda: LoopbackTunnel(record))


@contextlib.asynccontextmanager
async def server(name: str):
    """A host server that answers with its name, echoes a POST body, and streams."""

    async def hello(request):
        return web.Response(text=f"{name} {request.query.get('q', '')}")

    async def echo(request):
        return web.Response(body=await request.read(), headers={"mcp-session-id": name})

    async def stream(request):
        response = web.StreamResponse(headers={"content-type": "text/event-stream"})
        await response.prepare(request)
        for n in range(3):
            await response.write(f"data: {n}\n\n".encode())
            await asyncio.sleep(0.01)
        return response

    app = web.Application()
    app.router.add_get("/mcp", hello)
    app.router.add_post("/mcp", echo)
    app.router.add_get("/stream", stream)
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, "127.0.0.1", 0)
    await site.start()
    try:
        yield site._server.sockets[0].getsockname()[1]
    finally:
        await runner.cleanup()


async def test_a_server_is_reached_through_its_route(tunnels):
    _, host = tunnels
    async with (
        server("alpha") as port,
        host.route(port) as url,
        aiohttp.ClientSession() as http,
    ):
        async with http.get(f"{url}/mcp?q=hi") as reply:
            assert await reply.text() == "alpha hi"
        async with http.post(f"{url}/mcp", data=b'{"jsonrpc": "2.0"}') as reply:
            assert await reply.read() == b'{"jsonrpc": "2.0"}'
            assert reply.headers["mcp-session-id"] == "alpha"
        async with http.get(f"{url}/stream") as reply:
            assert await reply.text() == "data: 0\n\ndata: 1\n\ndata: 2\n\n"
    await host.close()


async def test_two_sessions_share_one_tunnel_and_reach_only_their_own_server(
    tunnels,
):
    record, host = tunnels
    async with (
        server("alpha") as a,
        server("beta") as b,
        host.route(a) as url_a,
        host.route(b) as url_b,
    ):
        assert len(record["opened"]) == 1
        async with aiohttp.ClientSession() as http:
            async with http.get(f"{url_a}/mcp") as reply:
                assert (await reply.text()).startswith("alpha")
            async with http.get(f"{url_b}/mcp") as reply:
                assert (await reply.text()).startswith("beta")
            gateway = url_a.rsplit("/", 1)[0]
            async with http.get(f"{gateway}/not-a-route/mcp") as reply:
                assert reply.status == 404
    await host.close()


async def test_a_following_session_reuses_the_tunnel_and_an_idle_one_closes(
    tunnels, monkeypatch
):
    monkeypatch.setattr(shared, "IDLE_SECONDS", 0.2)
    record, host = tunnels
    async with server("alpha") as port:
        async with host.route(port):
            pass
        async with host.route(port):
            pass
        assert len(record["opened"]) == 1 and record["closed"] == []
        await asyncio.sleep(0.4)
    assert record["closed"] == record["opened"]
    assert host.gateways == []


async def test_closing_the_shared_tunnel_closes_held_routes_tunnels(tunnels):
    record, host = tunnels
    async with server("alpha") as port, host.route(port):
        await host.close()
        assert record["closed"] == record["opened"]


async def test_a_full_gateway_opens_another_tunnel(tunnels, monkeypatch):
    monkeypatch.setattr(shared, "MULTIPLEX", 1)
    record, host = tunnels
    async with server("alpha") as port, host.route(port), host.route(port):
        assert len(record["opened"]) == 2
    await host.close()
    assert sorted(record["closed"]) == sorted(record["opened"])


async def test_host_tool_servers_of_remote_sessions_share_the_processs_tunnel(
    tunnels, monkeypatch
):
    record, host = tunnels
    monkeypatch.setattr(launch, "shared_host_tunnel", lambda: host)
    async with (
        server("alpha") as a,
        server("beta") as b,
        provision_runtime(SubprocessConfig()) as runtime,
    ):
        reach = [
            launch.reachable_url(
                runtime, port, colocated=False, consumer_is_local=False
            )
            for port in (a, b)
        ]
        async with reach[0] as url_a, reach[1] as url_b:
            assert url_a != url_b
            assert len(record["opened"]) == 1
            async with (
                aiohttp.ClientSession() as http,
                http.get(f"{url_b}/mcp") as reply,
            ):
                assert (await reply.text()).startswith("beta")
    await host.close()


class FakeClient:
    """A prime_tunnel client that starts, and records being stopped."""

    stopped: list["FakeClient"] = []  # noqa: RUF012 - shared record across instances

    def __init__(self, local_port, labels):
        self.is_running, self.is_gone = True, False

    async def start(self):
        return "https://t.example"

    def sync_stop(self):
        FakeClient.stopped.append(self)


async def test_a_tunnel_held_at_exit_is_stopped(monkeypatch):
    import prime_tunnel

    monkeypatch.setattr(prime_tunnel, "Tunnel", FakeClient)
    monkeypatch.setattr(prime, "ensure_prime_auth", lambda: None)
    FakeClient.stopped = []
    async with prime.PrimeTunnel().expose(1234):
        # A signal unwinds the loop here: the atexit backstop stops the tunnel.
        prime._stop_at_exit()
        assert len(FakeClient.stopped) == 1
    # Closed normally, it is no longer live, so exit stops nothing more.
    prime._stop_at_exit()
    assert len(FakeClient.stopped) == 2
    assert prime._LIVE == set()
