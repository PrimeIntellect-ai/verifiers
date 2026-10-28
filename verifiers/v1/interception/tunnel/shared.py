"""One host tunnel for many host-local servers.

A remote harness reaches a host-local tool server through a host tunnel. One tunnel per
server per session spends the account's tunnel creations (Prime caps them per hour) and
leaves a tunnel behind for every session a crash cuts short. Here every host port a
process publishes shares one tunnel instead: a loopback reverse proxy routes
`/<token>/<path>` to its port, and the tunnel exposes the proxy. The token is random and
names one route, so a URL reaches only the server it was issued for, as a tunnel URL of
its own did.

A gateway serves up to `MULTIPLEX` routes; past that another one opens. One left without
routes closes after `IDLE_SECONDS`, so sessions that follow each other reuse it instead
of opening a tunnel each.
"""

import asyncio
import contextlib
import logging
import secrets
from collections.abc import AsyncIterator, Callable

from aiohttp import ClientSession, ClientTimeout, web

from verifiers.v1.interception.tunnel.base import Tunnel

logger = logging.getLogger(__name__)

MULTIPLEX = 128
"""Routes one gateway (one tunnel) serves."""
IDLE_SECONDS = 30.0
"""How long a gateway without routes stays open for the next one."""

HOP_BY_HOP = {
    "connection",
    "keep-alive",
    "proxy-authenticate",
    "proxy-authorization",
    "te",
    "trailer",
    "transfer-encoding",
    "upgrade",
    "host",
    "content-length",
}


def forwarded(headers) -> dict[str, str]:
    return {k: v for k, v in headers.items() if k.lower() not in HOP_BY_HOP}


class HostGateway:
    """A loopback reverse proxy behind one tunnel: `/<token>/<path>` reaches
    `http://127.0.0.1:<port>/<path>` for the port registered under `token`."""

    def __init__(self, tunnel: Tunnel) -> None:
        self.tunnel = tunnel
        self.routes: dict[str, int] = {}
        self.url = ""
        self.stack = contextlib.AsyncExitStack()
        self.idle: asyncio.TimerHandle | None = None
        self.client: ClientSession | None = None

    async def start(self) -> None:
        async with contextlib.AsyncExitStack() as stack:
            self.client = await stack.enter_async_context(
                ClientSession(auto_decompress=False, timeout=ClientTimeout(total=None))
            )
            app = web.Application(client_max_size=0)
            app.router.add_route("*", "/{token}/{tail:.*}", self.handle)
            runner = web.AppRunner(app)
            await runner.setup()
            stack.push_async_callback(runner.cleanup)
            site = web.TCPSite(runner, "127.0.0.1", 0)
            await site.start()
            port = site._server.sockets[0].getsockname()[1]  # type: ignore[union-attr]
            self.url = await stack.enter_async_context(self.tunnel.expose(port))
            self.stack = stack.pop_all()

    async def close(self) -> None:
        if self.idle is not None:
            self.idle.cancel()
        await self.stack.aclose()

    def add(self, port: int) -> str:
        token = secrets.token_urlsafe(24)
        self.routes[token] = port
        if self.idle is not None:
            self.idle.cancel()
            self.idle = None
        return token

    async def handle(self, request: web.Request) -> web.StreamResponse:
        port = self.routes.get(request.match_info["token"])
        if port is None:
            raise web.HTTPNotFound()
        target = f"http://127.0.0.1:{port}/{request.match_info['tail']}"
        if request.query_string:
            target += f"?{request.query_string}"
        assert self.client is not None
        async with self.client.request(
            request.method,
            target,
            headers=forwarded(request.headers),
            data=request.content if request.body_exists else None,
            allow_redirects=False,
        ) as upstream:
            response = web.StreamResponse(
                status=upstream.status, headers=forwarded(upstream.headers)
            )
            await response.prepare(request)
            async for chunk in upstream.content.iter_any():
                await response.write(chunk)
            await response.write_eof()
            return response


class SharedHostTunnel:
    """The process's gateways, opened on demand from `make_tunnel` (one per event
    loop: see `shared_host_tunnel`)."""

    def __init__(self, make_tunnel: Callable[[], Tunnel]) -> None:
        self.make_tunnel = make_tunnel
        self.gateways: list[HostGateway] = []
        self.lock = asyncio.Lock()

    @contextlib.asynccontextmanager
    async def route(self, port: int) -> AsyncIterator[str]:
        """Yield a public URL that reaches the host's `port` while held."""
        async with self.lock:
            gateway = next(
                (g for g in self.gateways if len(g.routes) < MULTIPLEX), None
            )
            if gateway is None:
                gateway = HostGateway(self.make_tunnel())
                await gateway.start()
                self.gateways.append(gateway)
            token = gateway.add(port)
        try:
            yield f"{gateway.url.rstrip('/')}/{token}"
        finally:
            del gateway.routes[token]
            if not gateway.routes:
                loop = asyncio.get_running_loop()
                gateway.idle = loop.call_later(
                    IDLE_SECONDS, lambda: loop.create_task(self.retire(gateway))
                )

    async def retire(self, gateway: HostGateway) -> None:
        async with self.lock:
            if gateway.routes or gateway not in self.gateways:
                return
            self.gateways.remove(gateway)
        await gateway.close()

    async def close(self) -> None:
        """Close every gateway now, routes held or not."""
        async with self.lock:
            gateways, self.gateways = self.gateways, []
        for gateway in gateways:
            with contextlib.suppress(Exception):
                await gateway.close()


_shared: dict[asyncio.AbstractEventLoop, SharedHostTunnel] = {}


def shared_host_tunnel() -> SharedHostTunnel:
    """This event loop's shared host tunnel, over Prime tunnels."""
    from verifiers.v1.interception.tunnel.prime import PrimeTunnel

    loop = asyncio.get_running_loop()
    for stale in [other for other in _shared if other.is_closed()]:
        del _shared[stale]
    if loop not in _shared:
        _shared[loop] = SharedHostTunnel(PrimeTunnel)
    return _shared[loop]
