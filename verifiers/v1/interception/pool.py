"""Pools of shared interception servers, so N concurrent rollouts need ~N/multiplex
servers + tunnels rather than one each.

Behind a remote consumer each interception server needs a tunnel, and prime tunnel creation
is rate-capped per API token — so one-tunnel-per-rollout caps how wide a remote eval (or env
server) can fan out. Each shared `InterceptionServer` multiplexes rollouts behind one
tunnel; the harness is unchanged, authenticating with a per-rollout secret the server routes
by. Two shapes: `ElasticInterceptionPool` grows servers on demand (`multiplex` rollouts
each, always prime tunnels — the only kind the framework can mint) and fits both the bounded
eval runner and the env server's unbounded request load; `StaticInterceptionPool` is a fixed
set of servers (each with its own tunnel choice, e.g. bring-your-own endpoints), balanced
least-loaded.
"""

import asyncio
import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from verifiers.v1.interception.base import Interception, Slot
from verifiers.v1.interception.server import InterceptionServer, RolloutSession
from verifiers.v1.interception.tunnel import PrimeTunnel

logger = logging.getLogger(__name__)


class StaticInterceptionPool(Interception):
    """A fixed set of interception servers, all started up front; `acquire` hands a rollout
    a slot on the least-loaded one. No capacity cap — sizing the set to the load is the
    operator's call (it's the shape for pre-provisioned/bring-your-own endpoints)."""

    def __init__(self, servers: list[InterceptionServer]) -> None:
        super().__init__()
        if not servers:
            raise ValueError("a static interception pool needs at least one server")
        self.servers = servers

    async def start(self) -> None:
        for server in self.servers:
            await self._stack.enter_async_context(server)

    @asynccontextmanager
    async def acquire(self, session: RolloutSession) -> AsyncIterator[Slot]:
        # min + register have no await between them, so concurrent acquires can't all
        # land on the same "least-loaded" server.
        server = min(self.servers, key=lambda s: s.load)
        secret = server.register(session)
        try:
            yield server.base_url, secret
        finally:
            server.unregister(secret)


class ElasticInterceptionPool(Interception):
    """Interception servers grown on demand: `multiplex` rollouts share one server (one
    prime tunnel behind a remote consumer); `acquire` hands a rollout a slot on one,
    bringing up a new server when all are at capacity."""

    def __init__(self, *, multiplex: int = 32, is_local: bool = True) -> None:
        super().__init__()
        self.multiplex = max(1, multiplex)
        self.is_local = is_local
        self.servers: list[InterceptionServer] = []
        self._lock = asyncio.Lock()

    async def start(self) -> None:
        pass  # servers are brought up lazily in `acquire`, on `_stack`

    async def _server(self) -> InterceptionServer:
        """A server with spare capacity — reuse one under `multiplex`, else bring up a new
        one (its own tunnel, on `_stack`, torn down with the pool). The caller holds
        `_lock`."""
        for server in self.servers:
            if server.load < self.multiplex:
                return server
        server = InterceptionServer(None if self.is_local else PrimeTunnel())
        await self._stack.enter_async_context(server)
        self.servers.append(server)
        logger.info(
            "interception pool: %d server(s), multiplex=%d",
            len(self.servers),
            self.multiplex,
        )
        return server

    @asynccontextmanager
    async def acquire(self, session: RolloutSession) -> AsyncIterator[Slot]:
        # Register under the lock so concurrent acquires see each other's load.
        async with self._lock:
            server = await self._server()
            secret = server.register(session)
        try:
            yield server.base_url, secret
        finally:
            server.unregister(secret)
