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
from typing import Literal

from pydantic import Field

from verifiers.v1.interception.base import BaseInterceptionConfig, Interception, Slot
from verifiers.v1.interception.server import (
    InterceptionServer,
    InterceptionServerConfig,
)
from verifiers.v1.interception.tunnel import PrimeTunnelConfig
from verifiers.v1.session import RolloutSession

logger = logging.getLogger(__name__)


class StaticInterceptionPoolConfig(BaseInterceptionConfig):
    """A fixed set of interception servers, each configured like a `server` type; rollouts
    land on the least-loaded one. The shape for multiple bring-your-own endpoints (one
    `custom` tunnel per server)."""

    type: Literal["static"] = "static"
    servers: list[InterceptionServerConfig] = Field(min_length=1)
    """One entry per server, each with its own `tunnel` choice."""


class StaticInterceptionPool(Interception):
    """A fixed set of interception servers, all started up front; `acquire` hands a rollout
    a slot on the least-loaded one. No capacity cap — sizing the set to the load is the
    operator's call (it's the shape for pre-provisioned/bring-your-own endpoints)."""

    def __init__(
        self, config: StaticInterceptionPoolConfig, requires_tunnel: bool = False
    ) -> None:
        super().__init__()
        self.config = config
        self.servers = [
            InterceptionServer(server, requires_tunnel) for server in config.servers
        ]

    async def start(self) -> None:
        for server in self.servers:
            await self.stack.enter_async_context(server)

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


class ElasticInterceptionPoolConfig(BaseInterceptionConfig):
    """Interception servers grown on demand: `multiplex` rollouts share one server (and,
    behind a remote consumer, one prime tunnel). The default."""

    type: Literal["elastic"] = "elastic"
    multiplex: int = Field(32, ge=1)
    """Rollouts that share one interception server (and tunnel). N concurrent rollouts use
    ~N/multiplex servers + tunnels instead of one each — key past the per-token tunnel cap.
    1 = a server (+ tunnel) per rollout."""


class ElasticInterceptionPool(Interception):
    """Interception servers grown on demand: `multiplex` rollouts share one server (one
    prime tunnel behind a remote consumer); `acquire` hands a rollout a slot on one,
    bringing up a new server when all are at capacity."""

    def __init__(
        self,
        config: ElasticInterceptionPoolConfig | None = None,
        requires_tunnel: bool = False,
    ) -> None:
        super().__init__()
        self.config = config or ElasticInterceptionPoolConfig()
        self.requires_tunnel = requires_tunnel
        self.servers: list[InterceptionServer] = []
        self._lock = asyncio.Lock()

    async def start(self) -> None:
        pass  # servers are brought up lazily in `acquire`, on `stack`

    async def _server(self) -> InterceptionServer:
        """A server with spare capacity — reuse one under `multiplex`, else bring up a new
        one (its own tunnel, on `stack`, torn down with the pool). The caller holds
        `_lock`."""
        for server in self.servers:
            if server.load < self.config.multiplex:
                return server
        # Pin prime explicitly — the only tunnel kind that can be minted on demand.
        server = InterceptionServer(
            InterceptionServerConfig(tunnel=PrimeTunnelConfig()), self.requires_tunnel
        )
        await self.stack.enter_async_context(server)
        self.servers.append(server)
        logger.info(
            "interception pool: %d server(s), multiplex=%d",
            len(self.servers),
            self.config.multiplex,
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
