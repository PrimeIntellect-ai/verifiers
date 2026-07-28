"""Pools of shared interception servers, so N concurrent rollouts need ~N/multiplex
servers + tunnels rather than one each.

Behind a remote consumer each interception server needs a tunnel, and prime tunnel creation
is rate-capped per API token — so one-tunnel-per-rollout caps how wide a remote eval (or env
server) can fan out. Each shared `InterceptionServer` multiplexes rollouts behind one
tunnel; the harness is unchanged, authenticating with a per-rollout secret the server routes
by. Two shapes: `ElasticInterceptionPool` warms one server, then grows on demand (`multiplex`
rollouts each, always prime tunnels — the only kind the framework can mint) and fits both the
bounded eval runner and the env server's unbounded request load; `StaticInterceptionPool` is a
fixed set of servers (each with its own tunnel choice, e.g. bring-your-own endpoints), balanced
least-loaded.
"""

import asyncio
import contextlib
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

# How often the elastic pool polls its servers' tunnels. A dead tunnel keeps absorbing
# rollout assignments until the next poll (fast-failing ones drain and re-land on it), so
# the interval bounds how long that black hole lives; per-server checks are one GET to the
# tunnel service, so polling even a large pool this often is cheap.
_HEALTH_CHECK_INTERVAL = 60.0


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
        # server.acquire registers before its first yield, so concurrent acquires see the
        # updated load before choosing their own least-loaded server.
        server = min(self.servers, key=lambda s: s.load)
        async with server.acquire(session) as slot:
            yield slot


class ElasticInterceptionPoolConfig(BaseInterceptionConfig):
    """An eagerly warmed interception server, then more grown on demand: `multiplex`
    rollouts share one server (and, behind a remote consumer, one prime tunnel). The default."""

    type: Literal["elastic"] = "elastic"
    multiplex: int = Field(32, ge=1)
    """Rollouts that share one interception server (and tunnel). N concurrent rollouts use
    ~N/multiplex servers + tunnels instead of one each — key past the per-token tunnel cap.
    1 = a server (+ tunnel) per rollout."""


class ElasticInterceptionPool(Interception):
    """Warm the first interception server on start, then grow on demand: `multiplex`
    rollouts share one server (one prime tunnel behind a remote consumer); `acquire` hands
    a rollout a slot on one, bringing up a new server when all are at capacity. A health
    loop retires servers whose tunnel has died (see `Tunnel.is_alive`), so the pool grows
    back with fresh tunnels instead of black-holing rollouts on dead URLs."""

    def __init__(
        self,
        config: ElasticInterceptionPoolConfig | None = None,
        requires_tunnel: bool = False,
    ) -> None:
        super().__init__()
        self.config = config or ElasticInterceptionPoolConfig()
        self.requires_tunnel = requires_tunnel
        self.servers: list[InterceptionServer] = []
        self._draining: list[InterceptionServer] = []
        self._lock = asyncio.Lock()
        self._warm_task: asyncio.Task[InterceptionServer] | None = None
        self._health_task: asyncio.Task[None] | None = None

    async def start(self) -> None:
        self._warm_task = asyncio.create_task(self._server())
        if self.requires_tunnel:
            self._health_task = asyncio.create_task(self._health_loop())

    async def stop(self) -> None:
        for task in (self._warm_task, self._health_task):
            if task is not None:
                task.cancel()
                with contextlib.suppress(asyncio.CancelledError, Exception):
                    await task
        self._warm_task = None
        self._health_task = None
        await super().stop()

    async def _health_loop(self) -> None:
        """Retire servers whose tunnel has died, so rollouts stop landing on dead URLs.

        A dead server otherwise poisons the pool forever: its rollouts fail instantly, its
        load drops back to 0, and `_server` hands it out again — while healthy servers work
        real (slow) rollouts, so the dead one absorbs most assignments. Retiring also covers
        tunnels in a terminal status that still route on a surviving frpc connection: they
        die for good on the next connection drop, so they're replaced proactively."""
        while True:
            await asyncio.sleep(_HEALTH_CHECK_INTERVAL)
            try:
                await self._retire_dead_servers()
            except Exception:
                logger.exception("interception pool: health check failed")

    async def _retire_dead_servers(self) -> None:
        # Liveness checks call the tunnel service — keep them off the acquire lock.
        for server in list(self.servers):
            if server.tunnel is None or await server.tunnel.is_alive():
                continue
            async with self._lock:
                if server in self.servers:
                    self.servers.remove(server)
                    self._draining.append(server)
                    logger.warning(
                        "interception pool: retired server with dead tunnel %s (%d rollouts draining)",
                        server.base_url,
                        server.load,
                    )
        # Tear a retired server down only once its in-flight rollouts have drained; until
        # then its interception server keeps serving whatever still reaches it. The pool's
        # exit stack still holds every retired server, so this early stop is an optimization
        # (frees the frpc process and registration) and the double-stop at pool shutdown is
        # a no-op.
        for server in list(self._draining):
            if server.load > 0:
                continue
            self._draining.remove(server)
            with contextlib.suppress(Exception):
                await server.stop()

    async def _server(self) -> InterceptionServer:
        """A server with spare capacity — reuse one under `multiplex`, else bring up a new
        one (its own tunnel, on `stack`, torn down with the pool). Acquires hold `_lock`;
        the warm task runs before they reach this path."""
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
        if self._warm_task is not None:
            with contextlib.suppress(Exception):
                await asyncio.shield(self._warm_task)
            self._warm_task = None
        # Register under the lock so concurrent acquires see each other's load.
        async with self._lock:
            server = await self._server()
            secret = server.register(session)
        try:
            yield server.base_url, secret
        finally:
            server.unregister(secret)
