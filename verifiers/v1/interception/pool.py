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

from verifiers.v1.configs.interception import (
    ElasticInterceptionPoolConfig,
    InterceptionServerConfig,
    PrimeTunnelConfig,
    StaticInterceptionPoolConfig,
)
from verifiers.v1.interception.base import Interception, Slot
from verifiers.v1.interception.server import InterceptionServer
from verifiers.v1.session import RolloutSession

logger = logging.getLogger(__name__)


class StaticInterceptionPool(Interception):
    """A fixed set of interception servers, all started up front; `acquire` hands a rollout
    a slot on the least-loaded one. No capacity cap — sizing the set to the load is the
    operator's call (it's the shape for pre-provisioned/bring-your-own endpoints)."""

    def __init__(
        self,
        config: StaticInterceptionPoolConfig,
        requires_tunnel: bool = False,
        state_service_secrets: tuple[str, ...] = (),
    ) -> None:
        super().__init__()
        self.config = config
        self.servers = [
            InterceptionServer(server, requires_tunnel, state_service_secrets)
            for server in config.servers
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


class ElasticInterceptionPool(Interception):
    """Warm the first interception server on start, then grow on demand: `multiplex`
    rollouts share one server (one prime tunnel behind a remote consumer); `acquire` hands
    a rollout a slot on one, bringing up a new server when all are at capacity."""

    def __init__(
        self,
        config: ElasticInterceptionPoolConfig | None = None,
        requires_tunnel: bool = False,
        state_service_secrets: tuple[str, ...] = (),
    ) -> None:
        super().__init__()
        self.config = config or ElasticInterceptionPoolConfig()
        self.requires_tunnel = requires_tunnel
        self.state_service_secrets = state_service_secrets
        self.servers: list[InterceptionServer] = []
        self._lock = asyncio.Lock()
        self._warm_task: asyncio.Task[InterceptionServer] | None = None

    async def start(self) -> None:
        self._warm_task = asyncio.create_task(self._server())

    async def stop(self) -> None:
        if self._warm_task is not None:
            self._warm_task.cancel()
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await self._warm_task
        await super().stop()

    async def _server(self) -> InterceptionServer:
        """A server with spare capacity — reuse one under `multiplex`, else bring up a new
        one (its own tunnel, on `stack`, torn down with the pool). Acquires hold `_lock`;
        the warm task runs before they reach this path."""
        for server in self.servers:
            if server.load < self.config.multiplex:
                return server
        # Pin prime explicitly — the only tunnel kind that can be minted on demand.
        server = InterceptionServer(
            InterceptionServerConfig(tunnel=PrimeTunnelConfig()),
            self.requires_tunnel,
            self.state_service_secrets,
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
            model_secret, state_secret = server.register(session)
        try:
            yield server.base_url, model_secret, state_secret
        finally:
            server.unregister(model_secret, state_secret)
