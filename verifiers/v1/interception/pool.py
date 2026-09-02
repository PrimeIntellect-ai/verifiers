"""Pools of shared interception servers, so N concurrent rollouts need ~N/multiplex
servers + tunnels rather than one each.

Behind a remote consumer each interception server needs a tunnel, and prime tunnel creation
is rate-capped per API token — so one-tunnel-per-rollout caps how wide a remote eval (or env
server) can fan out. Each shared `InterceptionServer` multiplexes rollouts behind one
tunnel; the harness is unchanged, authenticating with a per-rollout secret the server routes
by. `ElasticInterceptionPool` warms one server, then grows on demand (`multiplex` rollouts
each, behind prime tunnels) and fits both the bounded eval runner and the env server's
unbounded request load.
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
from verifiers.v1.session import RolloutSession

logger = logging.getLogger(__name__)


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
        server = InterceptionServer(
            InterceptionServerConfig(),
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
