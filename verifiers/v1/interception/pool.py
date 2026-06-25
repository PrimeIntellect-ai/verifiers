"""A pool of shared interception servers (prime tunnels), grown on demand, so N concurrent rollouts
need ~N/multiplex servers + tunnels rather than one each.

Behind a remote runtime each rollout's interception endpoint needs a prime tunnel, and tunnel
creation is rate-capped per API token — so one-tunnel-per-rollout caps how wide a remote eval (or
env server) can fan out. Each shared `InterceptionServer` serves up to `multiplex` rollouts behind
one tunnel. The pool is elastic: `acquire` reuses a server with a free slot, else brings up a new
one — so it fits both the bounded eval runner and the env server's unbounded request load. The
harness is unchanged: it authenticates with a per-rollout secret, which is what the server routes
by. The custom (BYO endpoint) type uses a single `InterceptionServer` directly — one server, no pool.
"""

import asyncio
import logging
from contextlib import asynccontextmanager
from dataclasses import dataclass

from verifiers.v1.interception.base import Interception
from verifiers.v1.interception.config import PrimeInterceptionConfig
from verifiers.v1.interception.server import InterceptionServer, RolloutSession
from verifiers.v1.interception.tunnel import PrimeTunnel

logger = logging.getLogger(__name__)


@dataclass
class PooledServer:
    server: InterceptionServer
    load: int = 0


class InterceptionPool(Interception):
    """Shared interception servers behind prime tunnels. `multiplex` rollouts share one server (one
    tunnel); `acquire` hands a rollout a slot on one, bringing up a new server when all are at
    capacity. `is_local` (no consumer is remote) means servers reach the harness at localhost
    without a tunnel; otherwise each server is exposed via its own prime tunnel."""

    def __init__(self, is_local: bool, config: PrimeInterceptionConfig) -> None:
        super().__init__()
        self.is_local = is_local
        self.multiplex = max(1, config.multiplex)
        self._servers: list[PooledServer] = []
        self._lock = asyncio.Lock()

    async def _entry(self) -> PooledServer:
        """A server with spare capacity — reuse one under `multiplex`, else bring up a new one (its
        own prime tunnel, on `_stack`, torn down with the pool). The caller holds `_lock`."""
        for entry in self._servers:
            if entry.load < self.multiplex:
                return entry
        server = InterceptionServer(PrimeTunnel(), self.is_local)
        await self._stack.enter_async_context(server)
        entry = PooledServer(server)
        self._servers.append(entry)
        logger.info(
            "interception pool: %d server(s), multiplex=%d",
            len(self._servers),
            self.multiplex,
        )
        return entry

    @asynccontextmanager
    async def acquire(self, session: RolloutSession):
        """Pick a server with spare capacity (bringing one up if needed) and hand `session` a slot
        on it (see `InterceptionServer.acquire`); free the slot on exit."""
        async with self._lock:
            entry = await self._entry()
            entry.load += 1
        try:
            async with entry.server.acquire(session) as slot:
                yield slot
        finally:
            entry.load -= 1
