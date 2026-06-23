"""A pool of shared interception servers (prime tunnels), grown on demand, so N concurrent rollouts
need ~N/multiplex servers + tunnels rather than one each.

Behind a remote runtime each rollout's interception endpoint needs a prime tunnel, and tunnel
creation is rate-capped per API token — so one-tunnel-per-rollout caps how wide a remote eval (or
env server) can fan out. Each shared `InterceptionServer` serves up to `multiplex` rollouts behind
one tunnel. The pool is elastic: `acquire` reuses a server with a free slot, else brings up a new
one — so it fits both the bounded eval runner and the env server's unbounded request load. The
harness is unchanged: it authenticates with a per-rollout secret, which is what the server routes
by. The custom (BYO endpoint) type uses `SingleInterception` instead — one server, no pool.
"""

import asyncio
import logging
from contextlib import asynccontextmanager
from dataclasses import dataclass

from verifiers.v1.interception.base import Interception
from verifiers.v1.interception.config import PrimeInterceptionConfig
from verifiers.v1.interception.server import InterceptionServer, RolloutSession
from verifiers.v1.interception.tunnel import PrimeTunnel
from verifiers.v1.runtimes import RuntimeConfig, runtime_is_local

logger = logging.getLogger(__name__)


@dataclass
class PooledServer:
    server: InterceptionServer
    base_url: str  # reachable interception base: `{base_url}/v1` (model), `/state` + `/task` (servers)
    load: int = 0


class InterceptionPool(Interception):
    """Shared interception servers behind prime tunnels. `multiplex` rollouts share one server (one
    tunnel); `acquire` hands a rollout a slot on one, bringing up a new server when all are at
    capacity."""

    def __init__(
        self, runtime_config: RuntimeConfig, config: PrimeInterceptionConfig
    ) -> None:
        super().__init__()
        # The harness runtime's topology decides reachability: a remote one needs a host tunnel
        # to the interception port, a local one is reached at localhost. Read off the runtime
        # class (no provisioning) — the pool never runs a sandbox.
        self.runtime_type = runtime_config.type
        self.is_local = runtime_is_local(runtime_config)
        self.multiplex = max(1, config.multiplex)
        self._servers: list[PooledServer] = []
        self._lock = asyncio.Lock()

    async def _entry(self) -> PooledServer:
        """A server with spare capacity — reuse one under `multiplex`, else bring up a new one (its
        own prime tunnel). The caller holds `_lock`."""
        for entry in self._servers:
            if entry.load < self.multiplex:
                return entry
        # Each server owns its own prime tunnel; both are on `_stack`, torn down with it (LIFO).
        server = InterceptionServer(PrimeTunnel())
        await self._stack.enter_async_context(server)
        url = await self._stack.enter_async_context(
            server.reachable(is_local=self.is_local)
        )
        entry = PooledServer(server, url)
        self._servers.append(entry)
        logger.info(
            "interception pool: %d server(s), multiplex=%d (%s)",
            len(self._servers),
            self.multiplex,
            self.runtime_type,
        )
        return entry

    @asynccontextmanager
    async def acquire(self, session: RolloutSession):
        """Register `session` on a server with spare capacity (bringing one up if needed) and yield
        its `(endpoint, secret, port, base_url)` — `endpoint` is the model route (`{base_url}/v1`),
        `port` the interception server's host port (a per-rollout tool server's own channel), and
        `base_url` its reachable URL (how a `shared` server reaches this rollout's `/state` + `/task`);
        free the slot on exit."""
        async with self._lock:
            entry = await self._entry()
            secret = entry.server.register(session)
            entry.load += 1
        try:
            yield f"{entry.base_url}/v1", secret, entry.server.port, entry.base_url
        finally:
            entry.server.unregister(secret)
            entry.load -= 1
