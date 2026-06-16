"""A pool of shared interception servers, grown on demand, so N concurrent rollouts need
~N/multiplex servers + tunnels rather than one each.

Behind a remote runtime each rollout's interception endpoint needs a tunnel, and tunnel
creation is rate-capped per API token — so one-tunnel-per-rollout caps how wide a remote
eval (or env server) can fan out. Each shared `InterceptionServer` serves up to `multiplex`
rollouts behind one tunnel (created via a host-side exposer runtime of the harness's runtime
type). The pool is elastic: `acquire` reuses a server with a free slot, else brings up a new
one — so it fits both the bounded eval runner and the env server's unbounded request load.
The harness is unchanged: it authenticates with a per-rollout secret, which is what the
server routes by.
"""

import asyncio
import logging
from contextlib import AsyncExitStack, asynccontextmanager
from dataclasses import dataclass

from verifiers.v1.interception.server import InterceptionServer, RolloutSession
from verifiers.v1.runtimes import RuntimeConfig, host_endpoint, runtime_is_local

logger = logging.getLogger(__name__)


@dataclass
class PooledServer:
    server: InterceptionServer
    endpoint: str
    load: int = 0


class InterceptionPool:
    """Shared interception servers. `multiplex` rollouts share one
    server (one tunnel behind a remote runtime); `acquire` hands a rollout a slot on one,
    bringing up a new server when all are at capacity."""

    def __init__(self, runtime_config: RuntimeConfig, multiplex: int) -> None:
        # The harness runtime's topology decides reachability: a remote one needs a host tunnel
        # to the interception port, a local one is reached at localhost. Read off the runtime
        # class (no provisioning) — the pool never runs a sandbox.
        self.runtime_type = runtime_config.type
        self.is_local = runtime_is_local(runtime_config)
        self.multiplex = max(1, multiplex)
        self._servers: list[PooledServer] = []
        self._lock = asyncio.Lock()
        self._stack = AsyncExitStack()

    async def __aenter__(self) -> "InterceptionPool":
        await self._stack.__aenter__()
        return self

    async def __aexit__(self, *exc) -> None:
        # The stack tears down every server + its host tunnel (LIFO), even if one teardown fails.
        await self._stack.aclose()

    async def _entry(self) -> PooledServer:
        """A server with spare capacity — reuse one under `multiplex`, else bring up a new one
        (its own host endpoint). The caller holds `_lock`."""
        for entry in self._servers:
            if entry.load < self.multiplex:
                return entry
        server = InterceptionServer()
        await self._stack.enter_async_context(server)
        # Turn the server's host port into the URL the harness reaches: a tunnel for a remote
        # runtime, localhost otherwise. Owned by the pool's stack, torn down with the server.
        url = await self._stack.enter_async_context(
            host_endpoint(server.port, self.is_local)
        )
        entry = PooledServer(server, f"{url}/v1")
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
        """Register `session` on a server with spare capacity (bringing one up if needed)
        and yield its `(endpoint, secret)`; free the slot on exit."""
        async with self._lock:
            entry = await self._entry()
            secret = entry.server.register(session)
            entry.load += 1
        try:
            yield entry.endpoint, secret
        finally:
            entry.server.unregister(secret)
            entry.load -= 1
