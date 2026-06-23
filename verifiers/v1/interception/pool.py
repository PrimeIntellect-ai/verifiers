"""A pool of shared interception servers, grown on demand, so N concurrent rollouts need
~N/multiplex servers + tunnels rather than one each.

A harness runtime that cannot reach the host locally needs a tunnel for each interception
endpoint, and tunnel creation is rate-capped per API token — so one tunnel per rollout caps
fan-out. Each shared `InterceptionServer` serves up to `multiplex` rollouts behind one tunnel.
The pool is elastic: `acquire` reuses a server with a free slot, else brings up a new one — so it
fits both the bounded eval runner and the env server's unbounded request load. The harness is
unchanged: it authenticates with a per-rollout secret, which is what the server routes by.
"""

import asyncio
import logging
from contextlib import AsyncExitStack, asynccontextmanager
from dataclasses import dataclass

from verifiers.v1.interception.server import InterceptionServer, RolloutSession
from verifiers.v1.runtimes import (
    HOST,
    RuntimeConfig,
    reachable_url,
    runtime_reaches_host_locally,
)

logger = logging.getLogger(__name__)


@dataclass
class PooledServer:
    server: InterceptionServer
    base_url: str  # reachable interception base: `{base_url}/v1` (model), `/state` + `/task` (servers)
    load: int = 0


class InterceptionPool:
    """Shared interception servers. `multiplex` rollouts share one
    server (one tunnel behind a remote runtime); `acquire` hands a rollout a slot on one,
    bringing up a new server when all are at capacity."""

    def __init__(self, runtime_config: RuntimeConfig, multiplex: int) -> None:
        # The harness config decides reachability: provider sandboxes and interception-only Docker
        # need a host tunnel; ordinary local runtimes use localhost. The pool provisions neither.
        self.runtime_type = runtime_config.type
        self.host_is_local = runtime_reaches_host_locally(runtime_config)
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
        # The interception server is a HOST service the harness reaches through localhost or a
        # tunnel. The pool's stack owns both and tears them down together.
        url = await self._stack.enter_async_context(
            reachable_url(HOST, server.port, consumer_is_local=self.host_is_local)
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
