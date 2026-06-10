"""A pool of shared interception servers, so N rollouts need O(N/K) tunnels, not one each.

Behind a remote runtime each rollout's interception endpoint needs a tunnel, and tunnel
creation is rate-capped per API token — so one-tunnel-per-rollout caps how wide a remote
eval can fan out. This pool brings up `ceil(concurrency / multiplex)` shared
`InterceptionServer`s once for the eval and exposes each one (one tunnel per server, via a
host-side exposer runtime of the harness's runtime type), then hands each rollout a session
slot on one of them — a shared endpoint plus its own secret. The harness is unchanged: it
still authenticates with a per-rollout secret, which is exactly what the server routes by.
"""

import logging
import math
from contextlib import asynccontextmanager

from verifiers.v1.interception import InterceptionServer, RolloutSession
from verifiers.v1.runtimes import Runtime, RuntimeConfig, make_runtime

logger = logging.getLogger(__name__)


class InterceptionPool:
    """Shared interception servers for an eval. `multiplex` rollouts share one server (one
    tunnel behind a remote runtime); `acquire` hands a rollout a slot on one of them."""

    def __init__(
        self, runtime_config: RuntimeConfig, concurrency: int, multiplex: int
    ) -> None:
        self.runtime_config = runtime_config
        self.multiplex = max(1, multiplex)
        self.size = max(1, math.ceil(max(1, concurrency) / self.multiplex))
        self._servers: list[InterceptionServer] = []
        self._exposers: list[Runtime] = []
        self._endpoints: list[str] = []
        self._next = 0

    async def __aenter__(self) -> "InterceptionPool":
        for i in range(self.size):
            server = InterceptionServer()
            await server.__aenter__()
            # The exposer never runs a sandbox — it only turns the server's host port into
            # the URL the harness reaches (a tunnel for a remote runtime, localhost else);
            # its stop() in __aexit__ tears that tunnel down.
            exposer = make_runtime(self.runtime_config, name=f"interception-{i}")
            endpoint = f"{await exposer.expose(server.port)}/v1"
            self._servers.append(server)
            self._exposers.append(exposer)
            self._endpoints.append(endpoint)
        logger.info(
            "interception pool: %d server(s), multiplex=%d (%s)",
            self.size,
            self.multiplex,
            self.runtime_config.type,
        )
        return self

    async def __aexit__(self, *exc) -> None:
        for exposer in self._exposers:
            await exposer.stop()
        for server in self._servers:
            await server.__aexit__(*exc)

    @asynccontextmanager
    async def acquire(self, session: RolloutSession):
        """Register `session` on the next server (round-robin) and yield its
        `(endpoint, secret)`; deregister on exit."""
        idx = self._next % self.size
        self._next += 1
        server = self._servers[idx]
        secret = server.register(session)
        try:
            yield self._endpoints[idx], secret
        finally:
            server.unregister(secret)
