"""A pool of shared interception servers, grown on demand.

Each `InterceptionServer` serves up to `multiplex` rollouts. Remote runtimes share its host tunnel;
offline Docker agents share its Docker-gateway listener. The pool grows when every server is full.
"""

import asyncio
import logging
from contextlib import AsyncExitStack, asynccontextmanager
from dataclasses import dataclass

from verifiers.v1.interception.server import InterceptionServer, RolloutSession
from verifiers.v1.runtimes import (
    HOST,
    DockerConfig,
    RuntimeConfig,
    reachable_url,
    runtime_is_local,
)
from verifiers.v1.runtimes.docker import docker_bridge_gateway

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
        self.runtime_type = runtime_config.type
        self.is_local = runtime_is_local(runtime_config)
        self.offline_docker = (
            isinstance(runtime_config, DockerConfig)
            and not runtime_config.network_access
        )
        self.interception_host: str | None = None
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
        if self.offline_docker and self.interception_host is None:
            self.interception_host = await docker_bridge_gateway()
        server = InterceptionServer(host=self.interception_host)
        await self._stack.enter_async_context(server)
        # Local harnesses start with loopback (offline Docker rewrites it at seal); provider
        # sandboxes share a host tunnel. The pool's stack owns both endpoints.
        url = await self._stack.enter_async_context(
            reachable_url(HOST, server.port, consumer_is_local=self.is_local)
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
