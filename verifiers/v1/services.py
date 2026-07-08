"""Run-scoped serving resources shared by rollouts.

`RunServices` owns the things that are too expensive or stateful to create per
rollout, but still must not be process-global: shared MCP servers and interception
pools. An in-process eval enters one for the command; an env-server worker enters one
for its lifetime; a topology enters one while its instances run.
"""

import contextlib

from verifiers.v1.interception import InterceptionPool
from verifiers.v1.mcp import SharedServers
from verifiers.v1.runtimes import RuntimeConfig, runtime_is_local


class RunServices:
    """The active serving scope: shared MCP servers plus interception pools.

    Pools are keyed by the runtime placement a rollout actually uses. Interception only
    cares about reachability (local vs remote) and the runtime family for display, so a
    task-specific image/resource override should not fragment the pool.
    """

    def __init__(self, multiplex: int = 32) -> None:
        self.multiplex = multiplex
        self.shared: SharedServers | None = None
        self._stack = contextlib.AsyncExitStack()
        self._pools: dict[tuple[str, bool], InterceptionPool] = {}

    async def __aenter__(self) -> "RunServices":
        await self._stack.__aenter__()
        self.shared = await self._stack.enter_async_context(SharedServers())
        return self

    async def __aexit__(self, *exc) -> None:
        self.shared = None
        self._pools = {}
        await self._stack.__aexit__(*exc)

    async def pool_for(self, runtime_config: RuntimeConfig) -> InterceptionPool:
        """Return a pool compatible with where this rollout's harness actually runs."""
        if self.shared is None:
            raise RuntimeError("RunServices must be entered before use")
        key = (runtime_config.type, runtime_is_local(runtime_config))
        pool = self._pools.get(key)
        if pool is None:
            pool = await self._stack.enter_async_context(
                InterceptionPool(runtime_config, self.multiplex)
            )
            self._pools[key] = pool
        return pool
