"""Run-scoped serving resources shared by rollouts.

`RunServices` owns the interception pools — too expensive to create per rollout, but
not process-global: an in-process eval enters one for the command; a topology enters one
while its instances run. Shared MCP servers are NOT here: they are taskset-scoped
(`Taskset.tools`, served once by `TopologyRunner.serving` via `serve_shared`) and flow
into rollouts as a plain
`{name: SharedToolServer}` dict.
"""

import asyncio
import contextlib

from verifiers.v1.interception import InterceptionPool
from verifiers.v1.runtimes import RuntimeConfig, runtime_is_local


class RunServices:
    """The active serving scope: interception pools, keyed by the runtime placement a
    rollout actually uses. Interception only cares about reachability (local vs remote)
    and the runtime family for display, so a task-specific image/resource override does
    not fragment the pool."""

    def __init__(self, multiplex: int = 32) -> None:
        self.multiplex = multiplex
        self._entered = False
        self._stack = contextlib.AsyncExitStack()
        self._pools: dict[tuple[str, bool], InterceptionPool] = {}
        self._pool_locks: dict[tuple[str, bool], asyncio.Lock] = {}

    async def __aenter__(self) -> "RunServices":
        await self._stack.__aenter__()
        self._entered = True
        return self

    async def __aexit__(self, *exc) -> None:
        self._entered = False
        self._pools = {}
        self._pool_locks = {}
        await self._stack.__aexit__(*exc)

    async def pool_for(self, runtime_config: RuntimeConfig) -> InterceptionPool:
        """Return a pool compatible with where this rollout's harness actually runs."""
        if not self._entered:
            raise RuntimeError("RunServices must be entered before use")
        key = (runtime_config.type, runtime_is_local(runtime_config))
        pool = self._pools.get(key)
        if pool is not None:
            return pool
        lock = self._pool_locks.get(key)
        if lock is None:
            lock = asyncio.Lock()
            self._pool_locks[key] = lock
        async with lock:
            pool = self._pools.get(key)
            if pool is not None:
                return pool
            pool = await self._stack.enter_async_context(InterceptionPool(runtime_config, self.multiplex))
            self._pools[key] = pool
            return pool
