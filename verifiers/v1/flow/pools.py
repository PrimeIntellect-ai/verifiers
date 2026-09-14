"""Named capacity pools: how many node instances may run at once per resource."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Iterable
from contextlib import AsyncExitStack, asynccontextmanager


class PoolFull(Exception):
    pass


class Pool:
    def __init__(self, max_running: int, max_queued: int | None = None) -> None:
        self._sem = asyncio.Semaphore(max_running)
        self.max_running = max_running
        self.max_queued = max_queued
        self.running = 0
        self.waiting = 0

    @asynccontextmanager
    async def hold(self) -> AsyncIterator[None]:
        if self.max_queued is not None and self.waiting >= self.max_queued:
            raise PoolFull(f"{self.waiting} already waiting")
        self.waiting += 1
        try:
            await self._sem.acquire()
        finally:
            self.waiting -= 1
        self.running += 1
        try:
            yield
        finally:
            self.running -= 1
            self._sem.release()


class Pools:
    def __init__(self, sizes: dict[str, int]) -> None:
        self._pools = {name: Pool(size) for name, size in sizes.items()}

    @asynccontextmanager
    async def hold(self, names: Iterable[str]) -> AsyncIterator[None]:
        """Hold every named pool for the duration; unknown names are unbounded."""
        async with AsyncExitStack() as stack:
            for name in sorted(set(names)):
                if pool := self._pools.get(name):
                    await stack.enter_async_context(pool.hold())
            yield

    def gauges(self) -> dict[str, dict[str, int]]:
        return {
            name: {"running": p.running, "waiting": p.waiting, "max": p.max_running}
            for name, p in self._pools.items()
        }
