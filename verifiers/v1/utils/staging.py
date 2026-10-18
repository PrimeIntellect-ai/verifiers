"""Bound host memory while uploading files through whole-file runtime writes."""

import asyncio
import hashlib
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path

from verifiers.v1.runtimes import Runtime

STAGING_BYTES = 256 * 1024 * 1024


class _StagingBudget:
    def __init__(self) -> None:
        self.loop: asyncio.AbstractEventLoop | None = None
        self.condition = asyncio.Condition()
        self.used = 0

    @asynccontextmanager
    async def hold(self, size: int) -> AsyncIterator[None]:
        loop = asyncio.get_running_loop()
        if self.loop is not loop:
            self.loop, self.condition, self.used = loop, asyncio.Condition(), 0
        condition = self.condition
        async with condition:
            # An oversized file is allowed, but only while no other upload holds bytes.
            await condition.wait_for(
                lambda: not self.used or self.used + size <= STAGING_BYTES
            )
            self.used += size
        try:
            yield
        finally:
            async with condition:
                self.used -= size
                condition.notify_all()


_budget = _StagingBudget()


async def stage_file(runtime: Runtime, target: str, source: Path) -> str:
    """Upload off-loop-read bytes within the shared budget; return their SHA256."""
    async with _budget.hold(source.stat().st_size):
        data = await asyncio.to_thread(source.read_bytes)
        await runtime.write(target, data)
        return hashlib.sha256(data).hexdigest()
