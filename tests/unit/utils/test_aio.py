import asyncio
from contextlib import asynccontextmanager

import pytest

from verifiers.v1.utils.aio import background


def resource(events: list, *, start: float = 0.0, fail: bool = False):
    @asynccontextmanager
    async def cm():
        events.append("starting")
        try:
            await asyncio.sleep(start)
            if fail:
                raise RuntimeError("start failed")
            events.append("started")
            yield "value"
        finally:
            events.append("stopped")

    return cm()


async def test_the_caller_works_while_a_background_resource_starts():
    events = []
    async with background(resource(events, start=0.05)) as entered:
        events.append("caller works")
        assert await entered == "value"
        events.append("caller done")
    assert events == ["caller works", "starting", "started", "caller done", "stopped"]


async def test_leaving_cancels_a_start_in_progress():
    events = []
    async with background(resource(events, start=60)) as entered:
        await asyncio.sleep(0)
    assert entered.cancelled()
    assert events == ["starting", "stopped"]


async def test_leaving_before_the_start_ever_ran():
    events = []
    async with background(resource(events)) as entered:
        pass
    assert entered.cancelled()


async def test_a_failed_start_raises_where_it_is_awaited():
    events = []
    async with background(resource(events, fail=True)) as entered:
        with pytest.raises(RuntimeError, match="start failed"):
            await entered
    # One the caller never awaits is still handled when it leaves.
    async with background(resource([], fail=True)):
        await asyncio.sleep(0.01)
