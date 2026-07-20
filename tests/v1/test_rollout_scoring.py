import asyncio

import pytest

from verifiers.v1.rollout import gather_scoring


async def test_empty_and_singleton():
    assert await gather_scoring([]) == []
    assert await gather_scoring([asyncio.sleep(0, result=3)]) == [3]


async def test_result_order_is_preserved():
    async def value(value, delay):
        await asyncio.sleep(delay)
        return value

    assert await gather_scoring([value("first", 0.02), value("second", 0)]) == [
        "first",
        "second",
    ]


async def test_failure_cancels_and_drains_sibling():
    stopped = asyncio.Event()

    async def sibling():
        try:
            await asyncio.sleep(60)
        finally:
            stopped.set()

    async def failure():
        await asyncio.sleep(0)
        raise ValueError("handler failed")

    with pytest.raises(ValueError, match="handler failed"):
        await gather_scoring([failure(), sibling()])
    assert stopped.is_set()


async def test_timeout_cancels_and_drains_siblings():
    async def slow():
        await asyncio.sleep(60)

    with pytest.raises(asyncio.TimeoutError):
        await asyncio.wait_for(gather_scoring([slow(), asyncio.sleep(60)]), 0.01)


async def test_simultaneous_failures_original_wins():
    async def first():
        raise ValueError("first")

    async def second():
        raise RuntimeError("second")

    with pytest.raises(ValueError, match="first"):
        await gather_scoring([first(), second()])


async def test_cleanup_failure_does_not_replace_original():
    async def failure():
        raise ValueError("original")

    async def cleanup_failure():
        try:
            await asyncio.sleep(60)
        finally:
            raise RuntimeError("cleanup")

    with pytest.raises(ValueError, match="original"):
        await gather_scoring([failure(), cleanup_failure()])


async def test_repeated_cancellation_during_cleanup():
    started = asyncio.Event()

    async def sibling():
        try:
            started.set()
            await asyncio.sleep(60)
        finally:
            await asyncio.sleep(0)

    async def failure():
        await started.wait()
        raise ValueError("original")

    task = asyncio.create_task(gather_scoring([failure(), sibling()]))
    await started.wait()
    task.cancel()
    await asyncio.sleep(0)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task


async def test_no_tasks_leaked_after_failure():
    async def failure():
        raise ValueError("boom")

    with pytest.raises(ValueError):
        await gather_scoring([failure(), asyncio.sleep(60)])
    await asyncio.sleep(0)
    assert not [
        task for task in asyncio.all_tasks() if task is not asyncio.current_task()
    ]
