import asyncio

import pytest

from verifiers.v1.utils.aio import run_shielded


async def test_finishes_cleanup_then_reraises_cancellation():
    done = []

    async def cleanup():
        await asyncio.sleep(0.05)
        done.append(True)

    task = asyncio.ensure_future(run_shielded(cleanup()))
    await asyncio.sleep(0.01)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert done == [True]


async def test_hung_cleanup_is_given_up_after_its_grace():
    hung = asyncio.Event()

    async def cleanup():
        try:
            await asyncio.Event().wait()  # never answers
        except asyncio.CancelledError:
            hung.set()
            raise

    task = asyncio.ensure_future(run_shielded(cleanup(), grace=0.05))
    await asyncio.sleep(0.01)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await asyncio.wait_for(task, 2)
    assert hung.is_set()


async def test_cleanup_that_ignores_its_cancellation_is_left_behind():
    release = asyncio.Event()

    async def cleanup():
        while not release.is_set():
            try:
                await release.wait()
            except asyncio.CancelledError:
                pass  # swallows every cancellation until the test releases it

    task = asyncio.ensure_future(run_shielded(cleanup(), grace=0.05))
    await asyncio.sleep(0.01)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await asyncio.wait_for(task, 2)
    release.set()  # lets the left-behind coroutine end before the loop closes
    await asyncio.sleep(0)


async def test_second_cancellation_still_waits_for_the_unwind():
    unwound = []

    async def cleanup():
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            await asyncio.sleep(0.03)  # its own cleanup after the grace cancels it
            unwound.append(True)
            raise

    task = asyncio.ensure_future(run_shielded(cleanup(), grace=0.05))
    await asyncio.sleep(0.01)
    task.cancel()
    await asyncio.sleep(0.06)  # past the grace, while the coroutine unwinds
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await asyncio.wait_for(task, 2)
    assert unwound == [True]


async def test_uncancelled_cleanup_is_unbounded_and_returns():
    async def cleanup():
        await asyncio.sleep(0.1)
        return 7

    assert await run_shielded(cleanup(), grace=0.01) == 7


async def test_error_without_cancellation_propagates():
    async def cleanup():
        raise ValueError("boom")

    with pytest.raises(ValueError):
        await run_shielded(cleanup())
