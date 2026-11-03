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
