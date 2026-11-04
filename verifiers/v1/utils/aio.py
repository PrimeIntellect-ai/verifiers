"""Asyncio helpers shared across the framework."""

import asyncio
from collections.abc import AsyncIterator, Awaitable
from contextlib import AbstractAsyncContextManager, asynccontextmanager
from typing import TypeVar

T = TypeVar("T")


@asynccontextmanager
async def background(
    cm: AbstractAsyncContextManager[T],
) -> AsyncIterator[asyncio.Future[T]]:
    """Enter `cm` in a task of its own and yield a future of its value at once, so the
    caller keeps working while it starts. Leaving exits `cm` in that same task, after
    cancelling an entry still in progress; an error on the way out propagates."""
    entered: asyncio.Future[T] = asyncio.get_running_loop().create_future()
    leave = asyncio.Event()

    async def own() -> None:
        try:
            async with cm as value:
                entered.set_result(value)
                await leave.wait()
        except asyncio.CancelledError:
            if entered.done():
                raise
            entered.cancel()
        except Exception as error:
            if entered.done():
                raise
            entered.set_exception(error)

    task = asyncio.create_task(own())
    try:
        yield entered
    finally:
        leave.set()
        if not entered.done():
            task.cancel()
        await run_shielded(asyncio.wait([task]))
        entered.cancel()  # a task cancelled before it ever ran left it unset
        if not entered.cancelled():
            entered.exception()  # a failed entry the caller never awaited is seen here
        if not task.cancelled() and (error := task.exception()) is not None:
            raise error


async def run_shielded(coro: Awaitable[T]) -> T:
    """Run `coro` to completion even if the surrounding task is cancelled, then re-raise
    the cancellation. A bare `asyncio.shield` is not enough: on cancellation it re-raises
    immediately while the inner task runs on orphaned — and the loop's shutdown then
    cancels that orphan mid-await anyway. This keeps awaiting (absorbing repeated task
    cancellations) until `coro` finishes; the first CancelledError is re-raised after.
    If `coro` raises without a cancellation, the error propagates unchanged; with one,
    the cancellation wins and the error is chained under it (`from`), so it is never
    silently lost. A second Ctrl-C that raises KeyboardInterrupt out of the event loop
    itself is beyond any task-level shield — that path is the atexit backstop's job."""
    task = asyncio.ensure_future(coro)
    cancelled: asyncio.CancelledError | None = None
    while not task.done():
        try:
            await asyncio.shield(task)
        except asyncio.CancelledError as e:
            cancelled = e
        except BaseException:  # noqa: BLE001, S110 - re-raised or chained below
            pass  # `coro` itself failed → task is done; re-raised (or chained) below
    if cancelled is not None:
        raise cancelled from (None if task.cancelled() else task.exception())
    return task.result()
