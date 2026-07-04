"""Asyncio helpers shared across the framework."""

import asyncio
from collections.abc import Awaitable, Callable
from typing import TypeVar

T = TypeVar("T")


async def run_shielded(
    coro: Awaitable[T], interrupted: Callable[[], None] | None = None
) -> T:
    """Run `coro` to completion even if the surrounding task is cancelled, then re-raise
    the cancellation. A bare `asyncio.shield` is not enough: on cancellation it re-raises
    immediately while the inner task runs on orphaned — and the loop's shutdown then
    cancels that orphan mid-await anyway. This keeps awaiting (absorbing repeated task
    cancellations) until `coro` finishes; the first CancelledError is re-raised after.
    `interrupted`, if given, is invoked once on the first absorbed cancellation — so the
    caller can tell the user the wait is intentional, not a hang.
    If `coro` raises without a cancellation, the error propagates unchanged; with one,
    the cancellation wins and the error is chained under it (`from`), so it is never
    silently lost — except a `BaseException` (KeyboardInterrupt, SystemExit), which
    propagates immediately: a user abort must never lose to a pending cancellation.
    A second Ctrl-C that raises KeyboardInterrupt out of the event loop itself is beyond
    any task-level shield — that path is the atexit backstop's job."""
    task = asyncio.ensure_future(coro)
    cancelled: asyncio.CancelledError | None = None
    while not task.done():
        try:
            await asyncio.shield(task)
        except asyncio.CancelledError as e:
            if cancelled is None and interrupted is not None:
                interrupted()
            cancelled = e
        except Exception:
            pass  # `coro` itself failed → task is done; re-raised (or chained) below
    if cancelled is not None:
        raise cancelled from (None if task.cancelled() else task.exception())
    return task.result()
