"""Asyncio helpers shared across the framework."""

import asyncio
from collections.abc import Awaitable
from typing import TypeVar

T = TypeVar("T")


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


def cancel_requested() -> bool:
    """Whether the current task holds a cancellation no handler has answered: a library
    may swallow one (connectrpc turns a CancelledError inside a request into a
    ConnectError CANCELED, which arrives as a `SandboxError`) and the task would
    otherwise retry, reopen or replace a box while the run is ending."""
    task = asyncio.current_task()
    return task is not None and task.cancelling() > 0


def check_cancelled() -> None:
    """Raise the cancellation the current task holds, if any (`cancel_requested`)."""
    if cancel_requested():
        raise asyncio.CancelledError()
