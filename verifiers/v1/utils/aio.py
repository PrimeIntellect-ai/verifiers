"""Asyncio helpers shared across the framework."""

import asyncio
from collections.abc import Awaitable
from typing import TypeVar

T = TypeVar("T")


GRACE_S = 600.0
"""How long a shielded coroutine may still run once its caller is cancelled: cleanup that
hangs (a remote call that never answers) must not keep a cancelled run, or the process
stopping it, waiting forever."""


async def run_shielded(coro: Awaitable[T], *, grace: float | None = GRACE_S) -> T:
    """Run `coro` to completion even if the surrounding task is cancelled, then re-raise
    the cancellation. A bare `asyncio.shield` is not enough: on cancellation it re-raises
    immediately while the inner task runs on orphaned — and the loop's shutdown then
    cancels that orphan mid-await anyway. This keeps awaiting (absorbing repeated task
    cancellations) until `coro` finishes; the first CancelledError is re-raised after.
    If `coro` raises without a cancellation, the error propagates unchanged; with one,
    the cancellation wins and the error is chained under it (`from`), so it is never
    silently lost. Once cancelled, `coro` has `grace` seconds more (None: unbounded);
    past them it is cancelled too, and the cancellation re-raised. Uncancelled, it runs
    as long as it takes. A second Ctrl-C that raises KeyboardInterrupt out of the event
    loop itself is beyond any task-level shield — that path is the atexit backstop's job."""
    task = asyncio.ensure_future(coro)
    cancelled: asyncio.CancelledError | None = None
    deadline: float | None = None
    loop = asyncio.get_running_loop()
    while not task.done():
        left = None if deadline is None else deadline - loop.time()
        if left is not None and left <= 0:
            task.cancel()  # past its grace: given up, and awaited below as it unwinds
            await asyncio.wait([task])
            break
        try:
            await asyncio.wait_for(asyncio.shield(task), left)
        except asyncio.CancelledError as e:
            cancelled = e
            if deadline is None and grace is not None:
                deadline = loop.time() + grace
        except TimeoutError:
            pass  # its grace ran out: cancelled on the next look
        except BaseException:  # noqa: BLE001, S110 - re-raised or chained below
            pass  # `coro` itself failed → task is done; re-raised (or chained) below
    if cancelled is not None:
        raise cancelled from (None if task.cancelled() else task.exception())
    return task.result()
