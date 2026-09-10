"""Asyncio helpers shared across the framework."""

import asyncio
from collections.abc import AsyncIterable, Awaitable, Callable
from typing import TypeVar

T = TypeVar("T")
R = TypeVar("R")


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


async def run_stream(
    source: AsyncIterable[T],
    run: Callable[[T], Awaitable[R]],
    *,
    window: int | None,
) -> list[R]:
    """Consume `source` — an async iterable of tasks, typically a feed that waits for
    its next item — running each through `run` with at most `window` in flight (None:
    no bound), and return every result in submission order once the source has ended
    and the last run is done. Back-pressure: the next item is pulled only once a run
    has freed a place in the window, so a feed is read at the pace the runs can take.
    One run failing cancels the rest, waits for them to unwind (so nothing keeps
    uploading into a run the caller is already closing), and re-raises; the source is
    closed (`aclose()`) on any exit, so a feed's waiter is gone too. The caller decides
    when the source ends: `vf eval -n` hands in a finite one, a service of your own
    one that ends when the service does. The pull is its own task raced against the
    runs' first failure, so a run that fails while the feed is quiet aborts at once,
    not when the feed next yields (which may be never). Not a `TaskGroup`: that
    wraps errors in an `ExceptionGroup`, and a CLI's `main` would no longer see a
    `KeyboardInterrupt` as Ctrl-C."""
    items = aiter(source)
    started: list[asyncio.Task[R]] = []
    pending: set[asyncio.Task[R]] = set()
    pull: asyncio.Task[T] | None = None
    # Resolved with the first run to fail (by a done-callback, so the wait on the
    # pull stays O(1) however many runs a static eval has in flight).
    failed: asyncio.Future[asyncio.Task[R]] = asyncio.get_running_loop().create_future()

    def watch(task: asyncio.Task[R]) -> None:
        if not task.cancelled() and task.exception() is not None and not failed.done():
            failed.set_result(task)

    try:
        while True:
            # Back-pressure: the next item is pulled only once a run has freed.
            while window is not None and len(pending) >= window:
                done, pending = await asyncio.wait(
                    pending, return_when=asyncio.FIRST_COMPLETED
                )
                for task in done:
                    task.result()  # a failed run raises here: cancel the rest
            pull = asyncio.ensure_future(anext(items))
            await asyncio.wait({pull, failed}, return_when=asyncio.FIRST_COMPLETED)
            if failed.done():
                failed.result().result()  # raises the run's error: cancel the rest
            try:
                item = pull.result()
            except StopAsyncIteration:
                break
            started.append(asyncio.ensure_future(run(item)))
            started[-1].add_done_callback(watch)
            pending.add(started[-1])
        return await asyncio.gather(*started)
    except BaseException:
        # The pull too: the source must not still be running when it is closed.
        tasks = [*started, pull] if pull is not None else started
        for task in tasks:
            task.cancel()
        # return_exceptions: wait for every task, not just the first to cancel.
        await asyncio.gather(*tasks, return_exceptions=True)
        raise
    finally:
        if (aclose := getattr(items, "aclose", None)) is not None:
            await aclose()
