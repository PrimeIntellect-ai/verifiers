"""`Taskset` iteration and the `head`/`shuffle` views over list, generator,
and `INFINITE` `load` implementations; `stream()` and the runner's windowed
pull over a taskset whose tasks arrive over time."""

import asyncio
import contextlib
import itertools
from collections.abc import AsyncGenerator

import pytest

import verifiers.v1 as vf
from verifiers.v1.cli.eval.runner import _take, run_stream
from verifiers.v1.env import RunSlot


class CountTask(vf.Task[vf.TaskData]):
    pass


class InfiniteTaskset(vf.Taskset[CountTask, vf.TasksetConfig]):
    INFINITE = True

    def load(self):
        for i in itertools.count():
            yield CountTask(vf.TaskData(idx=i, prompt=f"task {i}"))


class FiniteTaskset(vf.Taskset[CountTask, vf.TasksetConfig]):
    def load(self):
        for i in range(10):
            yield CountTask(vf.TaskData(idx=i, prompt=f"task {i}"))


def idxs(tasks) -> list[int]:
    return [task.data.idx for task in tasks]


def test_iteration_materializes_a_finite_taskset() -> None:
    taskset = FiniteTaskset(vf.TasksetConfig())
    tasks = list(taskset)
    assert len(tasks) == 10
    assert all(task.key == task.hash for task in tasks)
    assert idxs(taskset.head(4)) == [0, 1, 2, 3]


def test_head_returns_the_same_taskset_type() -> None:
    view = FiniteTaskset(vf.TasksetConfig()).head(4)
    assert isinstance(view, FiniteTaskset)
    assert view.task_type() is CountTask


def test_shuffle_samples_the_whole_taskset_reproducibly() -> None:
    taskset = FiniteTaskset(vf.TasksetConfig())
    first = idxs(taskset.shuffle().head(5))
    assert first == idxs(taskset.shuffle().head(5))
    assert len(first) == 5 and set(first) <= set(range(10))
    assert first != [0, 1, 2, 3, 4]  # sampled from the whole set, not the head


def test_shuffle_seed_changes_the_sample() -> None:
    taskset = FiniteTaskset(vf.TasksetConfig())
    seeded = idxs(taskset.shuffle(seed=7).head(5))
    assert seeded == idxs(taskset.shuffle(seed=7).head(5))  # reproducible per seed
    assert seeded != idxs(taskset.shuffle().head(5))  # differs from the default seed


def test_shuffle_raises_on_infinite() -> None:
    with pytest.raises(ValueError, match="infinite"):
        InfiniteTaskset(vf.TasksetConfig()).shuffle()


def test_head_then_shuffle_bounds_an_infinite_taskset() -> None:
    head = idxs(InfiniteTaskset(vf.TasksetConfig()).head(5).shuffle())
    assert sorted(head) == [0, 1, 2, 3, 4]
    assert head != [0, 1, 2, 3, 4]  # shuffled within the bounded head


def test_head_only_builds_what_the_run_takes() -> None:
    built: list[int] = []

    class RecordingTaskset(InfiniteTaskset):
        def load(self):
            for i in itertools.count():
                built.append(i)
                yield CountTask(vf.TaskData(idx=i, prompt=f"task {i}"))

    tasks = list(RecordingTaskset(vf.TasksetConfig()).head(3))
    assert idxs(tasks) == [0, 1, 2]
    assert built == [0, 1, 2]


def test_views_do_not_mutate_the_base_taskset() -> None:
    taskset = InfiniteTaskset(vf.TasksetConfig())
    view = taskset.head(3)
    assert view.INFINITE is False and taskset.INFINITE is True
    assert idxs(taskset.head(2)) == [0, 1]  # base iterates untransformed


class QueueTaskset(vf.Taskset[CountTask, vf.TasksetConfig]):
    """Tasks that arrive over time: `stream()` waits on a queue (`None` ends it) and
    records what it pulled and whether the runner closed it."""

    queue: asyncio.Queue
    pulled = 0
    closed = False

    def load(self):
        return []

    async def stream(self):
        try:
            while (i := await self.queue.get()) is not None:
                self.pulled += 1
                yield CountTask(vf.TaskData(idx=i, prompt=f"task {i}"))
        finally:
            self.closed = True


def queued(*items: int | None) -> QueueTaskset:
    taskset = QueueTaskset(vf.TasksetConfig())
    taskset.queue = asyncio.Queue()
    for item in items:
        taskset.queue.put_nowait(item)
    return taskset


async def groups(tasks) -> AsyncGenerator[list[RunSlot], None]:
    """One slot per task, closing the source when closed — the runner's planner."""
    async with contextlib.aclosing(tasks):
        async for task in tasks:
            yield [RunSlot(task)]


async def test_default_stream_is_iteration() -> None:
    taskset = FiniteTaskset(vf.TasksetConfig())
    assert taskset.streaming is False
    assert [t.data.idx async for t in taskset.head(2).stream()] == [0, 1]
    assert [t.data.idx async for t in _take(taskset.stream(), 3)] == [0, 1, 2]


async def test_stream_pulls_only_as_slots_free() -> None:
    """The window is back-pressure: with `window` slots in flight the next task is
    not pulled; `-n` ends the pull without touching the (n+1)th; results keep
    submission order and the source is closed."""
    taskset = queued(0, 1, 2, 3, 4)  # never ended: the feed is still waiting
    assert taskset.streaming is True
    running = asyncio.Semaphore(0)
    gate = asyncio.Event()

    async def run_slot(slot: RunSlot):
        running.release()
        await gate.wait()
        return slot.task.data.idx

    run = asyncio.ensure_future(
        run_stream(groups(_take(taskset.stream(), 4)), run_slot, window=2)
    )
    for _ in range(2):
        await running.acquire()
    await asyncio.sleep(0)
    assert taskset.pulled == 2 and not run.done()
    gate.set()
    assert await run == [0, 1, 2, 3]
    assert taskset.pulled == 4 and taskset.closed


async def test_stream_ends_with_its_source() -> None:
    taskset = queued(2, 1, 0, None)
    order: list[int] = []

    async def run_slot(slot: RunSlot):
        await asyncio.sleep(0.01 * slot.task.data.idx)
        order.append(slot.task.data.idx)
        return slot.task.data.idx

    results = await run_stream(groups(taskset.stream()), run_slot, window=None)
    assert results == [2, 1, 0] and order == [0, 1, 2]
    assert taskset.closed


async def test_failing_slot_cancels_the_rest_and_closes_the_stream() -> None:
    taskset = queued(0, 1, 2, 3)  # never ended
    running = asyncio.Semaphore(0)
    failed = asyncio.Event()
    cancelled: list[int] = []

    async def run_slot(slot: RunSlot):
        running.release()
        if slot.task.data.idx == 1:
            await failed.wait()
            raise RuntimeError("boom")
        try:
            await asyncio.sleep(60)
        except asyncio.CancelledError:
            cancelled.append(slot.task.data.idx)
            raise

    run = asyncio.ensure_future(
        run_stream(groups(taskset.stream()), run_slot, window=3)
    )
    for _ in range(3):
        await running.acquire()
    failed.set()
    with pytest.raises(RuntimeError, match="boom"):
        await run
    assert sorted(cancelled) == [0, 2]
    assert taskset.pulled == 3 and taskset.closed  # the feed's waiter is gone too


async def test_cancelling_the_run_closes_a_waiting_stream() -> None:
    taskset = queued()  # nothing yet: `stream()` is parked on the feed
    run = asyncio.ensure_future(
        run_stream(groups(taskset.stream()), lambda slot: None, window=None)
    )
    await asyncio.sleep(0)
    run.cancel()
    with pytest.raises(asyncio.CancelledError):
        await run
    assert taskset.closed
