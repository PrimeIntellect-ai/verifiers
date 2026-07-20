"""`Taskset.select` over list, generator, and `INFINITE` `load` implementations."""

import itertools

import pytest

import verifiers.v1 as vf


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


def idxs(tasks: list[CountTask]) -> list[int]:
    return [task.data.idx for task in tasks]


def test_select_bounds_an_infinite_taskset() -> None:
    tasks = InfiniteTaskset(vf.TasksetConfig()).select(num_tasks=5)
    assert idxs(tasks) == [0, 1, 2, 3, 4]


def test_select_infinite_requires_num_tasks() -> None:
    with pytest.raises(ValueError, match="infinite"):
        InfiniteTaskset(vf.TasksetConfig()).select()


def test_select_infinite_ignores_shuffle() -> None:
    tasks = InfiniteTaskset(vf.TasksetConfig()).select(num_tasks=3, shuffle=True)
    assert idxs(tasks) == [0, 1, 2]


def test_select_finite() -> None:
    taskset = FiniteTaskset(vf.TasksetConfig())
    assert len(taskset.select()) == 10
    assert idxs(taskset.select(num_tasks=4)) == [0, 1, 2, 3]


def test_select_shuffle_samples_the_whole_taskset_reproducibly() -> None:
    taskset = FiniteTaskset(vf.TasksetConfig())
    first = idxs(taskset.select(num_tasks=5, shuffle=True))
    assert first == idxs(taskset.select(num_tasks=5, shuffle=True))
    assert len(first) == 5 and set(first) <= set(range(10))
    assert first != [0, 1, 2, 3, 4]  # sampled from the whole set, not the head


def test_select_only_builds_what_the_run_takes() -> None:
    built: list[int] = []

    class RecordingTaskset(InfiniteTaskset):
        def load(self):
            for i in itertools.count():
                built.append(i)
                yield CountTask(vf.TaskData(idx=i, prompt=f"task {i}"))

    RecordingTaskset(vf.TasksetConfig()).select(num_tasks=3)
    assert built == [0, 1, 2]
