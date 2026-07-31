"""`Taskset` iteration and the `head`/`shuffle` views over list, generator,
and `INFINITE` `load` implementations."""

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


def idxs(tasks) -> list[int]:
    return [task.data.idx for task in tasks]


def test_head_bounds_an_infinite_taskset() -> None:
    tasks = list(InfiniteTaskset(vf.TasksetConfig()).head(5))
    assert idxs(tasks) == [0, 1, 2, 3, 4]


def test_iteration_materializes_a_finite_taskset() -> None:
    taskset = FiniteTaskset(vf.TasksetConfig())
    assert len(list(taskset)) == 10
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

    list(RecordingTaskset(vf.TasksetConfig()).head(3))
    assert built == [0, 1, 2]


def test_views_do_not_mutate_the_base_taskset() -> None:
    taskset = InfiniteTaskset(vf.TasksetConfig())
    view = taskset.head(3)
    assert view.INFINITE is False and taskset.INFINITE is True
    assert idxs(taskset.head(2)) == [0, 1]  # base iterates untransformed
