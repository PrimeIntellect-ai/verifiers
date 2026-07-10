"""`Taskset.select` over list, generator, and infinite `load` implementations."""

import itertools

import pytest

import verifiers.v1 as vf


class CountTask(vf.Task[vf.TaskData]):
    pass


class CountConfig(vf.TasksetConfig):
    num_tasks: int | None = None


class CountTaskset(vf.Taskset[CountTask, CountConfig]):
    @property
    def infinite(self) -> bool:
        return self.config.num_tasks is None

    def load(self):
        indices = (
            itertools.count()
            if self.config.num_tasks is None
            else range(self.config.num_tasks)
        )
        for i in indices:
            yield CountTask(vf.TaskData(idx=i, prompt=f"task {i}"))


def idxs(tasks: list[CountTask]) -> list[int]:
    return [task.data.idx for task in tasks]


def test_select_bounds_an_infinite_taskset() -> None:
    assert idxs(CountTaskset(CountConfig()).select(num_tasks=5)) == [0, 1, 2, 3, 4]


def test_select_infinite_requires_num_tasks() -> None:
    with pytest.raises(ValueError, match="infinite"):
        CountTaskset(CountConfig()).select()


def test_select_infinite_ignores_shuffle() -> None:
    tasks = CountTaskset(CountConfig()).select(num_tasks=3, shuffle=True)
    assert idxs(tasks) == [0, 1, 2]


def test_select_finite() -> None:
    taskset = CountTaskset(CountConfig(num_tasks=10))
    assert len(taskset.select()) == 10
    assert idxs(taskset.select(num_tasks=4)) == [0, 1, 2, 3]


def test_select_shuffle_samples_the_whole_taskset_reproducibly() -> None:
    taskset = CountTaskset(CountConfig(num_tasks=10))
    first = idxs(taskset.select(num_tasks=5, shuffle=True))
    assert first == idxs(taskset.select(num_tasks=5, shuffle=True))
    assert len(first) == 5 and set(first) <= set(range(10))
    assert first != [0, 1, 2, 3, 4]  # sampled from the whole set, not the head


def test_select_only_builds_what_the_run_takes() -> None:
    built: list[int] = []

    class RecordingTaskset(CountTaskset):
        def load(self):
            for i in itertools.count():
                built.append(i)
                yield CountTask(vf.TaskData(idx=i, prompt=f"task {i}"))

    RecordingTaskset(CountConfig()).select(num_tasks=3)
    assert built == [0, 1, 2]
