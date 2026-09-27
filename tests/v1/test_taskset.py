"""`Taskset` iteration: framework-assigned `idx`, the selection views, and
`select(SelectConfig)` over list, generator, and `INFINITE` `load` implementations."""

import itertools
import logging
import random

import pytest
from pydantic import ValidationError

import verifiers.v1 as vf


class CountTask(vf.Task[vf.TaskData]):
    pass


def count_task(i: int) -> CountTask:
    return CountTask(
        vf.TaskData(idx=100 + i, id=f"id-{i}", name=f"t{i}", prompt=f"task {i}")
    )


class InfiniteTaskset(vf.Taskset[CountTask, vf.TasksetConfig]):
    INFINITE = True

    def load(self):
        for i in itertools.count():
            yield count_task(i)


class FiniteTaskset(vf.Taskset[CountTask, vf.TasksetConfig]):
    def load(self):
        return [count_task(i) for i in range(10)]


def finite() -> FiniteTaskset:
    return FiniteTaskset(vf.TasksetConfig())


def infinite() -> InfiniteTaskset:
    return InfiniteTaskset(vf.TasksetConfig())


def idxs(tasks) -> list[int]:
    return [task.data.idx for task in tasks]


def shuffled(xs, seed: int = 0) -> list[int]:
    xs = list(xs)
    random.Random(seed).shuffle(xs)
    return xs


def test_idx_is_the_load_position() -> None:
    # The taskset's own idx (100 + i) is overwritten.
    assert idxs(finite()) == list(range(10))
    assert idxs(finite().include(idx=[3, "5:7", "8:"])) == [3, 5, 6, 8, 9]
    assert idxs(finite().include(idx="0:2,4")) == [0, 1, 4]
    assert idxs(finite().exclude(idx=":8")) == [8, 9]
    for bad in [-1, "a", "5:5", "7:3", "1:2:3"]:
        with pytest.raises(ValidationError):
            vf.TaskMatchConfig(idx=[bad])


def test_include_and_exclude_match_any_listed_identity(caplog) -> None:
    key = list(finite())[4].key
    included = finite().include(idx=[0], ids=["id-1"], names=["t2"], keys=[key])
    assert idxs(included) == [0, 1, 2, 4]
    assert idxs(included.exclude(ids=["id-2"], idx=[0])) == [1, 4]
    with caplog.at_level(logging.WARNING):
        assert idxs(finite().include(names=["t1", "typo"], idx=[42])) == [1]
    assert "typo" in caplog.text and "42" in caplog.text


def test_select_chains_the_views_in_its_fixed_order() -> None:
    order = shuffled(range(10))
    assert idxs(finite().shuffle()) == order != list(range(10))
    assert idxs(finite().shuffle(seed=7)) == shuffled(range(10), 7)
    assert idxs(finite().shuffle().take(3)) == order[:3]
    assert idxs(finite().shuffle().skip(3).take(4)) == order[3:7]
    config = vf.SelectConfig(
        include={"idx": ["2:9"]},
        exclude={"names": ["t5"]},
        shuffle=True,
        skip=1,
        limit=2,
    )
    kept = [i for i in range(2, 9) if i != 5]
    assert idxs(finite().select(config)) == shuffled(kept)[1:3]
    # The Python views chain in any order: here shuffle the first 5.
    assert idxs(finite().take(5).shuffle()) == shuffled(range(5))


def test_infinite_tasksets_need_a_bound() -> None:
    assert finite().bounded and not infinite().bounded
    assert not infinite().include(idx="5:").bounded
    assert not infinite().include(idx=["0:4"], names=["t9"]).bounded
    assert infinite().take(3).bounded and infinite().include(idx=["0:4", 9]).bounded
    with pytest.raises(ValueError, match="infinite"):
        infinite().select(vf.SelectConfig(shuffle=True, limit=5))
    closed = vf.SelectConfig(include={"idx": "0:5"}, shuffle=True)
    assert sorted(idxs(infinite().select(closed))) == [0, 1, 2, 3, 4]


def test_views_are_lazy_and_leave_the_base_alone() -> None:
    built: list[int] = []

    class RecordingTaskset(InfiniteTaskset):
        def load(self):
            for i in itertools.count():
                built.append(i)
                yield count_task(i)

    base = RecordingTaskset(vf.TasksetConfig())
    assert idxs(base.take(3)) == [0, 1, 2] and built == [0, 1, 2]
    built.clear()
    assert idxs(base.include(idx=["2:4", 6])) == [2, 3, 6] and built == list(range(7))
    assert not base.bounded and base.transform is None
