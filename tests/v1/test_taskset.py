"""`Taskset` iteration and selection: the framework-assigned `idx`, the lazy views
(`include`, `exclude`, `shuffle`, `skip`, `take`), and `select(SelectConfig)`."""

import itertools
import logging
import random

import pytest
from pydantic import ValidationError

import verifiers.v1 as vf


class CountTask(vf.Task[vf.TaskData]):
    pass


def count_task(i: int) -> CountTask:
    # A taskset-set idx (100 + i) that iteration must overwrite.
    return CountTask(
        vf.TaskData(idx=100 + i, id=f"id-{i}", name=f"t{i}", prompt=f"task {i}")
    )


class FiniteTaskset(vf.Taskset[CountTask, vf.TasksetConfig]):
    def load(self):
        return [count_task(i) for i in range(10)]


class InfiniteTaskset(vf.Taskset[CountTask, vf.TasksetConfig]):
    INFINITE = True

    def load(self):
        for i in itertools.count():
            yield count_task(i)


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


# Views


def test_iteration_sets_idx_to_the_load_position() -> None:
    assert idxs(finite()) == list(range(10))


def test_include_keeps_tasks_that_any_list_names() -> None:
    assert idxs(finite().include(idx=[3, "5:7", "8:"])) == [3, 5, 6, 8, 9]
    assert idxs(finite().include(idx=":2")) == [0, 1]
    assert idxs(finite().include(idx=["1:8:3", "::4"])) == [0, 1, 4, 7, 8]
    assert idxs(finite().include(ids=["id-1"])) == [1]
    assert idxs(finite().include(names=["t2"])) == [2]
    key = list(finite())[4].key
    assert idxs(finite().include(keys=[key])) == [4]
    union = finite().include(idx=[0], ids=["id-1"], names=["t2"], keys=[key])
    assert idxs(union) == [0, 1, 2, 4]


def test_exclude_drops_tasks_that_any_list_names() -> None:
    assert idxs(finite().exclude(idx=":7", names=["t8"])) == [7, 9]


def test_unmatched_entries_warn(caplog) -> None:
    with caplog.at_level(logging.WARNING):
        assert idxs(finite().include(names=["t1", "typo"], idx=[42])) == [1]
    assert "typo" in caplog.text and "42" in caplog.text


def test_shuffle_is_seeded_and_reproducible() -> None:
    assert idxs(finite().shuffle()) == shuffled(range(10)) != list(range(10))
    assert idxs(finite().shuffle(seed=7)) == shuffled(range(10), 7)


def test_skip_and_take_slice_the_stream() -> None:
    assert idxs(finite().skip(7)) == [7, 8, 9]
    assert idxs(finite().take(3)) == [0, 1, 2]
    assert idxs(finite().skip(2).take(2)) == [2, 3]


def test_views_chain_in_call_order_and_leave_the_base_alone() -> None:
    base = finite()
    assert idxs(base.take(5).shuffle()) == shuffled(range(5))
    assert idxs(base.shuffle().take(5)) == shuffled(range(10))[:5]
    assert base.transform is None and idxs(base) == list(range(10))


def test_views_read_load_lazily() -> None:
    built: list[int] = []

    class RecordingTaskset(InfiniteTaskset):
        def load(self):
            for i in itertools.count():
                built.append(i)
                yield count_task(i)

    base = RecordingTaskset(vf.TasksetConfig())
    assert idxs(base.take(3)) == [0, 1, 2] and built == [0, 1, 2]
    built.clear()
    # An idx-only include stops reading after its last position.
    assert idxs(base.include(idx=["2:4", 6])) == [2, 3, 6] and built == list(range(7))


def test_only_a_bounded_view_can_shuffle() -> None:
    assert finite().bounded and not infinite().bounded
    assert infinite().take(3).bounded
    assert infinite().include(idx=["0:4", 9]).bounded
    assert not infinite().include(idx="::2").bounded
    assert not infinite().include(idx="5:").bounded
    assert not infinite().include(idx=["0:4"], names=["t9"]).bounded
    with pytest.raises(ValueError, match="infinite"):
        infinite().shuffle()
    assert sorted(idxs(infinite().take(5).shuffle())) == [0, 1, 2, 3, 4]


# Config and select


def test_task_match_config_parses_idx() -> None:
    assert vf.TaskMatchConfig(idx=["3", "0:2,4", "8:"]).idx == [3, "0:2", 4, "8:"]
    assert vf.TaskMatchConfig(idx=5).idx == [5]
    assert vf.TaskMatchConfig(idx="0:9:3").idx_stop == 9
    for bad in [-1, "a", "5:5", "7:3", "1:2:0", "1:2:3:4"]:
        with pytest.raises(ValidationError):
            vf.TaskMatchConfig(idx=[bad])


def test_select_config_defaults_and_short_aliases() -> None:
    config = vf.SelectConfig()
    assert config.include.empty and config.exclude.empty
    assert (config.shuffle, config.seed, config.skip, config.limit) == (
        False,
        0,
        0,
        None,
    )
    aliased = vf.SelectCLIConfig.model_validate({"n": 3, "s": True})
    assert (aliased.limit, aliased.shuffle) == (3, True)
    with pytest.raises(ValidationError):
        vf.SelectConfig.model_validate({"n": 3})
    with pytest.raises(ValidationError):
        vf.SelectConfig(limit=0)


def test_default_select_yields_every_task() -> None:
    assert idxs(finite().select(vf.SelectConfig())) == list(range(10))


def test_select_applies_its_steps_in_a_fixed_order() -> None:
    config = vf.SelectConfig(
        include={"idx": ["2:9"]},
        exclude={"names": ["t5"]},
        shuffle=True,
        seed=3,
        skip=1,
        limit=2,
    )
    kept = [i for i in range(2, 9) if i != 5]
    assert idxs(finite().select(config)) == shuffled(kept, 3)[1:3]
    chained = (
        finite().include(idx=["2:9"]).exclude(names=["t5"]).shuffle(3).skip(1).take(2)
    )
    assert idxs(finite().select(config)) == idxs(chained)


def test_select_bounds_an_infinite_taskset() -> None:
    assert infinite().select(vf.SelectConfig(limit=4)).bounded
    closed = vf.SelectConfig(include={"idx": "0:5"}, shuffle=True)
    assert sorted(idxs(infinite().select(closed))) == [0, 1, 2, 3, 4]
    # The shuffle comes before the limit, so the limit cannot bound it.
    with pytest.raises(ValueError, match="infinite"):
        infinite().select(vf.SelectConfig(shuffle=True, limit=5))
