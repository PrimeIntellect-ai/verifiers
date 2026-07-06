"""Tests for the v1 RubricJudge batching (`max_criteria`)."""

from pathlib import Path

import pytest

from verifiers.v1.judges.rubric import Criterion, RubricJudge, RubricJudgeConfig

CRITERIA = [Criterion(name=f"c{i}", text="t") for i in range(7)]


def _judge(max_criteria):
    # path is only read lazily (via `.criteria`); `_batches` takes criteria directly, so any
    # path works and no client call is made.
    return RubricJudge(
        RubricJudgeConfig(name="r", path=Path("unused.toml"), max_criteria=max_criteria)
    )


def test_batches_default_is_one_call():
    assert [len(b) for b in _judge(None)._batches(CRITERIA)] == [7]


def test_batches_per_criterion():
    assert [len(b) for b in _judge(1)._batches(CRITERIA)] == [1] * 7


def test_batches_chunked():
    assert [len(b) for b in _judge(3)._batches(CRITERIA)] == [3, 3, 1]


def test_batches_larger_than_rubric_is_one_call():
    assert [len(b) for b in _judge(99)._batches(CRITERIA)] == [7]


def test_batches_rejects_below_one():
    with pytest.raises(ValueError):
        _judge(0)._batches(CRITERIA)
