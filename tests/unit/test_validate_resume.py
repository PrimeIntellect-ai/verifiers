"""`vf-validate --resume` keeps a Harbor task's finished rows."""

from types import SimpleNamespace

import pytest

from verifiers.v1.cli import validate
from verifiers.v1.tasksets.harbor.taskset import HarborData, HarborTask
from verifiers.v1.validation import ValidationAttempt, ValidationResult


def scored(reward: float) -> ValidationAttempt:
    return ValidationAttempt(
        reward=reward,
        valid=True,
        trace={"rewards": {"reward": {"score": reward, "weight": 1.0}}},
    )


@pytest.mark.parametrize("mode", ["all", "noop"])
async def test_a_finished_harbor_row_is_final(monkeypatch, mode):
    """A Harbor untouched attempt records its scoring as a native noop row does, so a
    resume keeps the row instead of grading the task again."""
    task = HarborTask(HarborData(idx=0, prompt="p", name="t"))

    async def validate_(*args, **kwargs):
        return ValidationResult(
            gold=scored(1.0) if mode == "all" else ValidationAttempt(enabled=False),
            baseline=scored(0.0),
        )

    monkeypatch.setattr(task, "validate", validate_)
    config = SimpleNamespace(
        runtime=None, timeout=SimpleNamespace(setup=None, total=None)
    )
    row = await validate._run_harbor_check(task, config, mode)
    row["task_key"] = "k"
    assert validate._is_final(row, "k", mode)
