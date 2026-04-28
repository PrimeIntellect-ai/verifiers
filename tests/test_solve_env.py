"""Focused tests for SolveEnv (gold-patch validation Env)."""

from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

import verifiers as vf
from verifiers.envs.experimental.composable import (
    SandboxSpec,
    SandboxTaskSet,
    SolveEnv,
)
from verifiers.envs.experimental.composable.solve_env import SolveRubric
from verifiers.envs.experimental.composable.task import _classify_validate_outcome


class _MockTaskSet(SandboxTaskSet):
    def get_instruction(self, info):
        return f"solve {info.get('id', 0)}"

    def get_sandbox_spec(self, info):
        return SandboxSpec(image="python:3.11-slim", cpu_cores=2, memory_gb=2)

    def get_rubric(self):
        return vf.Rubric()


def _dataset(n=1):
    from datasets import Dataset

    return Dataset.from_dict(
        {"info": [{"id": i} for i in range(n)], "answer": ["" for _ in range(n)]}
    )


def _build_env(validate_returns=True, validate_raises=None):
    taskset = _MockTaskSet(dataset=_dataset(), name="mock")
    taskset.setup = AsyncMock()
    if validate_raises is not None:
        taskset.validate_instance = AsyncMock(side_effect=validate_raises)
    else:
        taskset.validate_instance = AsyncMock(return_value=validate_returns)
    env = SolveEnv(taskset=taskset)
    env.create_sandbox = AsyncMock(
        side_effect=lambda s, _r: s.update({"sandbox_id": "sbx"})
    )
    env.sandbox_client = SimpleNamespace(teardown=lambda: None)
    return env


@pytest.mark.asyncio
async def test_solve_completed_always_true():
    env = _build_env()
    assert await env.solve_completed({}) is True


@pytest.mark.asyncio
async def test_solve_reward_reads_state_reward():
    rubric = SolveRubric()
    assert await rubric.solve_reward({"reward": 1.0}) == 1.0
    assert await rubric.solve_reward({"reward": 0.0}) == 0.0
    assert await rubric.solve_reward({}) == 0.0


def test_classify_reasons():
    assert _classify_validate_outcome(True, None, {})[0] == "pass"
    assert _classify_validate_outcome(False, None, {})[0] == "test_failed"
    assert (
        _classify_validate_outcome(False, vf.InfraError("boom"), {})[0]
        == "sandbox_error"
    )
    assert (
        _classify_validate_outcome(False, RuntimeError("apply failed"), {})[0]
        == "gold_apply_failed"
    )
    assert (
        _classify_validate_outcome(False, ValueError("oops"), {})[0] == "setup_failed"
    )


@pytest.mark.asyncio
async def test_setup_state_populates_attempts_elapsed_and_reason():
    env = _build_env(validate_returns=True)
    state = await env.setup_state({"info": {"id": 0}, "answer": ""})
    assert state["attempts"] == 1
    assert state["reward"] == 1.0
    assert state["reason"] == "pass"
    assert isinstance(state["elapsed_s"], float) and state["elapsed_s"] >= 0.0
