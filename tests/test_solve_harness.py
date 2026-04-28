"""Tests for the SolveHarness — gold-patch validation through ComposableEnv.

Mirrors the patterns in ``test_composable_env.py``: a ``MockSandboxTaskSet``
backed by ``unittest.mock`` for sandbox operations, with the new
``solve_harness()`` factory wired in. We assert:

- ``ComposableEnv.post_sandbox_setup`` skips agent install when
  ``solve_only`` is set and instead invokes ``taskset.validate_instance``.
- The rollout never reaches LLM inference (no client used).
- ``state["solve_reason"]`` and ``state["solve_valid"]`` carry the
  ``validate(...)`` enum values.
- A failing ``validate_instance`` produces ``reason="test_failed"`` and
  ``solve_valid=False`` without crashing the rollout.
- A typed sandbox / infra failure surfaces in ``state["error"]`` and the
  reason is the corresponding enum.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

import verifiers as vf
from verifiers.envs.experimental.composable import (
    ComposableEnv,
    SandboxSpec,
    SandboxTaskSet,
)
from verifiers.envs.experimental.composable.harnesses.solve import solve_harness


# ── Mock rubric / taskset ────────────────────────────────────────────────


class _StubRubric(vf.Rubric):
    """Rubric that surfaces ``state["solve_valid"]`` as the reward.

    A real ``MultiSWERubric`` would re-run the test suite; this stub
    keeps the test focused on the harness wiring.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.add_reward_func(self.solved)

    async def solved(self, state, **kwargs) -> float:
        return float(state.get("solve_valid", False))


class _MockSolveTaskSet(SandboxTaskSet):
    """SandboxTaskSet with a mockable ``validate_instance``."""

    def __init__(self, *args, validate_outcome=True, validate_exc=None, **kwargs):
        super().__init__(*args, **kwargs)
        self._outcome = validate_outcome
        self._exc = validate_exc
        self.validate_calls = 0

    def get_instruction(self, info):
        return f"fix #{info.get('id', 0)}"

    def get_sandbox_spec(self, info):
        return SandboxSpec(image="python:3.11-slim", cpu_cores=2, memory_gb=2)

    def get_rubric(self):
        return _StubRubric()

    def get_workdir(self, info):
        return "/testbed"

    async def validate_instance(self, state) -> bool:
        self.validate_calls += 1
        if self._exc is not None:
            raise self._exc
        # Tasksets typically write test_output during validate_instance;
        # mirror that so the rubric path matches production.
        state["test_output"] = "PASSED" if self._outcome else "FAILED"
        return self._outcome


def _make_dataset(n: int = 1):
    from datasets import Dataset

    return Dataset.from_dict(
        {
            "info": [
                {"id": i, "instance_id": f"inst-{i}", "repo": "owner/repo"}
                for i in range(n)
            ],
            "answer": ["" for _ in range(n)],
        }
    )


def _make_env(taskset, harness=None):
    env = ComposableEnv(taskset=taskset, harness=harness or solve_harness())
    # Stub the sandbox client so post_sandbox_setup runs without a real backend.
    env.sandbox_client = SimpleNamespace(
        execute_command=AsyncMock(),
        teardown=lambda: None,
    )
    return env


# ── Factory ──────────────────────────────────────────────────────────────


def test_solve_harness_factory_defaults():
    h = solve_harness()
    assert h.solve_only is True
    assert h.install_script is None
    # run_command must be a no-op string — start_agent never runs it but
    # CliAgentEnv stores it on self.run_command, so it has to be valid.
    assert h.run_command == "true"


def test_solve_harness_auto_keeps_sandbox_for_scoring():
    taskset = _MockSolveTaskSet(dataset=_make_dataset(), name="solve-test")
    env = ComposableEnv(taskset=taskset, harness=solve_harness())
    assert env.keep_sandbox_for_scoring is True


# ── post_sandbox_setup short-circuits ────────────────────────────────────


@pytest.mark.asyncio
async def test_post_sandbox_setup_skips_install_and_runs_validate():
    taskset = _MockSolveTaskSet(dataset=_make_dataset(), name="solve-test")
    env = _make_env(taskset)

    state = {"sandbox_id": "sbx", "info": {"id": 0, "instance_id": "inst-0"}}
    await env.post_sandbox_setup(state)

    assert taskset.validate_calls == 1
    # No install / upload / mkdir on the sandbox client
    env.sandbox_client.execute_command.assert_not_awaited()
    assert state["solve_valid"] is True
    assert state["solve_reason"] == "pass"
    assert state["agent_completed"] is True
    assert state["test_output"] == "PASSED"


@pytest.mark.asyncio
async def test_failed_validation_marks_test_failed_without_crashing():
    taskset = _MockSolveTaskSet(
        dataset=_make_dataset(), name="solve-test", validate_outcome=False
    )
    env = _make_env(taskset)

    state = {"sandbox_id": "sbx", "info": {"id": 0}}
    await env.post_sandbox_setup(state)

    assert state["solve_valid"] is False
    assert state["solve_reason"] == "test_failed"
    assert "error" not in state  # non-exceptional failure
    assert state["agent_completed"] is True


@pytest.mark.asyncio
async def test_gold_apply_failure_classified_as_gold_apply_failed():
    taskset = _MockSolveTaskSet(
        dataset=_make_dataset(),
        name="solve-test",
        validate_exc=RuntimeError("Gold patch apply failed: exit_code=1"),
    )
    env = _make_env(taskset)

    state = {"sandbox_id": "sbx", "info": {"id": 0}}
    await env.post_sandbox_setup(state)

    assert state["solve_valid"] is False
    assert state["solve_reason"] == "gold_apply_failed"


@pytest.mark.asyncio
async def test_infra_error_surfaces_on_state():
    taskset = _MockSolveTaskSet(
        dataset=_make_dataset(),
        name="solve-test",
        validate_exc=vf.SandboxError("sandbox blew up"),
    )
    env = _make_env(taskset)

    state = {"sandbox_id": "sbx", "info": {"id": 0}}
    await env.post_sandbox_setup(state)

    assert state["solve_valid"] is False
    assert state["solve_reason"] == "sandbox_error"
    assert isinstance(state["error"], vf.SandboxError)


# ── stop condition ───────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_solve_harness_completed_stop_condition_fires():
    taskset = _MockSolveTaskSet(dataset=_make_dataset(), name="solve-test")
    env = _make_env(taskset)

    state = {"sandbox_id": "sbx", "info": {"id": 0}, "solve_reason": "pass"}
    assert await env.solve_harness_completed(state) is True

    # Without solve_reason, the stop must not fire (validation hasn't run).
    state = {"sandbox_id": "sbx", "info": {"id": 0}}
    assert await env.solve_harness_completed(state) is False


@pytest.mark.asyncio
async def test_solve_harness_completed_inactive_for_normal_harness():
    from verifiers.envs.experimental.composable import Harness

    taskset = _MockSolveTaskSet(dataset=_make_dataset(), name="solve-test")
    env = ComposableEnv(taskset=taskset, harness=Harness(run_command="true"))
    # Even if solve_reason somehow appears on a non-solve env, the stop
    # condition must not fire — solve_only gates it.
    state = {"sandbox_id": "sbx", "info": {"id": 0}, "solve_reason": "pass"}
    assert await env.solve_harness_completed(state) is False


# ── start_agent no-op ────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_start_agent_is_noop_under_solve_only():
    taskset = _MockSolveTaskSet(dataset=_make_dataset(), name="solve-test")
    env = _make_env(taskset)

    state = {"sandbox_id": "sbx", "info": {"id": 0}}
    # If start_agent were not a no-op, it would call into
    # sandbox_client.start_background_job; the mock has no such attr.
    await env.start_agent(state)
    assert state.get("background_job") is None


# ── full rollout (no LLM client invoked) ─────────────────────────────────


@pytest.mark.asyncio
async def test_full_rollout_does_not_invoke_llm(mock_client):
    """End-to-end: rollout completes via the solve_harness_completed stop
    without ever touching the LLM client. ``setup_state`` is patched to
    skip the real sandbox creation; ``post_sandbox_setup`` runs and
    triggers ``validate_instance`` so the standard stop loop exits.
    """
    taskset = _MockSolveTaskSet(dataset=_make_dataset(), name="solve-test")
    env = _make_env(taskset)

    # Track LLM invocations — the rollout must not produce any.
    original_get_response = env.get_model_response

    async def assert_no_llm(*args, **kwargs):  # pragma: no cover — must not run
        raise AssertionError("solve_harness rollout must not call get_model_response")

    env.get_model_response = assert_no_llm  # type: ignore[assignment]

    # Skip the real sandbox-creating path; inject pre-built rollout state
    # then call post_sandbox_setup so validate_instance fires.
    async def fake_setup_state(state):
        state.update(
            {
                "rollout_id": "rollout_test",
                "sandbox_id": "sbx",
                "agent_completed": False,
                "interception_base_url": "",
            }
        )
        await env.post_sandbox_setup(state)
        return state

    env.setup_state = fake_setup_state  # type: ignore[assignment]

    state = await env.rollout(
        input={
            "prompt": [{"role": "user", "content": "fix #0"}],
            "info": {"id": 0, "instance_id": "inst-0"},
            "answer": "",
            "task": "default",
        },
        client=mock_client,
        model="not-a-real-model",
    )
    assert state["stop_condition"] == "solve_harness_completed"
    assert state["solve_reason"] == "pass"
    assert state["solve_valid"] is True
    assert taskset.validate_calls == 1
    # Ensure we didn't accidentally rebind something that bypasses the assertion
    assert env.get_model_response is assert_no_llm
    assert original_get_response is not assert_no_llm
