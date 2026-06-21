from unittest.mock import AsyncMock, Mock

import pytest
from datasets import Dataset

import verifiers as vf
from verifiers.envs.experimental.composable import (
    SandboxDebugEnv,
    SandboxSpec,
    SandboxTaskSet,
)
from verifiers.envs.experimental.composable.tasksets.swe.swe_lego.taskset import (
    SWELegoRubric,
)
from verifiers.rubrics.experimental.hybrid_math_rubric import RemoteHybridMathRubric


class DummySandboxTaskSet(SandboxTaskSet):
    def __init__(self) -> None:
        super().__init__(
            dataset=Dataset.from_list([{"question": "noop", "info": {}, "answer": ""}])
        )

    def get_instruction(self, info: dict) -> str:
        return "Run the debug step."

    def get_rubric(self) -> vf.Rubric:
        return vf.Rubric()

    def get_sandbox_spec(self, info: dict) -> SandboxSpec:
        return SandboxSpec()


@pytest.mark.asyncio
async def test_swe_taskset_cleanup_delete_failure_records_rollout_error() -> None:
    client = AsyncMock()
    client.delete.side_effect = RuntimeError("delete outage")
    deregister = Mock()
    rubric = SWELegoRubric(taskset=object())
    state = {
        "error": None,
        "_sandbox_deregister": deregister,
        "sandbox_client": client,
        "sandbox_id": "sb-swe",
    }

    await rubric.cleanup(state)

    client.delete.assert_awaited_once_with("sb-swe")
    deregister.assert_not_called()
    assert isinstance(state["error"], vf.SandboxDeleteError)
    assert str(state["error"]) == "Failed to delete sandbox sb-swe: delete outage"
    assert state["cleanup_errors"] == [
        {
            "type": "SandboxDeleteError",
            "message": "Failed to delete sandbox sb-swe: delete outage",
            "scope": "taskset_cleanup",
            "sandbox_id": "sb-swe",
        }
    ]


@pytest.mark.asyncio
async def test_swe_taskset_cleanup_delete_success_deregisters_sandbox() -> None:
    client = AsyncMock()
    deregister = Mock()
    rubric = SWELegoRubric(taskset=object())
    state = {
        "_sandbox_deregister": deregister,
        "sandbox_client": client,
        "sandbox_id": "sb-swe",
    }

    await rubric.cleanup(state)

    client.delete.assert_awaited_once_with("sb-swe")
    deregister.assert_called_once_with("sb-swe")
    assert "cleanup_errors" not in state


@pytest.mark.asyncio
async def test_cleanup_delete_failure_does_not_overwrite_existing_error() -> None:
    client = AsyncMock()
    client.delete.side_effect = RuntimeError("delete outage")
    rubric = SWELegoRubric(taskset=object())
    existing_error = vf.ToolError("already failed")
    state = {
        "error": existing_error,
        "sandbox_client": client,
        "sandbox_id": "sb-swe",
    }

    await rubric.cleanup(state)

    assert state["error"] is existing_error
    assert state["cleanup_errors"] == [
        {
            "type": "SandboxDeleteError",
            "message": "Failed to delete sandbox sb-swe: delete outage",
            "scope": "taskset_cleanup",
            "sandbox_id": "sb-swe",
        }
    ]


@pytest.mark.asyncio
async def test_sandbox_debug_cleanup_delete_failure_records_rollout_error() -> None:
    env = SandboxDebugEnv(DummySandboxTaskSet(), debug_step="none")
    try:
        env.sandbox_client.delete = AsyncMock(side_effect=RuntimeError("delete outage"))
        env.register_sandbox("sb-debug")
        state = {"error": None, "sandbox_id": "sb-debug"}

        await env.destroy_sandbox(state)

        env.sandbox_client.delete.assert_awaited_once_with("sb-debug")
        assert isinstance(state["error"], vf.SandboxDeleteError)
        assert state["cleanup_errors"] == [
            {
                "type": "SandboxDeleteError",
                "message": "Failed to delete sandbox sb-debug: delete outage",
                "scope": "env_cleanup",
                "sandbox_id": "sb-debug",
            }
        ]
        assert "sb-debug" in env.active_sandboxes
    finally:
        env.active_sandboxes.clear()
        env.teardown_sandbox_client()


@pytest.mark.asyncio
async def test_remote_hybrid_math_cleanup_delete_failure_records_rollout_error() -> (
    None
):
    rubric = RemoteHybridMathRubric()
    try:
        rubric.sandbox_client.delete = AsyncMock(
            side_effect=RuntimeError("delete outage")
        )
        rubric.register_sandbox("sb-math")
        state = {"error": None, "sandbox_id": "sb-math"}

        await rubric.cleanup(state)

        rubric.sandbox_client.delete.assert_awaited_once_with("sb-math")
        assert isinstance(state["error"], vf.SandboxDeleteError)
        assert state["cleanup_errors"] == [
            {
                "type": "SandboxDeleteError",
                "message": "Failed to delete sandbox sb-math: delete outage",
                "scope": "rubric_cleanup",
                "sandbox_id": "sb-math",
            }
        ]
        assert "sb-math" in rubric.active_sandboxes
    finally:
        rubric.active_sandboxes.clear()
        rubric.teardown_sandbox_client()
