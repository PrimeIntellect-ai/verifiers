from unittest.mock import AsyncMock

import pytest

import verifiers as vf
from verifiers.envs.experimental.composable.tasksets.swe.swe_lego.taskset import (
    SWELegoRubric,
)


@pytest.mark.asyncio
async def test_swe_taskset_cleanup_delete_failure_surfaces_rollout_error() -> None:
    client = AsyncMock()
    client.delete.side_effect = RuntimeError("delete outage")
    rubric = SWELegoRubric(taskset=object())
    state = {"sandbox_client": client, "sandbox_id": "sb-swe"}

    await rubric.cleanup(state)

    assert isinstance(state["error"], vf.SandboxDeleteError)
    assert state["cleanup_errors"][0]["type"] == "SandboxDeleteError"
    client.delete.assert_awaited_once_with("sb-swe")


@pytest.mark.asyncio
async def test_cleanup_delete_failure_preserves_existing_rollout_error() -> None:
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
    assert state["cleanup_errors"][0]["type"] == "SandboxDeleteError"
