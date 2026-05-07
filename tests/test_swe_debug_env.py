from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

import verifiers as vf
from verifiers.envs.experimental.composable import (
    SandboxSpec,
    SandboxTaskSet,
    SWEDebugEnv,
)


class _MockTaskSet(SandboxTaskSet):
    def get_instruction(self, info):
        return f"debug {info.get('id', 0)}"

    def get_sandbox_spec(self, info):
        return SandboxSpec(image="python:3.11-slim", cpu_cores=2, memory_gb=2)

    def get_workdir(self, info):
        return "/testbed"

    def get_rubric(self):
        return vf.Rubric()

    def _calculate_reward(self, test_output, info):
        return 1.0 if test_output == "PASS" else 0.0


def _dataset():
    from datasets import Dataset

    return Dataset.from_dict({"info": [{"id": 0}], "answer": [""]})


def _result(exit_code=0, stdout="", stderr=""):
    return SimpleNamespace(exit_code=exit_code, stdout=stdout, stderr=stderr)


def _build_env(**kwargs):
    taskset = _MockTaskSet(dataset=_dataset(), name="mock")
    taskset.setup = AsyncMock()
    taskset._apply_gold_patch = AsyncMock()
    taskset._run_tests = AsyncMock(return_value="PASS")
    env = SWEDebugEnv(taskset=taskset, **kwargs)
    env.sandbox_client = SimpleNamespace(
        execute_command=AsyncMock(return_value=_result(stdout="ok")),
        upload_file=AsyncMock(),
        teardown=lambda: None,
    )

    async def create_sandbox(state, _request):
        state["sandbox_id"] = "sbx"
        await env.post_sandbox_setup(state)
        return "sbx"

    env.create_sandbox = AsyncMock(side_effect=create_sandbox)
    return env, taskset


@pytest.mark.asyncio
async def test_default_runs_setup_gold_patch_and_tests():
    env, taskset = _build_env()
    state = {"info": {"id": 0}, "answer": ""}

    await env.setup_state(state)

    taskset.setup.assert_awaited_once()
    taskset._apply_gold_patch.assert_awaited_once()
    taskset._run_tests.assert_awaited_once()
    assert state["reward"] == 1.0
    assert state["reason"] == "pass"
    assert state["debug_step"] == "gold_patch"


@pytest.mark.asyncio
async def test_can_skip_setup_and_tests_with_noop_step():
    env, taskset = _build_env(run_setup=False, debug_step="none", run_tests=False)
    state = {"info": {"id": 0}, "answer": ""}

    await env.setup_state(state)

    taskset.setup.assert_not_awaited()
    taskset._apply_gold_patch.assert_not_awaited()
    taskset._run_tests.assert_not_awaited()
    assert state["setup_s"] == 0.0
    assert state["reward"] == 1.0
    assert state["reason"] == "pass"


@pytest.mark.asyncio
async def test_command_step_records_output():
    env, taskset = _build_env(
        debug_step="command", debug_command="pwd", run_tests=False
    )
    state = {"info": {"id": 0}, "answer": ""}

    await env.setup_state(state)

    env.sandbox_client.execute_command.assert_awaited_once_with(
        "sbx", "pwd", working_dir="/testbed", timeout=900
    )
    taskset._run_tests.assert_not_awaited()
    assert state["debug_exit_code"] == 0
    assert state["debug_stdout_tail"] == "ok"
    assert state["reward"] == 1.0


@pytest.mark.asyncio
async def test_script_step_uploads_inline_script():
    env, _taskset = _build_env(
        debug_step="script", debug_script="echo hi", run_tests=False
    )
    state = {"info": {"id": 0}, "answer": ""}

    await env.setup_state(state)

    env.sandbox_client.upload_file.assert_awaited_once()
    env.sandbox_client.execute_command.assert_awaited_once()
    command = env.sandbox_client.execute_command.await_args.args[1]
    assert "chmod +x /tmp/swe_debug_script.sh" in command
    assert state["reward"] == 1.0


@pytest.mark.asyncio
async def test_failed_command_stops_before_tests():
    env, taskset = _build_env(
        debug_step="command", debug_command="false", run_tests=True
    )
    env.sandbox_client.execute_command = AsyncMock(
        return_value=_result(exit_code=1, stderr="nope")
    )
    state = {"info": {"id": 0}, "answer": ""}

    await env.setup_state(state)

    taskset._run_tests.assert_not_awaited()
    assert state["reward"] == 0.0
    assert state["reason"] == "debug_command_failed"
    assert state["debug_stderr_tail"] == "nope"
