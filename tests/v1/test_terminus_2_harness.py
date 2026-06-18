import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from harnesses.terminus_2 import Terminus2Harness, Terminus2HarnessConfig
from verifiers.v1.loaders import harness_class, harness_config_type
from verifiers.v1.runtimes import ProgramResult
from verifiers.v1.task import Task


def test_terminus_2_is_a_builtin_harness():
    assert harness_class("terminus-2") is Terminus2Harness
    assert harness_config_type("terminus-2") is Terminus2HarnessConfig


@pytest.mark.asyncio
async def test_terminus_2_launches_harbor_through_the_interception_endpoint():
    result = ProgramResult(exit_code=0, stdout="", stderr="")
    runtime = SimpleNamespace(
        run_uv_script=AsyncMock(return_value=result),
        run=AsyncMock(return_value=result),
    )
    harness = Terminus2Harness(
        Terminus2HarnessConfig(
            env={"EXTRA": "value", "OPENAI_API_KEY": "not-the-session-secret"}
        )
    )
    trace = SimpleNamespace(id="abc123", task=Task(idx=0, prompt="repair the project"))
    ctx = SimpleNamespace(model="provider/model")

    assert (
        await harness.launch(
            ctx,
            trace,
            runtime,
            "http://intercept/v1",
            "session-secret",
            {},
        )
        is result
    )

    script = runtime.run_uv_script.await_args.args[0]
    assert 'dependencies = ["harbor==0.14.0"]' in script
    assert runtime.run_uv_script.await_args.kwargs == {
        "args": ["provider/model", "repair the project"],
        "env": {
            "EXTRA": "value",
            "OPENAI_BASE_URL": "http://intercept/v1",
            "OPENAI_API_KEY": "session-secret",
            "TMUX_TMPDIR": "/tmp/vf-terminus-2-abc123",
        },
    }
    runtime.run.assert_awaited_once_with(
        [
            "sh",
            "-c",
            'tmux kill-server >/dev/null 2>&1 || true; rm -rf "$TMUX_TMPDIR"',
        ],
        {"TMUX_TMPDIR": "/tmp/vf-terminus-2-abc123"},
    )


@pytest.mark.asyncio
async def test_terminus_2_rejects_disabled_tools():
    harness = Terminus2Harness(Terminus2HarnessConfig(disabled_tools=["tmux"]))

    with pytest.raises(ValueError, match="does not support disabling tools"):
        await harness.launch(
            SimpleNamespace(model="provider/model"),
            SimpleNamespace(id="abc123", task=Task(idx=0, prompt="task")),
            SimpleNamespace(run_uv_script=AsyncMock(), run=AsyncMock()),
            "http://intercept/v1",
            "session-secret",
            {},
        )


@pytest.mark.asyncio
@pytest.mark.parametrize("error_type", [RuntimeError, asyncio.CancelledError])
async def test_terminus_2_cleans_up_after_program_failure(error_type):
    cleanup = ProgramResult(exit_code=0, stdout="", stderr="")
    runtime = SimpleNamespace(
        run_uv_script=AsyncMock(side_effect=error_type("program stopped")),
        run=AsyncMock(return_value=cleanup),
    )
    harness = Terminus2Harness(Terminus2HarnessConfig())

    with pytest.raises(error_type, match="program stopped"):
        await harness.launch(
            SimpleNamespace(model="provider/model"),
            SimpleNamespace(id="abc123", task=Task(idx=0, prompt="task")),
            runtime,
            "http://intercept/v1",
            "session-secret",
            {},
        )

    runtime.run.assert_awaited_once()


@pytest.mark.asyncio
async def test_terminus_2_cleanup_does_not_hide_program_failure():
    runtime = SimpleNamespace(
        run_uv_script=AsyncMock(side_effect=RuntimeError("program failed")),
        run=AsyncMock(side_effect=OSError("cleanup failed")),
    )
    harness = Terminus2Harness(Terminus2HarnessConfig())

    with pytest.raises(RuntimeError, match="program failed"):
        await harness.launch(
            SimpleNamespace(model="provider/model"),
            SimpleNamespace(id="abc123", task=Task(idx=0, prompt="task")),
            runtime,
            "http://intercept/v1",
            "session-secret",
            {},
        )
