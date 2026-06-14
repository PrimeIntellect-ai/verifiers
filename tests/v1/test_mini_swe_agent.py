from types import SimpleNamespace
from typing import cast
from unittest.mock import AsyncMock

import pytest
from harnesses import MiniSWEAgentHarness, MiniSWEAgentHarnessConfig

from verifiers.v1.clients import Client, RolloutContext
from verifiers.v1.loaders import harness_config_type, load_harness
from verifiers.v1.runtimes import ProgramResult, Runtime
from verifiers.v1.task import Task
from verifiers.v1.trace import Trace
from verifiers.v1.types import SamplingConfig


@pytest.mark.asyncio
async def test_mini_swe_agent_uses_native_parallel_tool_calls() -> None:
    run_uv_script = AsyncMock(return_value=ProgramResult(0, "", ""))
    runtime = cast(
        Runtime,
        SimpleNamespace(name="rollout-123", run_uv_script=run_uv_script),
    )
    harness = MiniSWEAgentHarness(
        MiniSWEAgentHarnessConfig(env={"EXTRA": "value", "OPENAI_API_KEY": "wrong"})
    )
    ctx = RolloutContext(
        client=cast(Client, object()),
        model="provider/model",
        sampling=SamplingConfig(),
    )
    trace = Trace(
        task=Task(
            idx=0,
            instruction="Fix the bug.",
            system_prompt="Use the repository tests.",
        )
    )

    result = await harness.launch(
        ctx,
        trace,
        runtime,
        "http://127.0.0.1:8000/v1",
        "secret",
        {},
    )

    assert result.exit_code == 0
    program = run_uv_script.await_args.args[0]
    args = run_uv_script.await_args.kwargs["args"]
    env = run_uv_script.await_args.kwargs["env"]
    assert 'dependencies = ["mini-swe-agent==2.2.8"]' in program
    assert args[:6] == [
        "--model",
        "provider/model",
        "--model-class",
        "litellm",
        "--task",
        "Fix the bug.",
    ]
    assert "mini" in args
    assert "model.model_kwargs.custom_llm_provider=openai" in args
    assert "model.model_kwargs.parallel_tool_calls=true" in args
    assert 'agent.system_template="Use the repository tests."' in args
    assert "mini_textbased" not in args
    assert "litellm_textbased" not in args
    assert env["EXTRA"] == "value"
    assert env["OPENAI_BASE_URL"] == "http://127.0.0.1:8000/v1"
    assert env["OPENAI_API_KEY"] == "secret"
    assert env["MSWEA_CONFIGURED"] == "true"


def test_mini_swe_agent_loads_by_id() -> None:
    config_type = harness_config_type("mini-swe-agent")
    config = config_type()

    assert config_type is MiniSWEAgentHarnessConfig
    assert isinstance(load_harness(config), MiniSWEAgentHarness)
    assert MiniSWEAgentHarness.SUPPORTS_TASK_TOOLS is False
