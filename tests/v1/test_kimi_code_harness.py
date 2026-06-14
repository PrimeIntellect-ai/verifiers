import json
from unittest.mock import AsyncMock

import pytest

import verifiers.v1 as vf
from harnesses.kimi_code import KimiCodeHarness, KimiCodeHarnessConfig


def test_kimi_code_config_resolves_from_harness_id() -> None:
    assert vf.harness_config_type("kimi-code") is KimiCodeHarnessConfig


@pytest.mark.asyncio
async def test_kimi_code_launch_configures_model_tools_and_runtime() -> None:
    harness = KimiCodeHarness(
        KimiCodeHarnessConfig(env={"CUSTOM": "value", "KIMI_MODEL_NAME": "wrong"})
    )
    runtime = AsyncMock(spec=vf.Runtime)
    runtime.run.side_effect = [vf.ProgramResult(0, "", ""), vf.ProgramResult(0, "", "")]
    trace = vf.Trace(
        task=vf.Task(
            idx=0,
            system_prompt="Follow the project instructions.",
            instruction="Fix the tests.",
        )
    )
    ctx = vf.RolloutContext(
        client=None,
        model="moonshotai/kimi-k2",
        sampling=vf.SamplingConfig(),
    )

    await harness.launch(
        ctx,
        trace,
        runtime,
        "http://127.0.0.1:8000/v1",
        "secret",
        {"docs": "http://127.0.0.1:9000/mcp"},
    )

    argv, env = runtime.run.await_args_list[1].args
    assert argv == [
        "/tmp/vf-kimi-code/bin/kimi",
        "--prompt",
        "Follow the project instructions.\n\nFix the tests.",
    ]
    assert env == {
        "CUSTOM": "value",
        "KIMI_CODE_HOME": ".vf-kimi-code",
        "KIMI_MODEL_NAME": "moonshotai/kimi-k2",
        "KIMI_MODEL_API_KEY": "secret",
        "KIMI_MODEL_PROVIDER_TYPE": "openai",
        "KIMI_MODEL_BASE_URL": "http://127.0.0.1:8000/v1",
        "KIMI_MODEL_CAPABILITIES": "tool_use",
        "KIMI_DISABLE_TELEMETRY": "1",
        "KIMI_CODE_NO_AUTO_UPDATE": "1",
    }
    path, data = runtime.write.await_args.args
    assert path == ".vf-kimi-code/mcp.json"
    assert json.loads(data) == {
        "mcpServers": {"docs": {"url": "http://127.0.0.1:9000/mcp"}}
    }


@pytest.mark.asyncio
async def test_kimi_code_install_failure_is_actionable() -> None:
    harness = KimiCodeHarness(KimiCodeHarnessConfig())
    runtime = AsyncMock(spec=vf.Runtime)
    runtime.run.return_value = vf.ProgramResult(1, "", "download failed")
    trace = vf.Trace(task=vf.Task(idx=0, instruction="Do the task."))
    ctx = vf.RolloutContext(
        client=None, model="test-model", sampling=vf.SamplingConfig()
    )

    with pytest.raises(
        vf.ProgramError, match="Kimi Code install failed: download failed"
    ):
        await harness.launch(ctx, trace, runtime, "http://localhost/v1", "secret", {})
