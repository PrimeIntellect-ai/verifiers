"""Codex sends its native `web_search` tool only when asked: self-hosted providers
(vLLM, Dynamo) do not serve it, and Dynamo refuses a request that carries it."""

import tomllib

import pytest
from test_prompt_transport import RecordingRuntime

import verifiers.v1 as vf
from verifiers.v1.clients import EvalClientConfig, ModelContext
from verifiers.v1.harnesses import CodexHarness, CodexHarnessConfig


@pytest.mark.parametrize(
    "config,expected",
    [({}, "disabled"), ({"web_search": "live"}, "live")],
    ids=["default", "live"],
)
async def test_codex_web_search_is_off_unless_asked(config, expected):
    harness = CodexHarness(CodexHarnessConfig(**config))
    runtime = RecordingRuntime()
    data = vf.TaskData(idx=0, prompt="task")
    trace = vf.Trace(
        agent=vf.AgentInfo(config=vf.AgentConfig()),
        task=vf.TraceTask(type="Task", data=data),
    )
    ctx = ModelContext(model="openai/test-model", client=EvalClientConfig())
    await harness.build_env(ctx, trace, runtime, "http://127.0.0.1:1/v1", "secret", {})
    written = runtime.files[f"{harness.trace_home(trace)}/config.toml"]
    assert tomllib.loads(written.decode())["web_search"] == expected
