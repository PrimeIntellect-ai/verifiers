"""Prime Agent's generated model config: the system prompt's role over chat completions."""

import json
from types import SimpleNamespace

import pytest

import verifiers.v1 as vf
from verifiers.v1.clients import ModelContext
from verifiers.v1.configs.sampling import Sampling
from verifiers.v1.harnesses.prime_agent import (
    PrimeAgentHarness,
    PrimeAgentHarnessConfig,
)


class RecordingRuntime:
    def __init__(self) -> None:
        self.files: dict[str, bytes] = {}

    async def run(self, command, env):
        return SimpleNamespace(exit_code=0, stdout="", stderr="")

    async def write(self, path: str, data: bytes) -> None:
        self.files[path] = data


async def provider_config(**config) -> dict:
    harness = PrimeAgentHarness(PrimeAgentHarnessConfig(**config))
    runtime = RecordingRuntime()
    trace = vf.Trace(
        agent=vf.AgentInfo(config=vf.AgentConfig()),
        task=vf.TraceTask(type="Task", data=vf.TaskData(idx=0, prompt="hi")),
    )
    ctx = ModelContext(
        model="internal/glm-5.3-flash",
        client=vf.EvalClientConfig(),
        sampling=Sampling(reasoning_effort="low"),
    )
    await harness.prepare_acp(
        ctx, trace, runtime, "http://proxy.invalid/v1", "secret", {}, trace.task.data
    )
    (path,) = [path for path in runtime.files if path.endswith("/models.json")]
    return json.loads(runtime.files[path])["providers"]["intercept"]


async def test_default_leaves_role_detection_to_prime_agent():
    assert "compat" not in await provider_config()


@pytest.mark.parametrize("supported", [False, True])
async def test_developer_role_override(supported):
    provider = await provider_config(supports_developer_role=supported)
    assert provider["models"][0]["reasoning"] is True
    assert provider["compat"] == {"supportsDeveloperRole": supported}


async def test_override_applies_only_to_chat_completions():
    provider = await provider_config(
        supports_developer_role=False, dialect="anthropic_messages"
    )
    assert "compat" not in provider
