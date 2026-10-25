"""mini-swe-agent's per-command and per-model-call timeouts come from the harness config."""

import json

from verifiers.v1.harnesses.mini_swe_agent.harness import MiniSWEAgentHarnessConfig


def test_the_command_timeout_defaults_to_a_day_and_can_be_set():
    assert MiniSWEAgentHarnessConfig(id="mini_swe_agent").command_timeout == 86400
    assert (
        MiniSWEAgentHarnessConfig(
            id="mini_swe_agent", command_timeout=1800
        ).command_timeout
        == 1800
    )


async def test_the_command_timeout_reaches_mini_swe_agents_environment(monkeypatch):
    from types import SimpleNamespace

    from verifiers.v1.harnesses.mini_swe_agent import harness as module

    written = {}

    class Runtime:
        async def write(self, path, data):
            written["args"] = json.loads(data)

        async def run(self, argv, env):
            return SimpleNamespace(exit_code=0, stdout="", stderr="")

        async def prepare_uv_script(self, source, env):
            return ["python", "program.py"]

    harness = module.MiniSWEAgentHarness.__new__(module.MiniSWEAgentHarness)
    harness.config = MiniSWEAgentHarnessConfig(
        id="mini_swe_agent", command_timeout=1800, request_timeout=7200
    )
    monkeypatch.setattr(
        harness, "resolve_text_prompt", lambda data: (None, "do it"), raising=False
    )
    await harness.launch(
        SimpleNamespace(model="m"),
        SimpleNamespace(id="t"),
        Runtime(),
        "http://x",
        "k",
        {},
        None,
    )
    assert "environment.timeout=1800" in written["args"]
    # LiteLLM's own 600 s default failed long model turns with litellm.Timeout
    assert "model.model_kwargs.timeout=7200" in written["args"]


def test_the_model_request_timeout_defaults_to_a_day():
    assert MiniSWEAgentHarnessConfig(id="mini_swe_agent").request_timeout == 86400
