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


async def test_mcp_servers_are_listed_in_the_task_prompt_as_harbor_does(monkeypatch):
    # TB4's medical-claims-processing serves a Playwright MCP server; mini-swe-agent
    # has no MCP client, so Harbor's adapter names each server and URL in the task.
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

    assert module.MiniSWEAgentHarness.SUPPORTS_MCP
    assert module.MiniSWEAgentHarness.SUPPORTS_SKILLS
    harness = module.MiniSWEAgentHarness.__new__(module.MiniSWEAgentHarness)
    harness.config = MiniSWEAgentHarnessConfig(id="mini_swe_agent")
    monkeypatch.setattr(
        harness, "resolve_text_prompt", lambda data: (None, "do it"), raising=False
    )
    urls = {"playwright": "http://127.0.0.1:9000/mcp/playwright"}
    await harness.launch(
        SimpleNamespace(model="m"),
        SimpleNamespace(id="t"),
        Runtime(),
        "http://x",
        "k",
        urls,
        None,
    )
    task = written["args"][written["args"].index("--task") + 1]
    assert task == (
        "do it\n\nMCP Servers:\nThe following MCP servers are available for this "
        "task.\n- playwright: streamable-http transport, "
        "url: http://127.0.0.1:9000/mcp/playwright\n"
    )
    assert module.with_mcp_servers("do it", {}) == "do it"
