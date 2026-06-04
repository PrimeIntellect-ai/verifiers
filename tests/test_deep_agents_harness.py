import asyncio
import sys
import types
import uuid

import pytest

from harnesses.deep_agents import run_deep_agent
from harnesses.utils.deep_agents_utils import langchain_tools_from_state


class FakeStructuredTool:
    def __init__(self, name, description, args_schema, coroutine):
        self.name = name
        self.description = description
        self.args_schema = args_schema
        self.coroutine = coroutine

    @classmethod
    def from_function(
        cls, coroutine=None, name=None, description=None, args_schema=None, **kwargs
    ):
        return cls(
            name=name,
            description=description,
            args_schema=args_schema,
            coroutine=coroutine,
        )


class FakeToolDef:
    def __init__(self, name, description, parameters):
        self.name = name
        self.description = description
        self.parameters = parameters


def install_fake_langchain_tools(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_langchain_core = types.ModuleType("langchain_core")
    fake_tools = types.ModuleType("langchain_core.tools")
    fake_tools.StructuredTool = FakeStructuredTool
    fake_langchain_core.tools = fake_tools
    monkeypatch.setitem(sys.modules, "langchain_core", fake_langchain_core)
    monkeypatch.setitem(sys.modules, "langchain_core.tools", fake_tools)


@pytest.mark.asyncio
async def test_langchain_tools_from_state_builds_structured_tools(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    install_fake_langchain_tools(monkeypatch)
    calls: list[dict[str, object]] = []

    async def click_link(**kwargs):
        calls.append(kwargs)
        return "clicked"

    async def go_back(**kwargs):
        return "back"

    class FakeState(dict):
        def get_tools(self):
            return {"click_link": click_link, "go_back": go_back}

    class FakeRuntime:
        def tool_defs(self, state):
            return [
                FakeToolDef(
                    "click_link",
                    "Navigate to a linked Wikipedia article.",
                    {
                        "type": "object",
                        "properties": {"article": {"type": "string"}},
                        "required": ["article"],
                    },
                ),
                FakeToolDef(
                    "go_back",
                    "Undo the last click_link.",
                    {"type": "object", "properties": {}},
                ),
            ]

    tools = langchain_tools_from_state(FakeState(), FakeRuntime())

    assert [tool.name for tool in tools] == ["click_link", "go_back"]
    assert tools[0].description == "Navigate to a linked Wikipedia article."
    assert tools[0].args_schema == {
        "type": "object",
        "properties": {"article": {"type": "string"}},
        "required": ["article"],
    }
    assert tools[1].args_schema == {"type": "object", "properties": {}}
    assert await tools[0].coroutine(article="B") == "clicked"
    assert calls == [{"article": "B"}]
    assert await tools[1].coroutine() == "back"


@pytest.mark.asyncio
async def test_langchain_tools_from_state_skips_unbacked_defs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    install_fake_langchain_tools(monkeypatch)

    async def click_link(**kwargs):
        return "clicked"

    class FakeState(dict):
        def get_tools(self):
            return {"click_link": click_link}

    class FakeRuntime:
        def tool_defs(self, state):
            return [
                FakeToolDef("click_link", "", {"type": "object", "properties": {}}),
                FakeToolDef("missing", "", {"type": "object", "properties": {}}),
            ]

    tools = langchain_tools_from_state(FakeState(), FakeRuntime())

    assert [tool.name for tool in tools] == ["click_link"]
    assert tools[0].description == "Call the click_link tool."


@pytest.mark.asyncio
async def test_langchain_tools_from_state_real_structured_tool() -> None:
    pytest.importorskip("langchain_core")
    from langchain_core.tools import StructuredTool
    from langchain_core.utils.function_calling import convert_to_openai_tool

    calls: list[dict[str, object]] = []

    async def click_link(**kwargs):
        calls.append(kwargs)
        return "clicked"

    async def go_back(**kwargs):
        return "back"

    class RealState(dict):
        def get_tools(self):
            return {"click_link": click_link, "go_back": go_back}

    class RealRuntime:
        def tool_defs(self, state):
            return [
                FakeToolDef(
                    "click_link",
                    "Navigate to a linked Wikipedia article.",
                    {
                        "type": "object",
                        "properties": {"article": {"type": "string"}},
                        "required": ["article"],
                    },
                ),
                FakeToolDef(
                    "go_back",
                    "Undo the last click_link.",
                    {"type": "object", "properties": {}},
                ),
            ]

    tools = langchain_tools_from_state(RealState(), RealRuntime())

    assert all(isinstance(tool, StructuredTool) for tool in tools)
    assert [tool.name for tool in tools] == ["click_link", "go_back"]
    schema = convert_to_openai_tool(tools[0])["function"]
    assert schema["name"] == "click_link"
    assert schema["description"] == "Navigate to a linked Wikipedia article."
    assert schema["parameters"]["properties"] == {"article": {"type": "string"}}
    assert schema["parameters"].get("required") == ["article"]
    assert await tools[0].ainvoke({"article": "B"}) == "clicked"
    assert calls == [{"article": "B"}]
    assert await tools[1].ainvoke({}) == "back"


class FakeEndpointConfig:
    model = "model"
    base_url = "https://example.invalid/v1"


class FakeClient:
    api_key = "key"

    def close(self) -> None:
        return None


class FakeState(dict):
    def get_endpoint_config(self, api: str):
        _ = api
        return FakeEndpointConfig()

    def get_client(self, api: str, *, sync: bool = False):
        _ = api, sync
        return FakeClient()

    def get_tools(self):
        return {}

    def get_max_turns(self, default: int):
        return default

    def stop(self, reason: str):
        self["stop_reason"] = reason


class FakeConfig:
    def __init__(self, agent_name: str, timeout_seconds: float, max_turns: int) -> None:
        self.agent_name = agent_name
        self.timeout_seconds = timeout_seconds
        self.max_turns = max_turns


class FakeRuntime:
    def tool_defs(self, state):
        return None


class FakeHarness:
    def __init__(self, config: FakeConfig) -> None:
        self.config = config
        self.runtime = FakeRuntime()


def install_fake_deepagents_stack(
    monkeypatch: pytest.MonkeyPatch,
    *,
    create_deep_agent,
    graph_recursion_error: type[Exception],
) -> None:
    install_fake_langchain_tools(monkeypatch)

    class FakeChatOpenAI:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    fake_deepagents = types.ModuleType("deepagents")
    fake_langchain_openai = types.ModuleType("langchain_openai")
    fake_langgraph = types.ModuleType("langgraph")
    fake_langgraph_errors = types.ModuleType("langgraph.errors")

    fake_deepagents.create_deep_agent = create_deep_agent
    fake_langchain_openai.ChatOpenAI = FakeChatOpenAI
    fake_langgraph_errors.GraphRecursionError = graph_recursion_error
    fake_langgraph.errors = fake_langgraph_errors
    monkeypatch.setitem(sys.modules, "deepagents", fake_deepagents)
    monkeypatch.setitem(sys.modules, "langchain_openai", fake_langchain_openai)
    monkeypatch.setitem(sys.modules, "langgraph", fake_langgraph)
    monkeypatch.setitem(sys.modules, "langgraph.errors", fake_langgraph_errors)


@pytest.mark.asyncio
async def test_run_deep_agent_recursion_limit_stops_rollout(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class GraphRecursionError(Exception):
        pass

    class FakeAgent:
        async def ainvoke(self, payload, config=None):
            raise GraphRecursionError("recursion limit")

    created_system_prompts: list[str] = []

    def fake_create_deep_agent(**kwargs):
        created_system_prompts.append(kwargs["system_prompt"])
        return FakeAgent()

    install_fake_deepagents_stack(
        monkeypatch,
        create_deep_agent=fake_create_deep_agent,
        graph_recursion_error=GraphRecursionError,
    )

    harness = FakeHarness(
        FakeConfig(agent_name="deep-agent", timeout_seconds=30, max_turns=50)
    )
    state = FakeState(
        {
            "trajectory_id": "0123456789abcdef0123456789abcdef",
            "info": {"source": "A", "target": "B", "shortest_path": 1},
            "prompt": [{"role": "user", "content": "start"}],
            "system_prompt": [
                {"role": "user", "content": "first prompt chunk"},
                {"role": "system", "content": "second prompt chunk"},
            ],
        }
    )

    result = await run_deep_agent({}, state, harness)

    assert created_system_prompts == ["first prompt chunk\n\nsecond prompt chunk"]
    assert result["agent_recursion_limit"] is True
    assert "agent_timeout" not in result
    assert result["stop_reason"] == "agent_recursion_limit"
    assert result["agent_completion"] == []


@pytest.mark.asyncio
async def test_run_deep_agent_timeout_sets_timeout_flag(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class GraphRecursionError(Exception):
        pass

    class FakeAgent:
        async def ainvoke(self, payload, config=None):
            await asyncio.sleep(1)
            return {"messages": []}

    def fake_create_deep_agent(**kwargs):
        return FakeAgent()

    install_fake_deepagents_stack(
        monkeypatch,
        create_deep_agent=fake_create_deep_agent,
        graph_recursion_error=GraphRecursionError,
    )

    harness = FakeHarness(
        FakeConfig(agent_name="deep-agent", timeout_seconds=0.01, max_turns=50)
    )
    state = FakeState(
        {
            "trajectory_id": "0123456789abcdef0123456789abcdef",
            "info": {"source": "A", "target": "B", "shortest_path": 1},
            "prompt": [{"role": "user", "content": "start"}],
        }
    )

    result = await run_deep_agent({}, state, harness)

    assert result["agent_timeout"] is True
    assert "agent_recursion_limit" not in result
    assert result["stop_reason"] == "agent_timeout"
    assert result["agent_completion"] == []


@pytest.mark.asyncio
async def test_run_deep_agent_emits_generic_tracing_config(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class GraphRecursionError(Exception):
        pass

    captured: dict[str, object] = {}
    created: dict[str, object] = {}

    class FakeAgent:
        async def ainvoke(self, payload, config=None):
            captured["payload"] = payload
            captured["config"] = config
            return {"messages": [{"role": "assistant", "content": "done"}]}

    def fake_create_deep_agent(**kwargs):
        created.update(kwargs)
        return FakeAgent()

    install_fake_deepagents_stack(
        monkeypatch,
        create_deep_agent=fake_create_deep_agent,
        graph_recursion_error=GraphRecursionError,
    )

    trajectory_id = "0123456789abcdef0123456789abcdef"
    run_id = uuid.UUID(hex=trajectory_id)
    harness = FakeHarness(
        FakeConfig(agent_name="wikispeedia-navigator", timeout_seconds=30, max_turns=12)
    )
    state = FakeState(
        {
            "trajectory_id": trajectory_id,
            "runtime": {"group_key": "group-1"},
            "prompt": [{"role": "user", "content": "start"}],
        }
    )
    task = {
        "task_id": "A->B",
        "taskset_id": "langchain-deep-agents-wikispeedia",
    }

    result = await run_deep_agent(task, state, harness)

    assert created["name"] == "wikispeedia-navigator"
    assert captured["payload"] == {"messages": [{"role": "user", "content": "start"}]}
    assert captured["config"] == {
        "run_name": "langchain-deep-agents-wikispeedia:A->B",
        "run_id": run_id,
        "configurable": {"thread_id": trajectory_id},
        "metadata": {
            "vf_env": "langchain-deep-agents-wikispeedia",
            "vf_task_id": "A->B",
            "vf_trajectory_id": trajectory_id,
            "vf_group_key": "group-1",
        },
        "tags": ["verifiers", "vf-v1", "langchain-deep-agents-wikispeedia"],
        "recursion_limit": 12,
    }
    assert result["langsmith_run_id"] == str(run_id)
    assert result["completion"] == [{"role": "assistant", "content": "done"}]
