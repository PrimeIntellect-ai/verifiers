import importlib
import sys
import types
import uuid
from pathlib import Path

import pytest

import verifiers as vf


def load_lab2_module(monkeypatch: pytest.MonkeyPatch):
    env_dir = (
        Path(__file__).parents[1]
        / "lab2"
        / "environments"
        / "langchain_deep_agents_wikispeedia"
    )
    monkeypatch.syspath_prepend(str(env_dir))
    sys.modules.pop("langchain_deep_agents_wikispeedia", None)
    sys.modules.pop("wiki_graph", None)
    return importlib.import_module("langchain_deep_agents_wikispeedia")


@pytest.mark.asyncio
async def test_lab2_deep_agents_program_passes_langsmith_config(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = load_lab2_module(monkeypatch)

    class GraphRecursionError(Exception):
        pass

    class FakeState(dict):
        def get_endpoint_config(self, api: str):
            return {
                "model": "model",
                "api_base": "https://example.invalid/v1",
                "api_key": "key",
            }

        def get_tools(self):
            return {}

        def get_max_turns(self, default: int):
            return default

        def stop(self, reason: str):
            self["stop_reason"] = reason

    class FakeChatOpenAI:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    class FakeAgent:
        async def ainvoke(self, payload, config=None):
            captured["payload"] = payload
            captured["config"] = config
            return {"messages": [{"role": "assistant", "content": "done"}]}

    captured: dict[str, object] = {}
    created: dict[str, object] = {}

    def fake_create_deep_agent(**kwargs):
        created.update(kwargs)
        return FakeAgent()

    fake_deepagents = types.ModuleType("deepagents")
    fake_langchain_openai = types.ModuleType("langchain_openai")
    fake_langgraph = types.ModuleType("langgraph")
    fake_langgraph_errors = types.ModuleType("langgraph.errors")
    fake_langchain_core = types.ModuleType("langchain_core")
    fake_tools_module = types.ModuleType("langchain_core.tools")

    fake_deepagents.create_deep_agent = fake_create_deep_agent
    fake_langchain_openai.ChatOpenAI = FakeChatOpenAI
    fake_langgraph_errors.GraphRecursionError = GraphRecursionError
    fake_langgraph.errors = fake_langgraph_errors
    fake_tools_module.tool = lambda func: func
    fake_langchain_core.tools = fake_tools_module
    monkeypatch.setitem(sys.modules, "deepagents", fake_deepagents)
    monkeypatch.setitem(sys.modules, "langchain_openai", fake_langchain_openai)
    monkeypatch.setitem(sys.modules, "langgraph", fake_langgraph)
    monkeypatch.setitem(sys.modules, "langgraph.errors", fake_langgraph_errors)
    monkeypatch.setitem(sys.modules, "langchain_core", fake_langchain_core)
    monkeypatch.setitem(sys.modules, "langchain_core.tools", fake_tools_module)

    trajectory_id = "0123456789abcdef0123456789abcdef"
    run_id = uuid.UUID(hex=trajectory_id)
    program = module.make_langchain_deep_agents_program(
        max_turns=12,
        timeout_seconds=30,
    )
    state = FakeState(
        {
            "trajectory_id": trajectory_id,
            "runtime": {"group_key": "group-1"},
            "info": {"source": "A", "target": "B", "shortest_path": 2},
            "prompt": [{"role": "user", "content": "start"}],
        }
    )

    result = await program({"task_id": "A->B"}, state)

    assert created["name"] == "wikispeedia-navigator"
    assert captured["payload"] == {"messages": [{"role": "user", "content": "start"}]}
    assert captured["config"] == {
        "run_name": "wikispeedia:A->B",
        "run_id": run_id,
        "configurable": {"thread_id": trajectory_id},
        "metadata": {
            "vf_env": "langchain-deep-agents-wikispeedia",
            "vf_task_id": "A->B",
            "vf_trajectory_id": trajectory_id,
            "vf_group_key": "group-1",
            "source": "A",
            "target": "B",
            "shortest_path": 2,
        },
        "tags": ["verifiers", "vf-v1", "langchain-deep-agents-wikispeedia"],
        "recursion_limit": 12,
    }
    assert result["langsmith_run_id"] == str(run_id)
    assert result["completion"] == [{"role": "assistant", "content": "done"}]


@pytest.mark.asyncio
async def test_lab2_navigation_metrics_use_state_log_when_completion_empty(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = load_lab2_module(monkeypatch)
    task = vf.Task({"prompt": [], "info": {"shortest_path": 1}}).freeze()
    state = vf.State.for_task(task)
    state["completion"] = []
    state[module.NAVIGATION_TOOL_CALLS_KEY] = [
        {"name": "click_link", "valid": False},
        {"name": "click_link", "valid": True},
        {"name": "go_back", "valid": True},
    ]

    assert await module.total_tool_calls(task, state) == 3.0
    assert await module.make_tool_count_metric("click_link")(task, state) == 2.0
    assert await module.make_tool_count_metric("go_back")(task, state) == 1.0
    assert await module.invalid_link_rate(task, state) == 0.5
