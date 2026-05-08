import importlib
import inspect
import sys
import types
from pathlib import Path

import pytest

import verifiers as vf


def load_module(monkeypatch: pytest.MonkeyPatch):
    env_dir = (
        Path(__file__).parents[1] / "environments" / "langchain_deep_agents_wikispeedia"
    )
    monkeypatch.syspath_prepend(str(env_dir))
    sys.modules.pop("langchain_deep_agents_wikispeedia", None)
    sys.modules.pop("wiki_graph", None)
    return importlib.import_module("langchain_deep_agents_wikispeedia")


class FakeWiki:
    articles = {"A": "Article A", "B": "Article B"}
    links = {"A": ["B"], "B": []}
    distances = {"A": {"B": 1}}

    def get_text(self, article: str) -> str:
        return self.articles[article]

    def get_links(self, article: str) -> list[str]:
        return self.links[article]

    def get_human_stats(self, source: str, target: str):
        return None


def make_small_wiki(module):
    articles = {
        "A": "Article A",
        "B": "Article B",
        "C": "Article C",
        "D": "Article D",
    }
    links = {
        source: [target for target in articles if target != source]
        for source in articles
    }
    distances = {
        source: {target: 1 for target in articles if target != source}
        for source in articles
    }
    return module.WikiGraph(articles=articles, links=links, distances=distances)


def test_wikispeedia_loads_as_v1_taskset_harness(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = load_module(monkeypatch)

    env = module.load_environment(train_size=1, eval_size=1)

    assert isinstance(env, vf.Env)
    assert isinstance(env.taskset, vf.Taskset)
    assert isinstance(env.harness, vf.Harness)
    assert env.taskset.taskset_id == "langchain-deep-agents-wikispeedia"


def test_wikispeedia_rows_use_v1_task_shape(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = load_module(monkeypatch)
    dataset = module.build_dataset(
        FakeWiki(),
        [("A", "B", 1)],
        links_only=False,
        max_turns=7,
    )
    row = dataset[0]

    assert "task" not in row
    assert row["task_id"] == "A->B"
    assert row["max_turns"] == 7
    assert row["info"] == {"source": "A", "target": "B", "shortest_path": 1}


def test_wikispeedia_taskset_sources_use_disjoint_target_split(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = load_module(monkeypatch)
    wiki = make_small_wiki(module)
    monkeypatch.setattr(module, "load_wiki_graph", lambda cache_dir=None: wiki)
    taskset = module.load_taskset(
        train_size=2,
        eval_size=1,
        min_path_length=1,
        max_path_length=1,
        eval_target_fraction=0.5,
    )

    train_rows = list(taskset.source())
    eval_rows = list(taskset.eval_source())

    assert len(train_rows) == 2
    assert len(eval_rows) == 1
    assert {row["answer"] for row in train_rows}.isdisjoint(
        {row["answer"] for row in eval_rows}
    )


def test_wikispeedia_efficiency_weight_uses_fresh_reward_wrapper(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = load_module(monkeypatch)
    wiki = make_small_wiki(module)
    monkeypatch.setattr(module, "load_wiki_graph", lambda cache_dir=None: wiki)

    weighted = module.load_taskset(efficiency_weight=0.5)
    plain = module.load_taskset(efficiency_weight=0.0)

    assert any(fn.__name__ == "path_efficiency" for fn in weighted.rewards)
    assert any(fn is module.path_efficiency for fn in plain.metrics)
    assert not getattr(module.path_efficiency, "reward", False)


def test_wikispeedia_taskset_owns_navigation_tools(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = load_module(monkeypatch)

    taskset = module.load_taskset(allow_go_back=True)
    names = [tool.__name__ for tool in taskset.toolsets[0].tools]
    no_back = module.load_taskset(allow_go_back=False)

    assert names == ["click_link", "go_back"]
    assert [tool.__name__ for tool in no_back.toolsets[0].tools] == ["click_link"]
    assert module.load_harness().toolsets == []


def test_wikispeedia_system_prompt_matches_available_tools(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = load_module(monkeypatch)

    with_back = module.load_taskset(allow_go_back=True)
    without_back = module.load_taskset(allow_go_back=False)

    assert "go_back" in with_back.system_prompt[0]["content"]
    assert "go_back" not in without_back.system_prompt[0]["content"]
    assert "Backtracking is disabled" in without_back.system_prompt[0]["content"]


@pytest.mark.asyncio
async def test_wikispeedia_tools_resolve_through_v1_runtime(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = load_module(monkeypatch)
    wiki = make_small_wiki(module)
    monkeypatch.setattr(module, "load_wiki_graph", lambda cache_dir=None: wiki)
    env = module.load_environment(
        train_size=2,
        eval_size=1,
        min_path_length=1,
        max_path_length=1,
    )
    task = module.vf.Task(list(env.taskset.source())[0]).freeze()
    state = module.vf.State.for_task(task)
    state = await env.harness.setup_state(task, state)

    tools = state.get_tools()
    state["current_article"] = state["info"]["source"]
    state["path"] = [state["info"]["source"]]
    state["reached_target"] = False
    state["links_only"] = False

    result = await tools["click_link"](article=state["info"]["target"])

    assert sorted(tools) == ["click_link", "go_back"]
    assert result.startswith("TARGET REACHED")
    assert state["reached_target"] is True


@pytest.mark.asyncio
async def test_wikispeedia_langchain_tools_keep_explicit_schema(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = load_module(monkeypatch)
    fake_langchain_core = types.ModuleType("langchain_core")
    fake_tools_module = types.ModuleType("langchain_core.tools")

    def tool(func):
        return func

    fake_tools_module.tool = tool
    fake_langchain_core.tools = fake_tools_module
    monkeypatch.setitem(sys.modules, "langchain_core", fake_langchain_core)
    monkeypatch.setitem(sys.modules, "langchain_core.tools", fake_tools_module)
    calls = []

    async def runtime_click_link(**kwargs):
        calls.append(kwargs)
        return "clicked"

    async def runtime_go_back():
        return "back"

    tools = module.langchain_navigation_tools(
        {"click_link": runtime_click_link, "go_back": runtime_go_back}
    )

    assert [tool.__name__ for tool in tools] == ["click_link", "go_back"]
    assert list(inspect.signature(tools[0]).parameters) == ["article"]
    assert list(inspect.signature(tools[1]).parameters) == []
    assert await tools[0](article="B") == "clicked"
    assert calls == [{"article": "B"}]


@pytest.mark.asyncio
async def test_wikispeedia_graph_recursion_limit_stops_rollout(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = load_module(monkeypatch)

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
            raise GraphRecursionError("recursion limit")

    fake_deepagents = types.ModuleType("deepagents")
    fake_langchain_openai = types.ModuleType("langchain_openai")
    fake_langgraph = types.ModuleType("langgraph")
    fake_langgraph_errors = types.ModuleType("langgraph.errors")
    fake_langchain_core = types.ModuleType("langchain_core")
    fake_tools_module = types.ModuleType("langchain_core.tools")

    fake_deepagents.create_deep_agent = lambda **kwargs: FakeAgent()
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

    program = module.make_langchain_deep_agents_program(
        max_turns=50,
        timeout_seconds=30,
    )
    state = FakeState(
        {
            "info": {"source": "A"},
            "prompt": [{"role": "user", "content": "start"}],
            "system_prompt": [{"role": "system", "content": "prompt"}],
        }
    )

    result = await program({}, state)

    assert result["agent_timeout"] is True
    assert result["stop_reason"] == "agent_recursion_limit"
    assert result["agent_completion"] == []


@pytest.mark.asyncio
async def test_wikispeedia_tool_metrics_use_agent_completion(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = load_module(monkeypatch)
    task = vf.Task({"prompt": [], "info": {"shortest_path": 1}}).freeze()
    state = vf.State.for_task(task)
    state["completion"] = [
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [{"id": "call_1", "name": "click_link"}],
        },
        {
            "role": "tool",
            "name": "click_link",
            "tool_call_id": "call_1",
            "content": "'C' is not a valid link from 'A'.",
        },
    ]

    assert await module.total_tool_calls(task, state) == 1.0
    assert await module.invalid_link_rate(task, state) == 1.0
