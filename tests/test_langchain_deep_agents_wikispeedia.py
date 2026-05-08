import importlib
import sys
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
