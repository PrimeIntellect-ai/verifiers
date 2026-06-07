import importlib
import sys
from pathlib import Path

import pytest

import verifiers.v1 as vf
from verifiers.v1.loaders import load_environment_from_components


def load_modules(monkeypatch: pytest.MonkeyPatch):
    env_dir = (
        Path(__file__).parents[1]
        / "environments"
        / "langchain_deep_agents_wikispeedia_v1"
    )
    monkeypatch.syspath_prepend(str(env_dir))
    for name in (
        "langchain_deep_agents_wikispeedia_v1",
        "langchain_deep_agents_wikispeedia_v1.taskset",
        "langchain_deep_agents_wikispeedia_v1.harness",
        "langchain_deep_agents_wikispeedia_v1.wiki_graph",
    ):
        sys.modules.pop(name, None)
    return (
        importlib.import_module("langchain_deep_agents_wikispeedia_v1"),
        importlib.import_module("langchain_deep_agents_wikispeedia_v1.taskset"),
    )


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
    package, _ = load_modules(monkeypatch)

    env = load_environment_from_components(package, {})

    assert isinstance(env, vf.Env)
    assert isinstance(env.taskset, vf.Taskset)
    assert isinstance(env.harness, vf.Harness)
    assert env.taskset.id == "langchain-deep-agents-wikispeedia"


def test_wikispeedia_env_config_reaches_taskset_and_harness(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    package, module = load_modules(monkeypatch)
    wiki = make_small_wiki(module)
    monkeypatch.setattr(module, "load_wiki_graph", lambda cache_dir=None: wiki)

    env = load_environment_from_components(
        package,
        {
            "config": {
                "taskset": {
                    "train_size": 2,
                    "eval_size": 1,
                    "min_path_length": 1,
                    "max_path_length": 1,
                    "eval_target_fraction": 0.5,
                    "allow_go_back": False,
                    "links_only": True,
                    "max_turns": 7,
                },
                "harness": {"max_turns": 8, "timeout_seconds": 9.0},
            }
        },
    )

    train_rows = list(env.taskset)
    eval_rows = [
        env.taskset.to_task(dict(row)) for row in env.taskset.get_eval_dataset()
    ]

    assert len(train_rows) == 2
    assert len(eval_rows) == 1
    assert train_rows[0].max_turns == 7
    assert train_rows[0].links_only is True
    assert train_rows[0].allow_go_back is False
    assert env.harness.config.max_turns == 8
    assert env.harness.config.timeout_seconds == 9.0


def test_wikispeedia_rows_use_v1_task_shape(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _, module = load_modules(monkeypatch)
    wiki = make_small_wiki(module)
    dataset = module.build_dataset(
        wiki,
        [("A", "B", 1)],
        cache_dir="/tmp/wiki",
        links_only=False,
        allow_go_back=True,
        max_turns=7,
    )
    row = dataset[0]

    assert "task" not in row
    assert row["task_id"] == "A->B"
    assert row["max_turns"] == 7
    assert row["cache_dir"] == "/tmp/wiki"
    assert row["source"] == "A"
    assert row["target"] == "B"
    assert row["shortest_path"] == 1


def test_wikispeedia_navigation_uses_state_extras(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _, module = load_modules(monkeypatch)
    wiki = make_small_wiki(module)
    task = module.WikispeediaTask(
        prompt=[{"role": "user", "content": "start"}],
        answer="B",
        source="A",
        target="B",
        shortest_path=1,
    )
    state = vf.State(task_id=task.task_id)

    module.init_navigation_state(task, state)
    result = module.click_link_result("B", wiki, state)

    assert result.startswith("TARGET REACHED")
    assert state.extras["current_article"] == "B"
    assert state.extras["path"] == ["A", "B"]
    assert state.extras["reached_target"] is True
    assert state.stop_condition == "target_reached"


@pytest.mark.asyncio
async def test_wikispeedia_scores_from_extras_and_transcript(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _, module = load_modules(monkeypatch)
    taskset = module.WikispeediaTaskset(
        module.WikispeediaTasksetConfig(efficiency_weight=0.5)
    )
    task = module.WikispeediaTask(
        prompt=[{"role": "user", "content": "start"}],
        answer="B",
        source="A",
        target="B",
        shortest_path=1,
    )
    state = vf.State(task_id=task.task_id)
    state.extras.update(
        {
            "path": ["A", "B"],
            "reached_target": True,
            "agent_timeout": False,
            "shortest_path": 1,
        }
    )
    state.add_turn(
        vf.Turn(
            prompt=[],
            completion=[vf.AssistantMessage(content="done")],
            tool_calls=[
                vf.ToolCall(id="call_1", name="click_link", arguments='{"article":"B"}')
            ],
        )
    )

    assert await taskset.reached_target(state) == 1.0
    assert await taskset.path_efficiency(state) == 1.0
    assert await taskset.path_efficiency_reward(state) == 0.5
    assert await taskset.total_tool_calls(state) == 1.0
