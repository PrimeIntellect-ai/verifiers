from collections.abc import Mapping
from typing import Any, cast

import pytest

import verifiers as vf
from verifiers.v1 import (
    add_advantage,
    add_metric,
    add_reward,
    build_signals,
    collect_signals,
    score_group,
    score_rollout,
)


@vf.metric
async def num_tool_calls(task: dict, state: dict) -> float:
    return float(len(state.get("tool_calls", [])))


@vf.metric
async def config_metric(task: dict, state: dict) -> float:
    return float(task["x"] + state["y"])


@vf.reward(weight=2.0)
async def exact_answer(task: dict, state: dict) -> float:
    return float(state.get("answer") == task["answer"])


@vf.reward(stage="group")
async def best_answer_bonus(tasks: list[dict], states: list[dict]) -> list[float]:
    return [
        float(state.get("answer") == task["answer"])
        for task, state in zip(tasks, states)
    ]


@vf.advantage
async def explicit_advantage(tasks: list[dict], states: list[dict]) -> list[float]:
    _ = tasks
    return [float(index) for index, _ in enumerate(states)]


@pytest.mark.asyncio
async def test_programmatic_metric_and_reward_share_signal_path() -> None:
    signals = build_signals()
    add_metric(signals, num_tool_calls)
    add_reward(signals, exact_answer)
    task = {"answer": "4"}
    state = {"answer": "4", "tool_calls": ["a", "b"]}

    await score_rollout(signals, task, state)

    metrics = cast(dict[str, float], state["metrics"])
    assert state["reward"] == 2.0
    assert metrics["exact_answer"] == 1.0
    assert metrics["num_tool_calls"] == 2.0


@pytest.mark.asyncio
async def test_config_overrides_default_signal_metadata() -> None:
    signals = build_signals(
        scoring={"exact_answer": {"weight": 0.5}},
        rewards=[exact_answer],
    )
    task = {"answer": "4"}
    state = {"answer": "4"}

    await score_rollout(signals, task, state)

    assert state["reward"] == 0.5


@pytest.mark.asyncio
async def test_config_tunes_imported_signal_by_name() -> None:
    signals = build_signals(
        metrics=[config_metric],
        scoring={"config_metric": {"priority": 10}},
    )
    task = {"x": 2}
    state = {"y": 3}

    await score_rollout(signals, task, state)

    metrics = cast(dict[str, float], state["metrics"])
    assert metrics["config_metric"] == 5.0


def test_signal_name_collisions_hard_fail() -> None:
    taskset_signals = build_signals(metrics=[num_tool_calls])
    harness_signals = build_signals(metrics=[num_tool_calls])

    with pytest.raises(ValueError, match="defined twice"):
        collect_signals(taskset_signals, harness_signals)


@pytest.mark.asyncio
async def test_group_signal_reports_unresolved_required_args() -> None:
    @vf.metric(stage="group")
    async def bad_group_metric(task: dict, state: dict) -> float:
        return 0.0

    signals = build_signals(metrics=[bad_group_metric])

    with pytest.raises(TypeError, match="metric signal 'bad_group_metric'.*task"):
        await score_group(signals, [{"answer": "a"}], [{"answer": "a"}])


@pytest.mark.asyncio
async def test_group_reward_scores_each_state() -> None:
    signals = build_signals(rewards=[best_answer_bonus])
    tasks: list[dict[str, Any]] = [{"answer": "a"}, {"answer": "b"}]
    states: list[dict[str, Any]] = [
        {
            "answer": "a",
            "trajectory": [{"advantage": None}, {"advantage": 9.0}],
        },
        {"answer": "c", "trajectory": [{"advantage": None}]},
    ]

    await score_group(signals, cast(list[Mapping[str, Any]], tasks), states)

    assert states[0]["reward"] == 1.0
    assert states[1]["reward"] == 0.0
    assert "advantage" not in states[0]
    assert "advantage" not in states[1]
    trajectory = cast(list[dict[str, Any]], states[0]["trajectory"])
    assert trajectory[0]["advantage"] is None
    assert trajectory[1]["advantage"] == 9.0


@pytest.mark.asyncio
async def test_advantage_signal_writes_group_advantages() -> None:
    signals = build_signals(rewards=[best_answer_bonus])
    add_advantage(signals, explicit_advantage)
    tasks: list[dict[str, Any]] = [{"answer": "a"}, {"answer": "b"}]
    states: list[dict[str, Any]] = [{"answer": "a"}, {"answer": "c"}]

    await score_group(signals, cast(list[Mapping[str, Any]], tasks), states)

    assert states[0]["advantage"] == 0.0
    assert states[1]["advantage"] == 1.0


def test_advantage_requires_group_plural_args() -> None:
    async def bad_advantage(task: dict, state: dict) -> float:
        return 0.0

    with pytest.raises(ValueError, match="stage='group'"):
        build_signals(advantages=[bad_advantage])
