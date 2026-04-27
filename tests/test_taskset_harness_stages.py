import pytest

import verifiers as vf


class DoneHarness(vf.Harness):
    async def setup_state(self, task, resources):
        state = await super().setup_state(task, resources)
        state["done"] = True
        return state


@pytest.mark.asyncio
async def test_harness_rollout_render_metric_cleanup_order(mock_client):
    events: list[str] = []

    class HarnessWithSignals(DoneHarness):
        @vf.render
        async def render_value(self, task, state, resources):
            events.append("render")
            state["value"] = 2

        @vf.metric
        async def value_metric(self, task, state, resources):
            events.append("metric")
            return state["value"]

        @vf.cleanup
        async def cleanup_value(self, task, state, resources):
            events.append("cleanup")
            state["cleanup_saw_metric"] = "value_metric" in state["metrics"]

    harness = HarnessWithSignals()
    resources = vf.Resources(harness=harness)
    task = vf.Task(prompt=[vf.UserMessage(content="hello")])
    async with resources.rollout(task, mock_client, "test-model"):
        state = await harness.run(task, resources)

    assert events == ["render", "metric", "cleanup"]
    assert state["metrics"]["value_metric"] == 2.0
    assert state["reward"] == 0.0
    assert state["cleanup_saw_metric"] is True


@pytest.mark.asyncio
async def test_group_reward_runs_after_rollout_stage(mock_client):
    events: list[str] = []

    class GroupTaskset(vf.Taskset):
        @vf.reward(stage="group")
        async def group_bonus(self, tasks, states, resources):
            events.append("reward")
            return [float(index + 1) for index, _ in enumerate(states)]

        @vf.cleanup(stage="group")
        async def cleanup_group(self, tasks, states, resources):
            events.append("cleanup")
            for state in states:
                state["group_cleanup_saw_reward"] = state["reward"]

    taskset = GroupTaskset(source=[{"prompt": [vf.UserMessage(content="a")]}])
    harness = DoneHarness()
    env = vf.Env(taskset=taskset, harness=harness)
    tasks = [
        taskset.to_task({"prompt": [vf.UserMessage(content="a")], "example_id": 0}),
        taskset.to_task({"prompt": [vf.UserMessage(content="b")], "example_id": 1}),
    ]

    states = []
    for task in tasks:
        async with env.resources.rollout(task, mock_client, "test-model"):
            states.append(await harness.run(task, env.resources))

    assert events == []
    await env.rubric.score_group(states)

    assert events == ["reward", "cleanup"]
    assert [state["reward"] for state in states] == [1.0, 2.0]
    assert [state["metrics"]["group_bonus"] for state in states] == [1.0, 2.0]
    assert [state["advantage"] for state in states] == [-0.5, 0.5]
    assert [state["group_cleanup_saw_reward"] for state in states] == [1.0, 2.0]


def test_group_signals_require_explicit_group_stage():
    class BadHarness(vf.Harness):
        @vf.metric
        async def bad_metric(self, states, resources):
            return 0.0

    with pytest.raises(ValueError, match="stage='group'"):
        vf.Resources(harness=BadHarness())
