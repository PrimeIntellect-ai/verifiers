from __future__ import annotations

from collections.abc import Mapping

import verifiers as vf


class Task(dict): ...


class State(dict): ...


@vf.reward(weight=1.0)
async def exact_answer(task: Mapping[str, object], state: dict[str, object]) -> float:
    return float(state["answer"] == task["answer"])


@vf.metric
async def num_tool_calls(task: Mapping[str, object], state: dict[str, object]) -> float:
    return float(len(state.get("tool_calls", [])))


async def execute(
    task: Mapping[str, object],
    state: dict[str, object],
) -> dict[str, object]: ...


def load_taskset(config=None) -> vf.Taskset:
    return vf.Taskset(
        source=lambda: ...,
        rewards=[exact_answer],
        config=config,
    )


def load_harness(config=None) -> vf.Harness:
    return vf.Harness(
        execute=execute,
        metrics=[num_tool_calls],
        config=config,
    )


def load_environment(config=None) -> vf.Environment:
    return vf.Env(
        taskset=load_taskset(config.taskset),
        harness=load_harness(config.harness),
    )


class Taskset:
    def __init__(
        self,
        source=None,
        metrics=None,
        rewards=None,
        cleanup=None,
        config=None,
    ):
        self.source = source
        self.metrics = metrics or []
        self.rewards = rewards or []
        self.cleanup = cleanup or []
        self.config = config

    def task(self, row) -> Task: ...

    async def init_group(
        self, task, num_rollouts
    ) -> tuple[list[Task], list[State]]: ...

    async def score_group(self, tasks, states): ...


class Harness:
    def __init__(
        self,
        execute=None,
        metrics=None,
        rewards=None,
        cleanup=None,
        config=None,
    ):
        self.execute = execute
        self.metrics = metrics or []
        self.rewards = rewards or []
        self.cleanup = cleanup or []
        self.config = config

    async def run(self, task, state): ...

    async def init_state(self, task, state): ...


class Env(vf.Environment):
    def __init__(self, taskset, harness, config=None):
        self.taskset = taskset
        self.harness = harness
        self.config = config

    async def run_group(self, task, num_rollouts, controls=None):
        tasks, states = await self.init_group(task, num_rollouts, controls)
        states = await self.run_rollouts(tasks, states, controls)
        states = await self.score_group(tasks, states, controls)
        return states

    async def init_group(self, task, num_rollouts, controls=None): ...

    async def run_rollouts(self, tasks, states, controls=None): ...

    async def rollout(self, task, state):
        state = await self.harness.run(task, state)
        return state

    async def score_group(self, tasks, states, controls=None): ...
