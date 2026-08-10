"""`Taskset` iteration and the `head`/`shuffle` views over list, generator,
and `INFINITE` `load` implementations."""

import asyncio
import itertools
from contextlib import asynccontextmanager
from types import SimpleNamespace

import pytest

import verifiers.v1 as vf


class CountTask(vf.Task[vf.TaskData]):
    pass


class InfiniteTaskset(vf.Taskset[CountTask, vf.TasksetConfig]):
    INFINITE = True

    def load(self):
        for i in itertools.count():
            yield CountTask(vf.TaskData(idx=i, prompt=f"task {i}"))


class FiniteTaskset(vf.Taskset[CountTask, vf.TasksetConfig]):
    def load(self):
        for i in range(10):
            yield CountTask(vf.TaskData(idx=i, prompt=f"task {i}"))


def idxs(tasks) -> list[int]:
    return [task.data.idx for task in tasks]


def test_head_bounds_an_infinite_taskset() -> None:
    tasks = list(InfiniteTaskset(vf.TasksetConfig()).head(5))
    assert idxs(tasks) == [0, 1, 2, 3, 4]


def test_iteration_materializes_a_finite_taskset() -> None:
    taskset = FiniteTaskset(vf.TasksetConfig())
    assert len(list(taskset)) == 10
    assert idxs(taskset.head(4)) == [0, 1, 2, 3]


def test_head_returns_the_same_taskset_type() -> None:
    view = FiniteTaskset(vf.TasksetConfig()).head(4)
    assert isinstance(view, FiniteTaskset)
    assert view.task_type() is CountTask


def test_shuffle_samples_the_whole_taskset_reproducibly() -> None:
    taskset = FiniteTaskset(vf.TasksetConfig())
    first = idxs(taskset.shuffle().head(5))
    assert first == idxs(taskset.shuffle().head(5))
    assert len(first) == 5 and set(first) <= set(range(10))
    assert first != [0, 1, 2, 3, 4]  # sampled from the whole set, not the head


def test_shuffle_seed_changes_the_sample() -> None:
    taskset = FiniteTaskset(vf.TasksetConfig())
    seeded = idxs(taskset.shuffle(seed=7).head(5))
    assert seeded == idxs(taskset.shuffle(seed=7).head(5))  # reproducible per seed
    assert seeded != idxs(taskset.shuffle().head(5))  # differs from the default seed


def test_shuffle_raises_on_infinite() -> None:
    with pytest.raises(ValueError, match="infinite"):
        InfiniteTaskset(vf.TasksetConfig()).shuffle()


def test_head_then_shuffle_bounds_an_infinite_taskset() -> None:
    head = idxs(InfiniteTaskset(vf.TasksetConfig()).head(5).shuffle())
    assert sorted(head) == [0, 1, 2, 3, 4]
    assert head != [0, 1, 2, 3, 4]  # shuffled within the bounded head


def test_head_only_builds_what_the_run_takes() -> None:
    built: list[int] = []

    class RecordingTaskset(InfiniteTaskset):
        def load(self):
            for i in itertools.count():
                built.append(i)
                yield CountTask(vf.TaskData(idx=i, prompt=f"task {i}"))

    list(RecordingTaskset(vf.TasksetConfig()).head(3))
    assert built == [0, 1, 2]


def test_views_do_not_mutate_the_base_taskset() -> None:
    taskset = InfiniteTaskset(vf.TasksetConfig())
    view = taskset.head(3)
    assert view.INFINITE is False and taskset.INFINITE is True
    assert idxs(taskset.head(2)) == [0, 1]  # base iterates untransformed


def test_harbor_multi_step_reward_strategies_match_harbor() -> None:
    from verifiers.v1.tasksets.harbor.taskset import (
        aggregate_step_rewards,
        min_reward_failure,
    )

    results = [
        {"name": "one", "rewards": {"reward": 1.0, "style": 0.5}},
        {"name": "two", "rewards": {"reward": 0.5}},
    ]
    assert aggregate_step_rewards(results, "mean") == {
        "reward": 0.75,
        "style": 0.25,
    }
    assert aggregate_step_rewards(results, "final") == {"reward": 0.5}
    assert aggregate_step_rewards(
        [*results, {"name": "failed", "rewards": None}], "mean"
    ) == {
        "reward": 0.75,
        "style": 0.25,
    }
    assert (
        aggregate_step_rewards([*results, {"name": "failed", "rewards": None}], "final")
        == {}
    )
    assert min_reward_failure({"reward": 0.5}, 1.0) is not None
    assert min_reward_failure({"quality": 0.8}, {"quality": 0.8}) is None
    assert min_reward_failure({}, {"missing": 0.0}) is not None


def test_harbor_taskset_inherits_its_dispatching_environment(monkeypatch) -> None:
    from types import ModuleType

    from verifiers.v1.tasksets.harbor import HarborEnv, HarborTaskset
    from verifiers.v1.utils import loaders

    class CustomHarborTaskset(HarborTaskset):
        pass

    module = ModuleType("custom_harbor")
    module.CustomHarborTaskset = CustomHarborTaskset
    module.__all__ = ["CustomHarborTaskset"]
    monkeypatch.setattr(loaders, "import_taskset", lambda _: module)

    assert HarborTaskset.ENV is HarborEnv
    assert CustomHarborTaskset.ENV is HarborEnv
    assert loaders.environment_class("custom-harbor") is HarborEnv


def test_harbor_taskset_loads_multi_step_layout(tmp_path, monkeypatch) -> None:
    pytest.importorskip("harbor")
    from verifiers.v1.tasksets.harbor import HarborConfig, HarborTaskset
    from verifiers.v1.tasksets.harbor import taskset as harbor_module

    task_dir = tmp_path / "multi"
    (task_dir / "environment").mkdir(parents=True)
    (task_dir / "tests").mkdir()
    (task_dir / "tests" / "test.sh").write_text(
        "#!/bin/sh\necho 1 > /logs/verifier/reward.txt\n"
    )
    for name in ("one", "two"):
        step_dir = task_dir / "steps" / name
        step_dir.mkdir(parents=True)
        (step_dir / "instruction.md").write_text(f"Do step {name}.\n")
    (task_dir / "task.toml").write_text(
        """schema_version = "1.4"

[task]
name = "tests/multi"
version = "1.0.0"
description = "multi-step fixture"

[environment]
docker_image = "alpine:3.20"
workdir = "/work"

[[steps]]
name = "one"
min_reward = 0.5

[steps.agent]
timeout_sec = 10.0

[steps.verifier]
timeout_sec = 4.0

[steps.verifier.env]
STEP = "one"

[[steps]]
name = "two"

[steps.healthcheck]
command = "test -f ready"
interval_sec = 0.1
timeout_sec = 1.0
retries = 2

[steps.verifier]
environment_mode = "separate"

[steps.verifier.environment]
docker_image = "alpine:3.20"
"""
    )

    monkeypatch.setattr(harbor_module, "dataset_dir", lambda config: tmp_path)
    config = HarborConfig(
        id="harbor",
        dataset="unused",
        ignore_timeouts=False,
        timeout_multiplier=2,
    )
    (task,) = list(HarborTaskset(config))

    assert task.data.prompt is None
    assert [step.name for step in task.data.steps] == ["one", "two"]
    assert task.data.steps[0].prompt == "Do step one."
    assert task.data.steps[0].timeout.agent == 20
    assert task.data.steps[0].timeout.scoring == 8
    assert task.data.steps[0].verifier_env == {"STEP": "one"}
    assert task.data.steps[1].healthcheck.command == "test -f ready"
    assert task.data.steps[1].verifier.image == "alpine:3.20"


@pytest.mark.asyncio
async def test_harbor_env_routes_and_aggregates_multi_step_tasks() -> None:
    from verifiers.v1.tasksets.harbor import (
        HarborData,
        HarborEnv,
        HarborStep,
        HarborTask,
    )
    from verifiers.v1.trace import AgentInfo, Trace, TraceTask

    class FakeRuntime:
        network_restricted = False

    class FakeAgent:
        def __init__(self, rewards):
            self.rewards = iter(rewards)
            self.runtime = FakeRuntime()
            self.tasks = []
            self.traces = []
            self.provisions = 0

        @asynccontextmanager
        async def provision(self, task):
            self.provisions += 1
            yield self.runtime

        async def run(self, task, runtime=None):
            self.tasks.append((task, runtime))
            trace = Trace(
                task=TraceTask(type=type(task).__name__, data=task.data),
                agent=AgentInfo(config=vf.AgentConfig()),
                is_completed=True,
                ok=True,
            )
            if task.data.current_step is not None:
                rewards = next(self.rewards)
                trace.info.update(
                    harbor_step=task.data.current_step,
                    harbor_step_rewards=rewards,
                )
                for key, value in rewards.items():
                    trace.record_reward(key, value)
            self.traces.append(trace)
            return trace

    steps = [
        HarborStep(name="one", prompt="one"),
        HarborStep(name="two", prompt="two"),
    ]
    task = HarborTask(HarborData(idx=0, prompt=None, task_dir="/task", steps=steps))
    agent = FakeAgent([{"reward": 1.0, "style": 0.5}, {"reward": 0.5}])
    env = object.__new__(HarborEnv)
    env.config = SimpleNamespace(resume_trajectory=False)

    await env.run(task, SimpleNamespace(agent=agent))
    assert agent.provisions == 1
    assert [run.data.current_step for run, _ in agent.tasks] == ["one", "two"]
    assert all(runtime is agent.runtime for _, runtime in agent.tasks)

    episode = vf.Episode(traces=agent.traces)
    await env.finalize(task, episode)
    assert all(trace.reward == pytest.approx(1.0) for trace in agent.traces)
    assert agent.traces[0].rewards["reward"].score == pytest.approx(0.75)
    assert agent.traces[0].rewards["style"].score == pytest.approx(0.25)
    assert agent.traces[1].metrics["harbor_step/two/reward"] == 0.5


@pytest.mark.asyncio
async def test_harbor_env_leaves_single_step_tasks_on_the_plain_path() -> None:
    from verifiers.v1.tasksets.harbor import HarborData, HarborEnv, HarborTask

    class FakeAgent:
        def __init__(self):
            self.calls = []

        async def run(self, task, runtime=None):
            self.calls.append((task, runtime))

    agent = FakeAgent()
    env = object.__new__(HarborEnv)
    env.config = SimpleNamespace(resume_trajectory=False)
    task = HarborTask(HarborData(idx=0, prompt="ordinary", task_dir="/task"))

    await env.run(task, SimpleNamespace(agent=agent))
    assert agent.calls == [(task, None)]


@pytest.mark.parametrize("phase", ["setup", "finalize"])
@pytest.mark.asyncio
async def test_harbor_resumed_steps_enforce_non_agent_timeouts(
    phase, monkeypatch
) -> None:
    from verifiers.v1.tasksets.harbor import (
        HarborData,
        HarborEnv,
        HarborStep,
        HarborTask,
    )

    class FakeInteraction:
        trace = SimpleNamespace(info={}, record_metric=lambda *args: None)

        async def turn(self, prompt):
            return SimpleNamespace(terminated=False)

    class FakeAgent:
        @asynccontextmanager
        async def interaction(self, task, runtime=None):
            yield FakeInteraction()

    async def setup(step_task, runtime):
        if phase == "setup" and step_task.data.current_step == "two":
            await asyncio.sleep(1)

    async def collect_step(step_task, trace, runtime):
        if phase == "finalize":
            await asyncio.sleep(1)

    async def stage_tests(step_task, runtime):
        return None

    async def step_graded(step_task, runtime):
        return {"reward": 1.0}

    monkeypatch.setattr(HarborTask, "setup", setup)
    monkeypatch.setattr(HarborTask, "collect_step", collect_step)
    monkeypatch.setattr(HarborTask, "_stage_tests", stage_tests)
    monkeypatch.setattr(HarborTask, "_step_graded", step_graded)

    steps = [
        HarborStep(
            name="one",
            prompt="one",
            timeout=vf.TaskTimeout(agent=1, finalize=0.001, scoring=1),
        )
    ]
    if phase == "setup":
        steps[0] = steps[0].model_copy(
            update={"timeout": vf.TaskTimeout(agent=1, finalize=1, scoring=1)}
        )
        steps.append(
            HarborStep(
                name="two",
                prompt="two",
                timeout=vf.TaskTimeout(setup=0.001, agent=1),
            )
        )
    task = HarborTask(HarborData(idx=0, task_dir="/task", steps=steps))
    env = object.__new__(HarborEnv)

    with pytest.raises(TimeoutError):
        await env._run_resumed(task, SimpleNamespace(agent=FakeAgent()), object())
