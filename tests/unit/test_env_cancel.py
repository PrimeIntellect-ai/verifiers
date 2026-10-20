"""`RunSlot.cancel()` ends an episode early: `run_slot` returns it flagged `cancelled`,
whether it was running, queued on the episode semaphore, or not yet started. Any
other cancellation still propagates."""

import asyncio
import sys
import types

import pytest

import verifiers.v1 as vf
from verifiers.v1.clients import ModelContext
from verifiers.v1.env import RunSlot
from verifiers.v1.envs.single_agent.env import SingleAgentEnv, SingleAgentEnvConfig
from verifiers.v1.types import SamplingConfig


class SleepTask(vf.Task[vf.TaskData]):
    pass


class SleepTaskset(vf.Taskset[SleepTask, vf.TasksetConfig]):
    def load(self):
        return [SleepTask(vf.TaskData(idx=0, prompt="sleep"))]


class SleepEnv(SingleAgentEnv):
    async def run(self, task, agents):
        await asyncio.sleep(60)


CTX = ModelContext(
    client=vf.EvalClientConfig(), model="model", sampling=SamplingConfig()
)


@pytest.fixture
def env(monkeypatch) -> SleepEnv:
    module = types.ModuleType("sleep_taskset")
    module.SleepTaskset = SleepTaskset
    module.__all__ = ["SleepTaskset"]
    monkeypatch.setitem(sys.modules, "sleep_taskset", module)
    return SleepEnv(
        SingleAgentEnvConfig(
            taskset={"id": "sleep-taskset"}, agent={"harness": {"id": "null"}}
        )
    )


def slot(env: SleepEnv) -> RunSlot:
    (task,) = env.taskset
    (planned,) = env.slots(task)
    return planned


async def test_cancel_running_episode(env):
    planned = slot(env)
    run = asyncio.create_task(env.run_slot(planned, CTX))
    await asyncio.sleep(0.05)
    planned.cancel()
    episode = await run
    assert episode.cancelled and not episode.ok
    assert episode.traces == [] and planned.done


async def test_cancel_queued_episode(env):
    planned = slot(env)
    run = asyncio.create_task(env.run_slot(planned, CTX, asyncio.Semaphore(0)))
    await asyncio.sleep(0.05)
    planned.cancel()
    episode = await run
    assert episode.cancelled and not episode.ok


async def test_cancel_before_start(env):
    planned = slot(env)
    planned.cancel()
    episode = await env.run_slot(planned, CTX)
    assert episode.cancelled and not episode.ok


async def test_task_cancellation_propagates(env):
    run = asyncio.create_task(env.run_slot(slot(env), CTX))
    await asyncio.sleep(0.05)
    run.cancel()
    with pytest.raises(asyncio.CancelledError):
        await run
