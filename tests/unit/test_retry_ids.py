"""Retries keep the id of the unit they repeat: an agent run's traces share one id
across attempts, and so do an episode's attempts."""

import pytest

import verifiers.v1 as vf
import verifiers.v1.agent as agent_module
import verifiers.v1.utils.retries as retries_module
from verifiers.v1.env import Env, RunSlot

RETRIES = vf.RetryConfig(max_retries=2)
FAILURE = vf.Error(type="SandboxError", message="boom")


@pytest.fixture(autouse=True)
def no_backoff(monkeypatch):
    monkeypatch.setattr(agent_module, "backoff", lambda attempt: 0.0)
    monkeypatch.setattr(retries_module, "backoff", lambda attempt: 0.0)


def make_trace(**kwargs) -> vf.Trace:
    return vf.Trace(
        agent=vf.AgentInfo(config=vf.AgentConfig()),
        task=vf.TraceTask(
            type="Task", data=vf.TaskData(idx=0, prompt=None), key="k", hash="h"
        ),
        **kwargs,
    )


@pytest.mark.asyncio
async def test_agent_retries_keep_the_trace_id(monkeypatch):
    minted: list[vf.Trace] = []

    class FakeRollout:
        def __init__(self, *, on_trace, trace_id, attempt, **_):
            self.trace = make_trace(id=trace_id, attempt=attempt)
            on_trace(self.trace)

        async def open(self) -> bool:
            if self.trace.attempt < 3:
                self.trace.errors.append(FAILURE)
            else:
                self.trace.ok = True
            return False

        async def close(self) -> vf.Trace:
            return self.trace

    monkeypatch.setattr(agent_module, "Rollout", FakeRollout)
    agent = vf.Agent(vf.AgentConfig(model="test-model", retries=RETRIES))
    monkeypatch.setattr(agent, "_rollout_params", lambda *args: {})

    trace = await agent.run(
        vf.Task(vf.TaskData(idx=0, prompt=None)), on_trace=minted.append
    )

    assert [t.attempt for t in minted] == [1, 2, 3]
    assert {t.id for t in minted} == {trace.id}
    assert trace.attempt == 3
    assert [e.type for e in trace.errors] == ["SandboxError", "SandboxError"]


@pytest.mark.asyncio
async def test_episode_retries_keep_the_episode_id():
    calls: list[tuple[str, int]] = []

    class FakeEnv:
        config = vf.EnvConfig(retries=RETRIES)

        async def run_episode(self, task, ctx, *, episode_id, attempt, **_):
            calls.append((episode_id, attempt))
            episode = vf.Episode(
                id=episode_id,
                attempt=attempt,
                task=make_trace().task,
                traces=[make_trace()],
            )
            if attempt < 3:
                episode.errors.append(FAILURE)
            else:
                episode.ok = True
            return episode

    slot = RunSlot(vf.Task(vf.TaskData(idx=0, prompt=None)))
    episode = await Env.run_slot(FakeEnv(), slot, ctx=None)

    assert calls == [(slot.id, 1), (slot.id, 2), (slot.id, 3)]
    assert (episode.id, episode.attempt) == (slot.id, 3)
    assert [e.type for e in episode.errors] == ["SandboxError", "SandboxError"]
