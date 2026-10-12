"""Env hooks over one episode, without a model or a box: `trace_labels` stamped into
every minted trace before observers see it, and the `run()` outcome kept on the
episode. Agent rollouts are replaced by a stub that mints a
trace through the real `_EpisodeAgent` wiring."""

from contextlib import asynccontextmanager

import pytest
from pydantic import BaseModel

import verifiers.v1 as vf
from verifiers.v1.agent import Agent
from verifiers.v1.clients import ModelContext
from verifiers.v1.configs.client import EvalClientConfig
from verifiers.v1.env import RunSlot
from verifiers.v1.trace import TraceTask
from verifiers.v1.utils import retries


class Outcome(BaseModel):
    verdict: str
    score: float = 0.0


class DuetConfig(vf.EnvConfig):
    solver: vf.AgentConfig = vf.AgentConfig()
    judge: vf.AgentConfig = vf.AgentConfig()


def mint(agent: Agent, task: vf.Task, on_trace) -> vf.Trace:
    trace = vf.Trace(
        task=TraceTask(type=type(task).__name__, data=task.data),
        agent=vf.AgentInfo(config=agent.config),
    )
    on_trace(trace)
    return trace


@pytest.fixture
def stub_rollouts(monkeypatch):
    """`Agent.run` and `Agent.interaction` mint a finished, ok trace through the
    caller's `on_trace`, as a real rollout does at construction."""

    async def run_once(
        self, task, runtime, tools, on_trace, collect_artifacts, checkpoint
    ):
        trace = mint(self, task, on_trace)
        trace.is_completed = trace.ok = True
        return trace

    @asynccontextmanager
    async def interaction(self, task, *, on_trace=None, **_):
        trace = mint(self, task, on_trace)
        try:
            yield trace
        finally:
            trace.is_completed = trace.ok = True

    monkeypatch.setattr(Agent, "_run_once", run_once)
    monkeypatch.setattr(Agent, "interaction", interaction)
    monkeypatch.setattr(retries, "backoff", lambda attempt: 0.0)


def ctx() -> ModelContext:
    return ModelContext(model="stub", client=EvalClientConfig())


def task() -> vf.Task:
    return vf.Task(vf.TaskData(idx=0, prompt="hi"))


class LabeledEnv(vf.Env[DuetConfig]):
    def __init__(self, config):
        super().__init__(config)
        self.seen_by_run: list[dict] = []

    def trace_labels(self, task):
        return {"job": "j1", "stage": {"name": "solve", "visit": 2}}

    async def run(self, task, agents):
        await agents.solver.run(
            task, on_trace=lambda t: self.seen_by_run.append(dict(t.info))
        )
        async with agents.judge.interaction(task):
            pass


def env(cls, **config) -> vf.Env:
    return cls(DuetConfig(taskset=vf.TasksetConfig(id="echo-v1"), **config))


async def test_trace_labels_land_before_observers(stub_rollouts):
    e = env(LabeledEnv)
    observed: list[tuple[str, dict]] = []
    episode = await e.run_episode(
        task(), ctx(), on_trace=lambda t: observed.append((t.agent.name, dict(t.info)))
    )
    labels = {"job": "j1", "stage": {"name": "solve", "visit": 2}}
    assert episode.ok
    # both the run() trace and the interaction() trace, at mint
    assert observed == [("solver", labels), ("judge", labels)]
    assert e.seen_by_run == [labels]
    assert [t.info for t in episode.traces] == [labels, labels]
    # each trace holds its own copy
    episode.traces[0].info["stage"]["visit"] = 3
    assert episode.traces[1].info["stage"]["visit"] == 2


async def test_trace_labels_reach_run_slot_observers(stub_rollouts):
    e = env(LabeledEnv)
    observed: list[dict] = []
    slot = RunSlot(task())
    episode = await e.run_slot(
        slot, ctx(), on_trace=lambda t: observed.append(dict(t.info))
    )
    assert [o["job"] for o in observed] == ["j1", "j1"]
    assert all(t.info["job"] == "j1" for t in slot.traces)
    assert episode.traces == slot.traces


class PlainEnv(vf.Env[DuetConfig]):
    async def run(self, task, agents):
        await agents.solver.run(task)


async def test_default_env_adds_no_labels_and_no_outcome(stub_rollouts):
    episode = await env(PlainEnv).run_episode(task(), ctx())
    assert episode.ok
    assert episode.traces[0].info == {}
    assert episode.outcome is None
    assert episode.outcome_as(Outcome) is None
    assert "outcome" not in episode.to_record()


class BadLabelsEnv(PlainEnv):
    def trace_labels(self, task):
        raise RuntimeError("no labels")


async def test_trace_labels_raise_fails_the_episode(stub_rollouts):
    episode = await env(BadLabelsEnv).run_episode(task(), ctx())
    assert not episode.ok
    assert episode.traces == []
    assert episode.last_error.type == "EnvError"
    assert "trace_labels()" in episode.last_error.message


class OutcomeEnv(vf.Env[DuetConfig]):
    async def run(self, task, agents):
        await agents.solver.run(task)
        return Outcome(verdict="pass")

    async def finalize(self, task, episode):
        episode.outcome["score"] = 0.5


async def test_run_outcome_is_kept_and_finalize_amends_it(stub_rollouts):
    episode = await env(OutcomeEnv).run_episode(task(), ctx())
    assert episode.ok
    assert episode.outcome == {"verdict": "pass", "score": 0.5}
    assert episode.outcome_as(Outcome) == Outcome(verdict="pass", score=0.5)
    record = episode.to_record()
    assert record["outcome"] == {"verdict": "pass", "score": 0.5}
    loaded = vf.WireEpisode.model_validate(record)
    assert loaded.outcome_as(Outcome) == Outcome(verdict="pass", score=0.5)


class NonModelReturnEnv(PlainEnv):
    async def run(self, task, agents):
        await agents.solver.run(task)
        return {"verdict": "ignored"}


async def test_non_model_return_is_ignored(stub_rollouts):
    episode = await env(NonModelReturnEnv).run_episode(task(), ctx())
    assert episode.ok
    assert episode.outcome is None


class RetriedOutcomeEnv(vf.Env[DuetConfig]):
    attempts = 0

    async def run(self, task, agents):
        await agents.solver.run(task)
        type(self).attempts += 1
        if self.attempts == 1:
            return Outcome(verdict="first")
        return None

    async def finalize(self, task, episode):
        if self.attempts == 1:
            assert episode.outcome == {"verdict": "first", "score": 0.0}
            raise ValueError("first attempt fails")


async def test_each_retry_attempt_starts_without_outcome(stub_rollouts):
    RetriedOutcomeEnv.attempts = 0
    e = env(RetriedOutcomeEnv, retries=vf.RetryConfig(max_retries=1))
    episode = await e.run_slot(RunSlot(task()), ctx())
    assert RetriedOutcomeEnv.attempts == 2
    assert episode.ok
    assert episode.outcome is None


class FinalizeOnlyOutcomeEnv(PlainEnv):
    async def finalize(self, task, episode):
        episode.outcome = Outcome(verdict="judged").model_dump(mode="json")


async def test_finalize_may_set_the_outcome(stub_rollouts):
    episode = await env(FinalizeOnlyOutcomeEnv).run_episode(task(), ctx())
    assert episode.outcome_as(Outcome) == Outcome(verdict="judged")
