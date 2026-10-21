"""Trace labels: what `trace_labels(task)` returned, kept once per trace under the one
reserved `info["labels"]` key (`Trace.labels`), apart from the rest of its `info`.
Agent rollouts are replaced by a stub that mints a finished trace through the real
agent wiring."""

import pytest

import verifiers.v1 as vf
from verifiers.v1.agent import Agent
from verifiers.v1.clients import ModelContext
from verifiers.v1.configs.client import EvalClientConfig
from verifiers.v1.trace import TraceTask
from verifiers.v1.utils import retries

LABELS = {"job": "j1", "stage": {"name": "solve", "visit": 2}}


@pytest.fixture
def stub_rollouts(monkeypatch):
    async def run_once(
        self,
        task,
        runtime,
        tools,
        on_trace,
        collect_artifacts,
        checkpoint,
        on_save,
        **_,
    ):
        trace = vf.Trace(
            task=TraceTask(type=type(task).__name__, data=task.data),
            agent=vf.AgentInfo(config=self.config),
        )
        on_trace(trace)
        trace.is_completed = trace.ok = True
        return trace

    monkeypatch.setattr(Agent, "_run_once", run_once)
    monkeypatch.setattr(retries, "backoff", lambda attempt: 0.0)


class SoloConfig(vf.EnvConfig):
    solver: vf.AgentConfig = vf.AgentConfig()


class PlainEnv(vf.Env[SoloConfig]):
    async def run(self, task, agents):
        await agents.solver.run(task)


class LabeledEnv(PlainEnv):
    def trace_labels(self, task):
        return {"job": "j1", "stage": {"name": "solve", "visit": 2}}


class BadLabelsEnv(PlainEnv):
    def trace_labels(self, task):
        raise RuntimeError("no labels")


async def run(cls) -> vf.Episode:
    env = cls(SoloConfig(taskset=vf.TasksetConfig(id="echo")))
    return await env.run_episode(
        vf.Task(vf.TaskData(idx=0, prompt="hi")),
        ModelContext(model="stub", client=EvalClientConfig()),
    )


async def test_each_trace_keeps_the_labels_under_one_info_key(stub_rollouts):
    episode = await run(LabeledEnv)
    assert episode.ok
    (trace,) = episode.traces
    assert trace.info == {"labels": LABELS}
    assert trace.labels == LABELS
    trace.info["harness_note"] = "x"
    trace.labels["stage"]["visit"] = 3
    # each trace gets its own copy: an edit on one never reaches the env's labels
    assert (await run(LabeledEnv)).traces[0].labels == LABELS


async def test_labels_round_trip_through_the_record(stub_rollouts):
    record = (await run(LabeledEnv)).to_record()
    assert "labels" not in record
    assert record["traces"][0]["info"]["labels"] == LABELS
    assert vf.WireEpisode.model_validate(record).traces[0].labels == LABELS


async def test_no_labels_without_trace_labels_or_after_it_raised(stub_rollouts):
    plain = await run(PlainEnv)
    assert plain.traces[0].info == {} and plain.traces[0].labels == {}
    failed = await run(BadLabelsEnv)
    assert not failed.ok
    assert failed.traces == []


def test_an_episode_record_with_its_old_labels_field_still_loads():
    record = vf.Episode(
        task=TraceTask(type="Task", data=vf.TaskData(idx=0, prompt="hi"))
    ).to_record()
    record["labels"] = LABELS
    assert vf.WireEpisode.model_validate(record).traces == []
