"""Offline checks for Flow's checkpoints, call reuse, steering, and admission."""

import asyncio
import json
import os

import pytest

import verifiers.v1 as vf
from verifiers.v1.flow import (
    Flow,
    FlowConfig,
    GitArtifacts,
    Transition,
    UnitData,
    stage,
)
from verifiers.v1.flow.artifacts import git
from verifiers.v1.flow.unit import Unit


class Data(UnitData):
    value: int = 0


class AgentFlowConfig(FlowConfig):
    worker: vf.AgentConfig = vf.AgentConfig(
        harness={"id": "null"}, runtime=vf.SubprocessConfig()
    )


async def test_recorded_trace_survives_interrupted_stage(tmp_path, monkeypatch):
    from verifiers.v1.trace import AgentInfo, TraceTask

    monkeypatch.setenv("VF_RUN_LABEL", "outer")
    recorded, finish = asyncio.Event(), asyncio.Event()
    traces = []

    async def rollout(agent, task, *, on_trace, **kwargs):
        for ok in (False, True):
            trace = vf.Trace(
                task=TraceTask(type="Task", data=task.data),
                agent=AgentInfo(config=agent.config),
                ok=ok,
                is_completed=True,
            )
            traces.append(trace)
            on_trace(trace)
        return trace

    monkeypatch.setattr(vf.Agent, "run", rollout)

    class Example(Flow[AgentFlowConfig]):
        async def setup(self):
            assert os.environ["VF_RUN_LABEL"] == self.label
            self.create_unit("t", stage="work", data=Data())

        @stage
        async def work(self, unit: Unit[Data]):
            trace = await self.agents.worker.run(
                vf.Task(vf.TaskData(prompt="solve")),
                key="solve",
                inputs={},
            )
            assert trace.id == traces[-1].id
            unit.data.value = 1
            recorded.set()
            await finish.wait()
            return Transition("done", status="terminal", data=unit.data)

    cfg = AgentFlowConfig(
        model="offline", client={"type": "eval", "base_url": "http://localhost:1"}
    )
    flow = Example(cfg, root=tmp_path)
    running = asyncio.create_task(flow.run())
    await asyncio.wait_for(recorded.wait(), 10)
    running.cancel()
    with pytest.raises(asyncio.CancelledError):
        await running
    assert flow.unit("t").state().data.value == 0 and not flow.active
    assert os.environ["VF_RUN_LABEL"] == "outer"

    finish.set()
    flow = Example(cfg, root=tmp_path)
    assert (await flow.run()).counts == {"terminal": 1}
    assert flow.unit("t").state().data.value == 1 and len(traces) == 2
    assert flow.traces.get(traces[0].id) is None
    assert flow.traces.get(traces[-1].id) is not None
    assert not list((tmp_path / "live").iterdir())
    with (tmp_path / "transitions.jsonl").open() as file:
        events = [json.loads(line) for line in file]
    produced, attached = [
        e
        for e in events
        if e["type"] == "call" and e["status"] in ("succeeded", "attached")
    ]
    producer, consumer = produced["invocation"], attached["invocation"]
    assert attached["source_call"] == producer["call"] != consumer["call"]
    assert (
        attached["source_execution"] == producer["execution"] != consumer["execution"]
    )
    assert produced["trace_id"] == attached["trace_id"] == traces[-1].id


async def test_parallel_calls_reuses_successes_until_inputs_change(tmp_path):
    calls, unavailable = [], True

    async def work(value, index):
        calls.append((value, index))
        if unavailable and index == 1:
            raise RuntimeError("unavailable")
        return value

    class Example(Flow):
        async def setup(self):
            self.create_unit("t", stage="work", data=Data())

        @stage
        async def work(self, unit: Unit[Data]):
            results = await self.gather(
                *(
                    self.attempt(
                        work,
                        unit.data.value,
                        i,
                        output=int,
                        key=str(i),
                        inputs=unit.data,
                    )
                    for i in range(2)
                )
            )
            return Transition(
                "evaluated", status="terminal" if all(r.ok for r in results) else "held"
            )

    flow = Example(FlowConfig(), root=tmp_path)
    assert (await flow.run()).counts == {"held": 1}
    unit = flow.unit("t")
    unit.steer(status="ready")
    unavailable = False
    assert (await flow.run()).counts == {"terminal": 1}
    assert calls.count((0, 0)) == 1 and calls.count((0, 1)) == 2
    unit.steer(data={"value": 1}, expected=unit.state().revision, status="ready")
    assert (await flow.run()).counts == {"terminal": 1}
    assert {i for value, i in calls if value == 1} == {0, 1}


@pytest.mark.parametrize("route", [None, "work"])
async def test_live_controls_survive_stage_publication(tmp_path, route):
    entered, finish = asyncio.Event(), asyncio.Event()

    class Example(Flow):
        async def setup(self):
            unit = self.create_unit("t", stage="work", data=Data())
            unit.steer(note="first")

        @stage
        async def work(self, unit: Unit[Data]):
            assert unit.notes == "first"
            entered.set()
            await finish.wait()
            unit.data.value = 1
            return Transition("built", stage="review", data=unit.data)

        review = repair = work

    flow = Example(FlowConfig(), root=tmp_path)
    running = asyncio.create_task(flow.run())
    await asyncio.wait_for(entered.wait(), 10)
    unit = flow.unit("t")
    try:
        if route is not None:
            unit.steer(stage="repair")
        unit.steer(status="held", stage=route, note="late")
        assert unit.inspect().active and flow.active["t"].stage == "work"
        with pytest.raises(RuntimeError, match="still active"):
            unit.steer(data={"value": 2}, expected=unit.state().revision)
    finally:
        finish.set()
        await running
    state = unit.state()
    assert (state.stage, state.status, state.data.value) == (
        route or "review",
        "held",
        1,
    )
    assert state.notes == ["late"]
    assert not unit.inspect().active
    old = unit.state().revision
    unit.steer(data={"value": 2}, expected=old)
    with pytest.raises(ValueError, match="stale"):
        unit.steer(data={"value": 3}, expected=old)


def test_artifact_revisions_are_retained_and_independent_of_workflow(tmp_path):
    unit = Unit.create(
        tmp_path / "units" / "unit",
        stage="work",
        data=Data(),
        stages=["work"],
    )
    store, revision = GitArtifacts(unit), unit.state().revision
    base = store.write(base=None, files={"rubric.md": "first"})
    newer = store.write(base=base, files={"rubric.md": "second"})
    assert store.write(base=base, files={"rubric.md": "second"}) == newer
    assert unit.state().revision == revision
    git(unit.path, "gc", "--prune=now")
    assert store.read(base, "rubric.md") == "first"
    assert store.read(newer, "rubric.md") == "second"
    with pytest.raises(ValueError, match="unsafe"):
        store.write(base=base, files={"../escape": "bad"})


async def test_admission_reserves_units_and_run_reports_idle_or_drain(tmp_path):
    seen = []

    class Example(Flow):
        async def setup(self):
            self.create_unit("campaign", stage="work", data=Data(value=2))
            self.create_unit("other", stage="work", data=UnitData())

        def admit(self, unit):
            return not self.active

        @stage
        async def work(self, unit):
            await asyncio.sleep(0)
            seen.append((unit.id, list(self.active)))
            return Transition("waiting", status="waiting")

    flow = Example(FlowConfig(), root=tmp_path)
    result = await flow.run()
    assert result.reason == "idle" and result.counts == {"waiting": 2}
    assert seen == [("campaign", ["campaign"]), ("other", ["other"])]
    assert isinstance(result.units["campaign"].data, Data)
    assert type(result.units["other"].data) is UnitData
    flow.unit("campaign").steer(status="ready")
    flow.drain()
    result = await flow.run()
    assert result.reason == "draining" and result.units["campaign"].status == "ready"
    assert len(seen) == 2 and not flow.active
