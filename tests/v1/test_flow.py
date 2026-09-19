"""Offline checks for Flow's checkpoints, call reuse, steering, and admission."""

import asyncio

import pytest

import verifiers.v1 as vf
from verifiers.v1.cli.output import read_jsonl
from verifiers.v1.flow import (
    AgentWork,
    Ctx,
    Flow,
    FlowConfig,
    GitArtifacts,
    Pipeline,
    Transition,
    UnitData,
    agent,
    fn,
)
from verifiers.v1.flow.unit import Unit, git


class Data(UnitData):
    value: int = 0


class AgentFlowConfig(FlowConfig):
    worker: vf.AgentConfig = vf.AgentConfig(
        harness={"id": "null"}, runtime=vf.SubprocessConfig()
    )


async def test_recorded_trace_survives_interrupted_stage(tmp_path, monkeypatch):
    from verifiers.v1.trace import AgentInfo, TraceTask

    recorded, finish = asyncio.Event(), asyncio.Event()
    traces = []

    async def rollout(work, agent, on_trace):
        for ok in (False, True):
            trace = vf.Trace(
                task=TraceTask(type="Task", data=work.task.data),
                agent=AgentInfo(config=agent.config),
                ok=ok,
                is_completed=True,
            )
            traces.append(trace)
            on_trace(trace)
        return trace

    monkeypatch.setattr(AgentWork, "rollout", rollout)

    async def stage(ctx: Ctx[Data]):
        trace = await ctx.call(
            agent("worker", vf.Task(vf.TaskData(prompt="solve")), inputs={}),
            key="solve",
        )
        assert trace.id == traces[-1].id
        ctx.data.value = 1
        recorded.set()
        await finish.wait()
        return Transition("done", status="terminal", data=ctx.data)

    pipeline = Pipeline({"work": stage})
    cfg = AgentFlowConfig(
        model="offline", client={"type": "eval", "base_url": "http://localhost:1"}
    )
    async with Flow(tmp_path, cfg, pipeline) as flow:
        unit = flow.create_unit("t", stage="work", data=Data())
        running = asyncio.create_task(flow.run())
        await asyncio.wait_for(recorded.wait(), 10)
        running.cancel()
        with pytest.raises(asyncio.CancelledError):
            await running
        assert unit.state().data.value == 0 and not flow.active

    finish.set()
    async with Flow(tmp_path, cfg, pipeline) as flow:
        assert (await flow.run()).counts == {"terminal": 1}
        assert flow.unit("t").state().data.value == 1 and len(traces) == 2
        assert all(flow.traces.get(t.id) is not None for t in traces)
    events = read_jsonl(tmp_path / "transitions.jsonl")
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
    rollouts = [e for e in events if e["type"] == "rollout"]
    assert all(e["invocation"] == producer for e in rollouts)
    assert [(e["rollout"], e["status"]) for e in rollouts] == [
        (1, "started"),
        (1, "failed"),
        (2, "started"),
        (2, "succeeded"),
    ]


async def test_partial_spread_reuses_successes_until_inputs_change(tmp_path):
    calls, unavailable = [], True

    async def work(value, index):
        calls.append((value, index))
        if unavailable and index == 1:
            raise RuntimeError("unavailable")
        return value

    async def stage(ctx: Ctx[Data]):
        results = await ctx.spread(
            [
                fn(work, ctx.data.value, i, output=int, inputs=ctx.data)
                for i in range(2)
            ],
            key=str,
        )
        return Transition(
            "evaluated", status="terminal" if all(r.ok for r in results) else "held"
        )

    async with Flow(tmp_path, FlowConfig(), Pipeline({"work": stage})) as flow:
        unit = flow.create_unit("t", stage="work", data=Data())
        assert (await flow.run()).counts == {"held": 1}
        unit.steer(status="ready")
        unavailable = False
        assert (await flow.run()).counts == {"terminal": 1}
        assert calls.count((0, 0)) == 1 and calls.count((0, 1)) == 2
        unit.steer(data={"value": 1}, expected=unit.head(), status="ready")
        assert (await flow.run()).counts == {"terminal": 1}
        assert {i for value, i in calls if value == 1} == {0, 1}


@pytest.mark.parametrize("route", [None, "repair"])
async def test_live_controls_survive_stage_publication(tmp_path, route):
    entered, finish = asyncio.Event(), asyncio.Event()

    async def stage(ctx: Ctx[Data]):
        assert ctx.notes() == "first"
        entered.set()
        await finish.wait()
        ctx.data.value = 1
        return Transition("built", stage="review", data=ctx.data)

    pipeline = Pipeline(dict.fromkeys(("work", "review", "repair"), stage))
    async with Flow(tmp_path, FlowConfig(), pipeline) as flow:
        unit = flow.create_unit("t", stage="work", data=Data())
        unit.steer(note="first")
        running = asyncio.create_task(flow.run())
        await asyncio.wait_for(entered.wait(), 10)
        try:
            unit.steer(status="held", stage=route, note="late")
            assert unit.inspect().active.stage == flow.active["t"].stage == "work"
            with pytest.raises(RuntimeError, match="still active"):
                unit.steer(data={"value": 2}, expected=unit.head())
        finally:
            finish.set()
            await running
        state = unit.state()
        assert (state.stage, state.status, state.data.value) == (
            route or "review",
            "held",
            1,
        )
        assert [note.text for note in state.notes] == ["late"]
        assert unit.inspect().active is None
        old = unit.head()
        unit.steer(data={"value": 2}, expected=old)
        with pytest.raises(ValueError, match="stale"):
            unit.steer(data={"value": 3}, expected=old)


def test_artifact_revisions_are_retained_and_independent_of_workflow(tmp_path):
    unit = Unit.create(
        tmp_path / "unit",
        stage="work",
        data=Data(),
        stages=["work"],
        events=tmp_path / "events.jsonl",
    )
    store, head = GitArtifacts(unit), unit.head()
    base = store.write(base=None, files={"rubric.md": "first"})
    newer = store.write(base=base, files={"rubric.md": "second"})
    assert store.write(base=base, files={"rubric.md": "second"}) == newer
    assert unit.head() == head
    unit.check_clean()
    git(unit.path, "gc", "--prune=now")
    assert store.read(base, "rubric.md") == "first"
    assert store.read(newer, "rubric.md") == "second"
    with pytest.raises(ValueError, match="unsafe"):
        store.write(base=base, files={"../escape": "bad"})


async def test_admission_reserves_units_and_run_reports_quiescence_or_drain(tmp_path):
    seen = []

    async def stage(ctx):
        await asyncio.sleep(0)
        seen.append((ctx.unit.id, list(ctx.flow.active)))
        return Transition("waiting", status="waiting")

    pipeline = Pipeline({"work": stage}, admit=lambda unit, flow: not flow.active)
    async with Flow(tmp_path, FlowConfig(), pipeline) as flow:
        assert flow.units() == []
        unit = flow.create_unit("campaign", stage="work", data=Data(value=2))
        head = unit.head()
        assert flow.create_unit("campaign", stage="work", data=Data()).head() == head
        assert unit.state().data.value == 2
        flow.create_unit("other", stage="work", data=UnitData())
        result = await flow.run()
        assert result.reason == "quiescent" and result.counts == {"waiting": 2}
        assert seen == [("campaign", ["campaign"]), ("other", ["other"])]
        assert isinstance(result.units["campaign"].data, Data)
        assert type(result.units["other"].data) is UnitData
        unit.steer(status="ready")
        flow.drain()
        result = await flow.run()
        assert (
            result.reason == "draining" and result.units["campaign"].status == "ready"
        )
        assert len(seen) == 2 and not flow.active
