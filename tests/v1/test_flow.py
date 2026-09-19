"""Offline execution tests for the durable boundaries: calls, artifacts, and controls."""

import asyncio
import io
import json
import tarfile
from contextlib import suppress

import pytest
from pydantic import Field, ValidationError

import verifiers.v1 as vf
from verifiers.v1.cli.output import read_jsonl
from verifiers.v1.flow import (
    AgentWork,
    ArtifactRevision,
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
from verifiers.v1.flow.__main__ import inspect
from verifiers.v1.flow.traces import Traces
from verifiers.v1.flow.unit import Unit, git


class Data(UnitData):
    revision: ArtifactRevision | None = None
    credits: int = Field(default=1, ge=0)


async def parked(ctx: Ctx) -> Transition:
    return Transition.wait("planned")


def pipeline(stage):
    return Pipeline({"plan": parked, "work": stage, "review": parked})


async def test_restart_after_call_record_recovers_output_without_repeating_work(
    tmp_path,
):
    recorded, blocked = asyncio.Event(), asyncio.Event()
    writes = 0

    async def produce(repo: str, base: str) -> ArtifactRevision:
        nonlocal writes
        writes += 1
        return GitArtifacts(repo).write(base=base, files={"rubric.md": "revised"})

    async def stage(ctx: Ctx[Data]) -> Transition[Data]:
        revision = await ctx.call(
            fn(
                produce,
                str(ctx.unit.path),
                ctx.data.revision,
                output=str,
                inputs=ctx.data.revision,
            ),
            key="author",
        )
        recorded.set()
        await ctx.call(fn(blocked.wait, output=bool))
        ctx.data.revision = revision
        return Transition.end("done", data=ctx.data)

    p = pipeline(stage)
    async with Flow(tmp_path, FlowConfig(), p) as flow:
        unit = flow.create_unit("t", stage="work", data=Data())
        artifacts = GitArtifacts(unit)
        base = artifacts.write(base=None, files={"rubric.md": "initial"})
        unit.steer(data={"revision": base}, expected=unit.head())
        running = asyncio.create_task(flow.run())
        await asyncio.wait_for(recorded.wait(), 10)
        running.cancel()
        with suppress(asyncio.CancelledError):
            await running
        assert unit.state().stage == "work"
        assert unit.state().data.revision == base
        assert not flow.active

    # Readers leave interrupted appends alone; the next exclusive launch trims them.
    for name in ("transitions.jsonl", "traces.jsonl"):
        file = tmp_path / name
        original = file.read_bytes() + b'{"incomplete":'
        file.write_bytes(original)
        read_jsonl(file)
        Traces(tmp_path).get("missing")
        assert file.read_bytes() == original

    blocked.set()
    async with Flow(tmp_path, FlowConfig(), p) as flow:
        assert (await flow.run()).counts == {"terminal": 1}
        unit = flow.unit("t")
        revision = unit.state().data.revision
        assert writes == 1 and revision != base
        GitArtifacts(unit).materialize(revision, tmp_path / "fresh")
        assert (tmp_path / "fresh" / "rubric.md").read_text() == "revised"
        assert GitArtifacts(unit).read(base, "rubric.md") == "initial"
        events = [
            json.loads(line)
            for line in (tmp_path / "transitions.jsonl").read_text().splitlines()
        ]
        calls = [e for e in events if e["type"] == "call"]
        original = next(
            e for e in calls if e["status"] == "succeeded" and e["key"] == "author"
        )
        attached = next(e for e in calls if e["status"] == "attached")
        assert attached["source_call"] == original["call"] != attached["call"]
        assert (
            attached["source_execution"]
            == original["execution"]
            != attached["execution"]
        )
        assert any(e["key"] is None and e["status"] == "cancelled" for e in calls)
        record = json.loads(next((tmp_path / "calls/t").glob("*.json")).read_text())
        assert (
            record["execution"] == original["execution"]
            and record["call"] == original["call"]
        )


async def test_partial_spread_reuses_successes_and_artifact_edit_changes_inputs(
    tmp_path,
):
    executions = []
    unavailable = True

    async def work(revision: str, index: int) -> str:
        executions.append((revision, index))
        if unavailable and index == 1:
            raise RuntimeError("provider unavailable")
        return revision

    async def stage(ctx: Ctx[Data]) -> Transition[Data]:
        results = await ctx.spread(
            [
                fn(work, ctx.data.revision, i, output=str, inputs=ctx.data.revision)
                for i in range(3)
            ],
            key=lambda i: f"solve/{i}",
        )
        if any(not result.ok for result in results):
            return Transition.hold("one call failed")
        return Transition.end("done")

    p = pipeline(stage)
    async with Flow(tmp_path, FlowConfig(), p) as flow:
        unit = flow.create_unit("t", stage="work", data=Data())
        base = GitArtifacts(unit).write(base=None, files={"task.txt": "v1"})
        unit.steer(data={"revision": base}, expected=unit.head())
        assert (await flow.run()).counts == {"held": 1}
        unit.steer(status="ready", note="retry the failed call")
        unavailable = False
        assert (await flow.run()).counts == {"terminal": 1}
        assert executions.count((base, 0)) == executions.count((base, 2)) == 1
        assert executions.count((base, 1)) == 2
        edited = GitArtifacts(unit).write(base=base, files={"task.txt": "v2"})
        unit.steer(
            data={"revision": edited},
            expected=unit.head(),
            stage="work",
            status="ready",
        )
        assert (await flow.run()).counts == {"terminal": 1}
        assert [index for revision, index in executions if revision == edited] == [
            0,
            1,
            2,
        ]


@pytest.mark.parametrize("route", [False, True])
async def test_live_controls_win_and_updates_require_settled_current_state(
    tmp_path, route
):
    entered, finish = asyncio.Event(), asyncio.Event()

    async def stage(ctx: Ctx[Data]) -> Transition[Data]:
        assert ctx.notes() == "first note"
        entered.set()
        await finish.wait()
        ctx.data.credits = 0
        return Transition.to("review", "built", data=ctx.data)

    async with Flow(tmp_path, FlowConfig(), pipeline(stage)) as flow:
        unit = flow.create_unit("t", stage="work", data=Data())
        unit.steer(note="first note")
        running = asyncio.create_task(flow.run())
        await asyncio.wait_for(entered.wait(), 10)
        try:
            unit.steer(
                status="held",
                stage="plan" if route else None,
                note="arrived during work",
            )
            snapshot = inspect(tmp_path, "t").units[0]
            assert snapshot.active.stage == flow.active["t"].stage == "work"
            with pytest.raises(TypeError):
                flow.active["other"] = flow.active["t"]
            assert snapshot.state.status == "held"
            with pytest.raises(RuntimeError, match="still active"):
                unit.steer(data={"credits": 2}, expected=unit.head())
        finally:
            finish.set()
            await running
        state = unit.state()
        assert (state.stage, state.status, state.data.credits) == (
            "plan" if route else "review",
            "held",
            0,
        )
        assert [note.text for note in state.notes] == ["arrived during work"]
        assert inspect(tmp_path, "t").units[0].active is None
        old = unit.head()
        with pytest.raises(ValidationError):
            unit.steer(data={"credits": -1}, expected=old)
        with pytest.raises(ValueError, match="unknown stage"):
            unit.steer(stage="typo")
        assert unit.head() == old
        unit.steer(data={"credits": 2}, expected=old)
        with pytest.raises(ValueError, match="stale"):
            unit.steer(data={"credits": 3}, expected=old)
        events = [
            json.loads(line)
            for line in (tmp_path / "transitions.jsonl").read_text().splitlines()
        ]
        assert any(
            event["type"] == "steer" and event["sha"] == unit.head() for event in events
        )


def test_artifacts_are_retained_isolated_and_cannot_escape(tmp_path):
    unit = Unit.create(
        tmp_path / "unit",
        stage="work",
        data=Data(),
        stages=["work"],
        events=tmp_path / "events.jsonl",
    )
    store = GitArtifacts(unit)
    head = unit.head()
    base = store.write(
        base=None, files={"rubric.md": "first", "remove.txt": "obsolete"}
    )
    newer = store.write(base=base, files={"rubric.md": "second", "remove.txt": None})
    assert newer == store.write(
        base=base, files={"rubric.md": "second", "remove.txt": None}
    )
    assert store.write(base=newer, files={"rubric.md": "second"}) == newer
    assert unit.head() == head
    unit.check_clean()
    git(unit.path, "gc", "--prune=now")
    assert store.read(base, "rubric.md") == "first"
    assert store.read(newer, "rubric.md") == "second"
    assert store.read(newer, "remove.txt") is None
    with pytest.raises(ValueError, match="unsafe"):
        store.write(base=base, files={"../escape": "bad"})
    assert not (tmp_path / "escape").exists()

    # A tar link may stay inside the archive yet escape the selected task root.
    archive = io.BytesIO()
    with tarfile.open(fileobj=archive, mode="w") as tar:
        link = tarfile.TarInfo("task/link")
        link.type, link.linkname = tarfile.SYMTYPE, "../outside"
        tar.addfile(link)
    with pytest.raises(ValueError, match="symlink escapes"):
        store.capture(
            base=base, archive=archive.getvalue(), prefix="task", only=("link",)
        )


async def test_uniform_admission_reserves_before_next_candidate_and_reports_facts(
    tmp_path,
):
    seen = []

    async def stage(ctx):
        seen.append((ctx.unit.id, list(ctx.flow.active)))
        await asyncio.sleep(0)
        assert list(ctx.flow.active) == [ctx.unit.id]
        return Transition.wait("operator decision")

    p = Pipeline({"work": stage}, admit=lambda unit, flow: not flow.active)
    async with Flow(tmp_path, FlowConfig(), p) as flow:
        assert flow.units() == []
        unit = flow.create_unit("campaign", stage="work", data=Data(credits=2))
        head = unit.head()
        assert flow.create_unit("campaign", stage="work", data=Data()).head() == head
        assert unit.state().data.credits == 2
        flow.create_unit("other", stage="work", data=UnitData())
        result = await flow.run()
        assert result.reason == "quiescent" and result.counts == {"waiting": 2}
        assert set(result.units) == {"campaign", "other"}
        assert seen == [("campaign", ["campaign"]), ("other", ["other"])]
        assert isinstance(flow.unit("campaign").state().data, Data)
        assert type(flow.unit("other").state().data) is UnitData
        unit.steer(status="ready")
        flow.drain()
        result = await flow.run()
        assert (
            result.reason == "draining" and result.units["campaign"].status == "ready"
        )
        assert len(seen) == 2 and not flow.active


class AgentFlowConfig(FlowConfig):
    worker: vf.AgentConfig = vf.AgentConfig(
        harness={"id": "null"}, runtime=vf.SubprocessConfig()
    )


async def test_agent_retry_evidence_and_declared_semantic_identity(
    tmp_path, monkeypatch
):
    from verifiers.v1.trace import AgentInfo, TraceTask

    executed = 0
    grading = {"rubric": "v1", "judge": "judge-v1"}

    async def rollout(work, agent, on_trace):
        nonlocal executed
        executed += 1
        for ok in (False, True):
            trace = vf.Trace(
                task=TraceTask(type="Task", data=work.task.data),
                agent=AgentInfo(config=agent.config),
                ok=ok,
                is_completed=True,
            )
            on_trace(trace)
        return trace

    monkeypatch.setattr(AgentWork, "rollout", rollout)

    async def stage(ctx):
        trace = await ctx.call(
            agent("worker", vf.Task(vf.TaskData(prompt="solve")), inputs=grading),
            key="solve",
        )
        assert trace.ok
        return Transition.end("done")

    cfg = AgentFlowConfig(
        model="solver-v1",
        client={"type": "eval", "base_url": "http://localhost:1"},
        sampling={},
    )
    async with Flow(tmp_path, cfg, Pipeline({"work": stage})) as flow:
        unit = flow.create_unit("t", stage="work", data=Data())
        assert (await flow.run()).counts == {"terminal": 1}
        cfg.model = "solver-v2"
        cfg.sampling.max_tokens = 123
        cfg.worker.max_turns = 100
        cfg.worker.retries.max_retries = 5
        cfg.client.api_key_var = "DIFFERENT_CREDENTIAL"
        unit.steer(status="ready", note="retry budget changed")
        assert (await flow.run()).counts == {"terminal": 1}
        assert executed == 1
        grading["judge"] = "judge-v2"
        unit.steer(status="ready")
        await flow.run()
        assert executed == 2  # Core hashes only declared inputs, not model settings.
        events = [
            json.loads(line)
            for line in (tmp_path / "transitions.jsonl").read_text().splitlines()
        ]
        rollouts = [e for e in events if e["type"] == "rollout"]
        assert [(e["rollout"], e["status"]) for e in rollouts[:4]] == [
            (1, "started"),
            (1, "failed"),
            (2, "started"),
            (2, "succeeded"),
        ]
        assert all(flow.traces.get(e["trace_id"]) is not None for e in rollouts)
