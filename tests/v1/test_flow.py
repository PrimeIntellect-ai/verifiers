"""The flow core: units and transitions, durable calls, spreads, holds, the campaign, drain."""

import asyncio
import importlib
import json
import subprocess

import pytest

import verifiers.v1 as vf
from verifiers.v1.flow import (
    CAMPAIGN,
    Ctx,
    Flow,
    FlowConfig,
    Pipeline,
    Result,
    Transition,
    agent,
    fn,
    succeeded,
)
from verifiers.v1.flow.unit import Unit


def seat(**kw) -> vf.AgentConfig:
    return vf.AgentConfig(harness={"id": "null"}, runtime=vf.SubprocessConfig(), **kw)


class Cfg(FlowConfig):
    alpha: vf.AgentConfig = seat()
    beta: vf.AgentConfig = seat()


calls: list[str] = []
failing: set[str] = set()  # which work fails, outside the key: what a rerun redoes


def work(tag: str) -> str:
    calls.append(tag)
    if tag in failing:
        raise ValueError(f"{tag} is bad")
    return tag


async def plan(ctx: Ctx) -> Transition:
    for i in range(2):
        ctx.flow.create_task(f"t{i}", {"stage": "build", "credits": 1})
    return Transition.wait("planned")


async def build(ctx: Ctx) -> Transition:
    value = await ctx.call(fn(work, f"build-{ctx.unit.id}"), key="build")
    return Transition.to(
        "check", "built", value, files={"out.txt": value}, state={"credits": 0}
    )


async def check(ctx: Ctx) -> Transition:
    results: list[Result[str]] = await ctx.spread(
        [fn(work, f"item-{ctx.unit.id}-{i}") for i in range(3)],
        key=lambda i: f"check/{i}",
    )
    if failed := [r for r in results if not r.ok]:
        return Transition.hold(f"{len(failed)} items failed: {failed[0].error}")
    if ctx.unit.state().get("boom"):
        raise RuntimeError("stage blew up")
    return Transition.end("done", ", ".join(r.value or "" for r in results))


pipeline = Pipeline(
    {"plan": plan, "build": build, "check": check}, start="plan", config=Cfg
)


async def run(root) -> dict[str, int]:
    async with Flow(root, Cfg(), pipeline) as flow:
        return await flow.run()


async def test_units_move_through_stages_and_a_resume_reruns_nothing(tmp_path):
    calls.clear()
    assert await run(tmp_path) == {"terminal": 2}
    t0 = Unit(tmp_path / "tasks" / "t0")
    assert t0.state() | {"reason": ""} == {
        "stage": "check",
        "status": "terminal",
        "outcome": "done",
        "credits": 0,
        "reason": "",
    }
    assert t0.read("out.txt") == "build-t0" and [c["message"] for c in t0.log()] == [
        "check: done",
        "build: built",
        "init",
    ]
    assert Unit(tmp_path / CAMPAIGN).state()["status"] == "waiting"
    events = [
        json.loads(line)
        for line in (tmp_path / "transitions.jsonl").read_text().splitlines()
    ]
    planned = next(
        e for e in events if e["type"] == "transition" and e["unit"] == CAMPAIGN
    )
    assert planned["links"] == [
        {"unit": "t0", "label": "created"},
        {"unit": "t1", "label": "created"},
    ]
    before = len(calls)
    assert (
        await run(tmp_path) == {"terminal": 2} and len(calls) == before
    )  # nothing was runnable


async def test_a_partial_spread_holds_and_a_release_reruns_only_what_failed(tmp_path):
    calls.clear()
    failing.add("item-t1-1")
    counts = await run(tmp_path)
    assert counts == {"terminal": 1, "held": 1}
    t1 = Unit(tmp_path / "tasks" / "t1")
    assert t1.state()["status"] == "held" and "1 items failed" in t1.state()["reason"]
    failing.clear()
    t1.steer(status="ready")
    n = len(calls)
    assert await run(tmp_path) == {"terminal": 2}
    assert calls[n:] == ["item-t1-1"]  # items 0 and 2 attached; only the failed one ran


async def test_a_stage_that_raises_holds_its_unit_with_the_error(tmp_path):
    await run(tmp_path)
    t0 = Unit(tmp_path / "tasks" / "t0")
    t0.commit("route", state={"stage": "check", "status": "ready", "boom": True})
    await run(tmp_path)
    assert (
        t0.state()["status"] == "held"
        and t0.state()["reason"] == "RuntimeError: stage blew up"
    )


async def test_agent_calls_key_on_their_resolved_seat_and_nothing_else(tmp_path):
    task = vf.Task(vf.TaskData(prompt="hi"))
    works = {"alpha": agent("alpha", task), "beta": agent("beta", task), "fn": fn(len)}

    def keys(cfg: FlowConfig) -> dict[str, str]:
        ctx = Ctx(Flow(tmp_path, cfg, pipeline), Unit(tmp_path / "x"), "s")
        return {name: str(w.content(ctx)) for name, w in works.items()}

    base, beta = keys(Cfg()), keys(Cfg(beta=seat(model="beta/2")))
    filled = keys(Cfg(model="run/1"))  # the run's model fills every unpinned seat
    assert {k for k in base if beta[k] != base[k]} == {"beta"}
    assert {k for k in base if filled[k] != base[k]} == {"alpha", "beta"}


async def test_drain_leaves_units_ready_and_a_resume_finishes(tmp_path):
    async def slow(ctx: Ctx) -> Transition:
        ctx.flow.drain()  # a drain request lands while the stage runs
        await ctx.call(
            fn(work, "after-drain"), key="k"
        )  # refused: no call starts under a drain
        return Transition.end("done")

    p = Pipeline(
        {"plan": plan, "build": slow, "check": check}, start="plan", config=Cfg
    )
    async with Flow(tmp_path, Cfg(), p) as flow:
        await flow.run()
    t0 = Unit(tmp_path / "tasks" / "t0")
    assert t0.state() == {"status": "ready", "stage": "build", "credits": 1}
    lines = (tmp_path / "transitions.jsonl").read_text().splitlines()
    kinds = [json.loads(line)["type"] for line in lines]
    assert kinds[:2] == ["started", "transition"] and {"drain", "stopped"} <= set(kinds)
    assert await run(tmp_path) == {"terminal": 2}


def test_unit_reads_are_lossless_and_paths_are_contained(tmp_path):
    text = "  leading\r\ntrailing  \n\n"
    unit = Unit.create(tmp_path / "unit", {"stage": "build"}, {"out.txt": text})
    assert unit.read("out.txt") == unit.read("out.txt", unit.head()) == text
    for rel in ("../escape", ".git/config", "state.json"):
        with pytest.raises(ValueError, match="unsafe|reserved"):
            unit.commit("bad", files={rel: "bad"})
    assert unit.state() == {"status": "ready", "stage": "build"}


async def test_failed_publication_never_schedules_worktree_state(tmp_path, monkeypatch):
    module = importlib.import_module("verifiers.v1.flow.unit")
    called = []

    async def stage(ctx):
        called.append(ctx.stage)
        return Transition.end("done")

    async with Flow(
        tmp_path, Cfg(), Pipeline({"plan": stage, "ghost": stage}, "plan")
    ) as flow:
        real_git = module.git

        def fail_commit(path, *args, **kw):
            if args[0] == "commit":
                raise subprocess.CalledProcessError(1, ["git", "commit"])
            return real_git(path, *args, **kw)

        with monkeypatch.context() as patch:
            patch.setattr(module, "git", fail_commit)
            with pytest.raises(subprocess.CalledProcessError):
                flow.campaign.apply(Transition.to("ghost", "not committed"))
        assert flow.campaign.read_json("state.json")["stage"] == "ghost"
        assert flow.campaign.state()["stage"] == "plan"
        await (
            flow.run()
        )  # the dirty unit parks with a `dirty` line; the run does not die
        with pytest.raises(RuntimeError, match="dirty or incomplete publication"):
            flow.campaign.steer(status="ready")
    events = [
        json.loads(line)
        for line in (tmp_path / "transitions.jsonl").read_text().splitlines()
    ]
    assert [e["type"] for e in events] == ["dirty"] and "dirty or incomplete" in events[
        0
    ]["reason"]
    assert not called


async def test_live_hold_and_route_keep_output_notes_and_application_commits(tmp_path):
    started, finish = asyncio.Event(), asyncio.Event()
    active = set()
    next_stages = []

    async def planning(ctx):
        for name in ("held", "routed"):
            ctx.flow.create_task(name, {"stage": "working"})
        return Transition.wait("planned")

    async def working(ctx):
        active.add(ctx.unit.id)
        if len(active) == 2:
            started.set()
        await finish.wait()
        # Application commits during a stage are legitimate, not stale HEAD conflicts.
        ctx.unit.commit("artifact", files={"artifact.txt": "artifact"})
        return Transition.to(
            "next", "built", files={"out.txt": "saved"}, state={"notes": []}
        )

    async def done(ctx):
        next_stages.append((ctx.unit.id, ctx.stage))
        return Transition.end("done")

    pipeline = Pipeline(
        {"plan": planning, "working": working, "next": done, "alternate": done}, "plan"
    )
    async with Flow(tmp_path, Cfg(), pipeline) as flow:
        running = asyncio.create_task(flow.run())
        await started.wait()
        held, routed = flow.unit("held"), flow.unit("routed")
        held.steer(status="held", reason="inspect output", note="keep this note")
        routed.steer(stage="alternate", status="ready", note="route note")
        routed.steer(note="a later note does not erase the route")
        finish.set()
        assert await running == {"held": 1, "terminal": 1}
        assert held.state()["stage"] == "next"
        assert held.state()["reason"] == "inspect output"
        assert held.state()["outcome"] == "built"
        assert held.state()["notes"] == ["keep this note"]
        assert held.read("out.txt", "HEAD") == "saved"
        assert held.read("artifact.txt", "HEAD") == "artifact"
        assert next_stages == [("routed", "alternate")]
        assert len(routed.state()["notes"]) == 2
        assert not succeeded(flow.campaign, flow.tasks())
        await flow._stage(held)  # a hold after admission but before start still wins
        assert held.state()["status"] == "held"
        held.steer(status="ready")
        assert await flow.run() == {"terminal": 2}
        assert succeeded(flow.campaign, flow.tasks())


async def test_spread_drain_settles_admitted_children_and_does_not_retry(tmp_path):
    admitted, drained, finish = asyncio.Event(), asyncio.Event(), asyncio.Event()
    ran = []

    async def slow():
        admitted.set()
        await finish.wait()
        ran.append("slow")
        return "saved"

    async def stage(ctx):
        async def stop():
            await admitted.wait()
            ctx.flow.drain()
            drained.set()
            return "drained"

        await ctx.spread(
            [fn(slow), fn(stop), fn(work, "not-admitted")], key=lambda i: str(i)
        )
        return Transition.end("unreachable")

    calls.clear()
    async with Flow(tmp_path, Cfg(), Pipeline({"start": stage}, "start")) as flow:
        running = asyncio.create_task(flow.run())
        await drained.wait()
        assert not running.done()
        finish.set()
        await running
        assert ran == ["slow"] and "not-admitted" not in calls
        assert len(list((tmp_path / "calls" / CAMPAIGN).glob("*.json"))) == 2
        assert flow.campaign.state()["status"] == "ready"

    retries = []

    async def retrying(ctx):
        async def fail():
            retries.append(1)
            ctx.flow.drain()
            raise ValueError("failed while draining")

        await ctx.call(fn(fail), retries=3)
        return Transition.end("unreachable")

    async with Flow(tmp_path, Cfg(), Pipeline({"start": retrying}, "start")) as flow:
        await flow.run()
        assert retries == [1]
        assert flow.campaign.state()["status"] == "ready"


async def test_cancel_joins_stages_and_spread_before_serving_exits(tmp_path):
    from contextlib import asynccontextmanager

    entered, settled = asyncio.Event(), []

    async def child(index):
        try:
            entered.set()
            await asyncio.Event().wait()
        finally:
            settled.append(index)

    async def stage(ctx):
        await ctx.spread([fn(child, 0), fn(child, 1)])
        return Transition.end("unreachable")

    async def planning(ctx):
        ctx.flow.create_task("child", {"stage": "work"})
        return Transition.wait("planned")

    class LocalFlow(Flow):
        @asynccontextmanager
        async def _serving(self):
            try:
                yield
            finally:
                assert sorted(settled) == [0, 1]

    async with LocalFlow(
        tmp_path, Cfg(), Pipeline({"plan": planning, "work": stage}, "plan")
    ) as flow:
        running = asyncio.create_task(flow.run())
        await entered.wait()
        running.cancel()
        with pytest.raises(asyncio.CancelledError):
            await running
        assert sorted(settled) == [0, 1]


async def test_failed_startup_releases_lock_and_held_campaign_is_not_success(
    tmp_path, monkeypatch
):
    from verifiers.v1.flow.__main__ import status

    module = importlib.import_module("verifiers.v1.flow.flow")
    flow = Flow(tmp_path, Cfg(), pipeline)
    with monkeypatch.context() as patch:
        patch.setattr(
            module,
            "trim_torn_tail",
            lambda _: (_ for _ in ()).throw(ValueError("startup")),
        )
        with pytest.raises(ValueError, match="startup"):
            await flow.__aenter__()
    assert flow._lock is None
    async with Flow(tmp_path, Cfg(), pipeline) as resumed:
        resumed.campaign.steer(status="held")
        assert await resumed.run() == {}
        assert not succeeded(resumed.campaign, resumed.tasks())
        assert status(tmp_path) == 1
        with pytest.raises(ValueError, match="task id"):
            resumed.create_task("../outside", {"stage": "build"})


async def test_flywheel_caches_build_and_lint_as_one_operation(tmp_path, monkeypatch):
    from contextlib import asynccontextmanager
    from types import SimpleNamespace

    from examples.flow.flywheel import BuildAndLint
    from verifiers.v1.flow import AgentWork, CallFailed
    from verifiers.v1.flow.calls import CommandWork
    from verifiers.v1.runtimes import ProgramResult

    class BuildConfig(Cfg):
        builder: vf.AgentConfig = seat()

    boxes = []
    built = []

    @asynccontextmanager
    async def runtime(ctx, seat, task=None):
        box = {}
        boxes.append(box)
        yield box

    async def build(work, ctx, name):
        work.runtime["code"] = "fresh source"
        built.append(work.runtime)
        return SimpleNamespace(id="builder-trace", last_reply="built")

    async def lint(work, ctx, name):
        assert work.runtime["code"] == "fresh source"
        if len(boxes) == 1:
            raise ValueError("lint transport failed")
        return ProgramResult(exit_code=0, stdout="", stderr="")

    monkeypatch.setattr(Ctx, "runtime", runtime)
    monkeypatch.setattr(AgentWork, "execute", build)
    monkeypatch.setattr(CommandWork, "execute", lint)
    async with Flow(tmp_path, BuildConfig(), pipeline) as flow:
        ctx = Ctx(flow, flow.campaign, "build")
        with pytest.raises(CallFailed, match="lint transport failed"):
            await ctx.call(BuildAndLint("brief"), key="build/0")
        value = await ctx.call(BuildAndLint("brief"), key="build/0")
        assert value == {"reply": "built", "lint": 0}
        flow.campaign.steer(status="held", note="ordinary operator commit")
        flow.campaign.steer(status="ready")
        assert await ctx.call(BuildAndLint("brief"), key="build/0") == value
        assert (
            len(boxes) == len(built) == 2
        )  # retry rebuilds; cached success needs no box


async def test_agent_calls_survive_a_budget_change_and_find_older_records(tmp_path):
    task = vf.Task(vf.TaskData(prompt="hi"))
    work_ = agent("alpha", task)

    def ctx(cfg: FlowConfig) -> Ctx:
        return Ctx(Flow(tmp_path, cfg, pipeline), Unit(tmp_path / "x"), "s")

    base = ctx(Cfg())
    longer = ctx(Cfg(alpha=seat(timeout=vf.agent.TimeoutConfig(rollout=7200))))
    retried = ctx(Cfg(alpha=seat(retries=vf.RetryConfig(max_retries=3))))
    assert work_.content(longer) == work_.content(base) == work_.content(retried)
    assert work_.content(ctx(Cfg(alpha=seat(model="alpha/2")))) != work_.content(base)
    # The identity an older launch used carried the whole seat; it is still looked up.
    (legacy,) = work_.legacy_contents(longer)
    assert legacy != work_.content(longer)
    assert legacy[-1]["timeout"]["rollout"] == 7200


async def test_attach_by_key_reuses_a_record_written_under_another_identity(tmp_path):
    from verifiers.v1.flow.calls import Record

    calls.clear()
    failing.clear()
    cfg = Cfg(attach_by_key=True)
    async with Flow(tmp_path, cfg, pipeline) as flow:
        unit = Unit(tmp_path / "campaign")
        ctx = Ctx(flow, unit, "s")
        stale = tmp_path / "calls" / unit.id / ("0" * 24 + ".json")
        stale.parent.mkdir(parents=True, exist_ok=True)
        stale.write_text(
            Record(
                key="build",
                unit=unit.id,
                stage="s",
                kind="fn",
                started_at="2026-01-01T00:00:00+00:00",
                finished_at="2026-01-01T00:00:01+00:00",
                payload="from-before",
            ).model_dump_json()
        )
        assert await ctx.call(fn(work, "build-x"), key="build") == "from-before"
        assert calls == []  # attached by key; nothing ran
    async with Flow(tmp_path, Cfg(), pipeline) as flow:
        ctx = Ctx(flow, Unit(tmp_path / "campaign"), "s")
        assert await ctx.call(fn(work, "build-x"), key="build") == "build-x"
        assert calls == [
            "build-x"
        ]  # the switch off: an unmatched identity runs the work
