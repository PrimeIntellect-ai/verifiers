"""The flow core: units and transitions, durable calls, spreads, holds, the campaign, drain."""

import json

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
    value = await ctx.call(
        fn(work, f"build-{ctx.unit.id}"), key=f"build/{ctx.unit.head()}"
    )
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
    t1.commit("release", state={"status": "ready"})  # the operator's commit
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
