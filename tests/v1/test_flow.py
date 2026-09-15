"""The flow core: memoized steps, what a resume keys on, retries, spreads, drain, events."""

import verifiers.v1 as vf
from verifiers.v1.flow import Ctx, FlowConfig, Run, agent, fn


def seat(**kw) -> vf.AgentConfig:
    return vf.AgentConfig(harness={"id": "null"}, runtime=vf.SubprocessConfig(), **kw)


class Cfg(FlowConfig):
    alpha: vf.AgentConfig = seat()
    beta: vf.AgentConfig = seat()


async def test_steps_attach_on_resume_under_scopes_and_config_changes(tmp_path):
    calls: list[str] = []

    def work(tag: str) -> str:
        calls.append(tag)
        return tag

    async def flow(ctx: Ctx, row) -> list[str]:
        out = [await ctx.step("start", fn(work, "start"))]
        for i in range(2):
            with ctx.scope(f"visit{i}"):
                out.append(await ctx.step("build", fn(work, f"build{i}")))
        return [*out, await ctx.step("build", fn(work, "again"))]

    (first,) = await Run(tmp_path, Cfg()).run(flow, [{"id": 1}])
    changed = Cfg(max_concurrent_rows=1, pools={"runtimes": 1})
    (again,) = await Run(tmp_path, changed).run(flow, [{"id": 1}])
    assert first.value == again.value == ["start", "build0", "build1", "again"]
    assert len(calls) == 4  # the resume ran nothing


def test_agent_steps_key_on_their_resolved_seat_and_nothing_else(tmp_path):
    task = vf.Task(vf.TaskData(prompt="hi"))
    works = {"alpha": agent("alpha", task), "beta": agent("beta", task), "fn": fn(len)}

    def keys(cfg: FlowConfig) -> dict[str, str]:
        ctx = Ctx(Run(tmp_path, cfg), "k", {"id": 1})
        return {name: ctx._key("p#0", work) for name, work in works.items()}

    base, beta = keys(Cfg()), keys(Cfg(beta=seat(model="beta/2")))
    filled = keys(Cfg(model="run/1"))  # the run's model fills every unpinned seat
    assert {k for k in base if beta[k] != base[k]} == {"beta"}
    assert {k for k in base if filled[k] != base[k]} == {"alpha", "beta"}


async def test_failures_consume_retries_and_fail_only_their_row(tmp_path):
    attempts: list[int] = []

    def flaky() -> None:
        attempts.append(1)
        raise ValueError("no")

    async def flow(ctx: Ctx, row) -> int:
        if row["id"] == 1:
            return await ctx.step("flaky", fn(flaky), retries=2)
        return await ctx.step("fine", fn(len, "ab"))

    results = await Run(tmp_path, Cfg()).run(flow, [{"id": 1}, {"id": 2}])
    assert sorted(r.state for r in results) == ["failed", "ok"] and len(attempts) == 3
    assert any("flaky#0: ValueError: no" in (r.error or "") for r in results)


async def test_spread_runs_every_item_and_fails_when_one_does(tmp_path):
    def item(i: int, bad: int) -> int:
        if i == bad:
            raise ValueError("bad item")
        return i * i

    async def flow(ctx: Ctx, row) -> dict[int, int]:
        return await ctx.spread("sq", [fn(item, i, row["bad"]) for i in range(3)])

    (ok,) = await Run(tmp_path, Cfg()).run(flow, [{"bad": -1}])
    (failed,) = await Run(tmp_path, Cfg()).run(flow, [{"bad": 1}])
    assert ok.value == {0: 0, 1: 1, 2: 4}
    assert failed.state == "failed" and "1/3 items failed" in failed.error


async def test_drain_stops_a_row_and_a_resume_finishes_it_with_events_to_show(tmp_path):
    draining = True

    async def flow(ctx: Ctx, row) -> int:
        a = await ctx.step("a", fn(len, "a"))
        if draining:
            ctx.run.drain()
        return a + await ctx.step("b", fn(len, "bb"))

    run = Run(tmp_path, Cfg())
    (stopped,) = await run.run(flow, [{"id": 1}])
    assert stopped.state == "stopped" and run.rows == {stopped.row: "stopped"}
    draining = False
    (done,) = await Run(tmp_path, Cfg()).run(flow, [{"id": 1}])
    first = "run row_started step_started step_completed drain row_finished"
    resume = "run row_started step_attached step_started step_completed row_finished"
    kinds = [e["type"] for e in run.ledger.events()]
    assert done.value == 3 and kinds == (first + " " + resume).split()
