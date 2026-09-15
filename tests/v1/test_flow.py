"""The flow core on host functions and subprocess commands: memoized steps, scopes,
retries, spreads, runtime scopes, drain, streaming, what a resume keys
on, and the event stream."""

import asyncio

import verifiers.v1 as vf
from verifiers.v1.flow import (
    Ctx,
    FlowConfig,
    Ledger,
    Run,
    StepRecord,
    agent,
    command,
    fn,
)
from verifiers.v1.flow.ledger import now, row_key


def config(**kw) -> FlowConfig:
    return FlowConfig(**kw)


def one() -> int:
    return 1


def two() -> int:
    return 2


def seat(**kw) -> vf.AgentConfig:
    return vf.AgentConfig(harness={"id": "null"}, runtime=vf.SubprocessConfig(), **kw)


class ToyTask(vf.Task[vf.TaskData]):
    """The smallest task an `agent(...)` step can carry."""


async def test_steps_attach_on_resume_and_scopes_separate_repeats(tmp_path):
    calls: list[str] = []

    def work(tag: str) -> str:
        calls.append(tag)
        return tag

    async def flow(ctx: Ctx, row) -> list[str]:
        out = [await ctx.step("start", fn(work, "start"))]
        for i in range(2):
            with ctx.scope(f"visit{i}"):
                out.append(await ctx.step("build", fn(work, f"build{i}")))
        out.append(await ctx.step("build", fn(work, "again")))
        return out

    (first,) = await Run(tmp_path, config()).run(flow, [{"id": 1}])
    assert first.state == "ok" and first.value == ["start", "build0", "build1", "again"]
    (again,) = await Run(tmp_path, config()).run(flow, [{"id": 1}])
    assert again.value == first.value and len(calls) == 4


async def test_operational_knobs_and_policy_fields_never_rekey_a_resume(tmp_path):
    calls: list[str] = []

    class Cfg(FlowConfig):
        rounds: int = 3
        """A pipeline's own policy knob, outside the engine's fields."""

    def work(tag: str) -> str:
        calls.append(tag)
        return tag

    async def flow(ctx: Ctx, row) -> list[str]:
        return [await ctx.step("a", fn(work, "a")), await ctx.step("b", fn(work, "b"))]

    run_dir = tmp_path / "run"
    (first,) = await Run(run_dir, Cfg()).run(flow, [{"id": 1}])
    assert first.state == "ok" and first.value == ["a", "b"]
    (again,) = await Run(
        run_dir,
        Cfg(
            rounds=8,
            max_concurrent_rows=1,
            pools={"runtimes": 1},
            payload_cap=2048,
        ),
    ).run(flow, [{"id": 1}])
    assert again.state == "ok" and again.value == first.value and calls == ["a", "b"]


async def test_agent_steps_key_on_their_resolved_seat_and_nothing_else(tmp_path):
    class Cfg(FlowConfig):
        alpha: vf.AgentConfig = seat()
        beta: vf.AgentConfig = seat()

    task = ToyTask(vf.TaskData(prompt="hi"))

    def keys(cfg: FlowConfig) -> dict[str, str]:
        ctx = Ctx(Run(tmp_path / "run", cfg), "k", {"id": 1})
        return {
            "alpha": ctx._key("p#0", agent("alpha", task)),
            "beta": ctx._key("p#0", agent("beta", task)),
            "fn": ctx._key("p#0", fn(one)),
        }

    base = keys(Cfg())
    only_beta = keys(Cfg(beta=seat(model="beta/2")))
    assert only_beta["alpha"] == base["alpha"] and only_beta["fn"] == base["fn"]
    assert only_beta["beta"] != base["beta"]
    defaulted = keys(Cfg(model="run/1"))  # the run's model fills the unpinned seats
    assert defaulted["fn"] == base["fn"]
    assert defaulted["alpha"] != base["alpha"] and defaulted["beta"] != base["beta"]


async def test_a_seat_model_change_re_runs_only_that_seats_agent_steps(tmp_path):
    class Cfg(FlowConfig):
        alpha: vf.AgentConfig = seat()

    calls: list[str] = []

    def work(tag: str) -> str:
        calls.append(tag)
        return tag

    task = ToyTask(vf.TaskData(prompt="hi"))

    async def flow(ctx: Ctx, row) -> list[str]:
        prep = await ctx.step("prep", fn(work, "prep"))
        return [prep, (await ctx.step("rollout", agent("alpha", task))).id]

    run_dir = tmp_path / "run"
    run = Run(run_dir, Cfg())
    key = row_key({"id": 1})
    trace = vf.Trace(
        task=vf.TraceTask(type="ToyTask", data=task.data),
        agent=vf.AgentInfo(config=run.seat("alpha")),
    )
    await run.ledger.append(trace)
    run.ledger.put(
        StepRecord(
            key=Ctx(run, key, {"id": 1})._key("rollout#0", agent("alpha", task)),
            row=key,
            path="rollout#0",
            kind="agent",
            terminal="completed",
            started_at=now(),
            finished_at=now(),
            trace_id=trace.id,
        )
    )
    (first,) = await run.run(flow, [{"id": 1}])
    assert (
        first.state == "ok" and first.value == ["prep", trace.id] and calls == ["prep"]
    )

    moved = Run(run_dir, Cfg(alpha=seat(model="alpha/2")))
    ctx = Ctx(moved, key, {"id": 1})
    assert ctx._attached("prep#0", fn(work, "prep"), None) is not None
    assert ctx._attached("rollout#0", agent("alpha", task), None) is None


async def test_turn_failures_consume_retries_and_fail_only_their_row(tmp_path):
    attempts: list[int] = []

    def flaky() -> None:
        attempts.append(1)
        raise ValueError("no")

    async def flow(ctx: Ctx, row) -> int:
        if row["id"] == 2:
            return await ctx.step("fine", fn(two))
        return await ctx.step("flaky", fn(flaky), retries=2)

    results = await Run(tmp_path, config()).run(flow, [{"id": 1}, {"id": 2}])
    failed = [r for r in results if r.state == "failed"]
    assert (
        len(attempts) == 3 and len(failed) == 1 and "ValueError: no" in failed[0].error
    )
    assert [r.value for r in results if r.state == "ok"] == [2]


async def test_a_step_timeout_fails_the_step(tmp_path):
    async def slow() -> None:
        await asyncio.sleep(1)

    async def flow(ctx: Ctx, row) -> None:
        await ctx.step("slow", fn(slow), timeout=0.01)

    (result,) = await Run(tmp_path, config()).run(flow, [{"id": 1}])
    assert result.state != "ok" and "TimeoutError" in result.error


async def test_spread_runs_every_item_and_a_resume_attaches_them(tmp_path):
    started: list[int] = []

    async def item(i: int) -> int:
        started.append(i)
        return i * i

    async def flow(ctx: Ctx, row) -> dict[int, int]:
        return await ctx.spread("sq", [fn(item, i) for i in range(3)])

    (first,) = await Run(tmp_path, config()).run(flow, [{"id": 1}])
    assert first.state == "ok" and first.value == {0: 0, 1: 1, 2: 4}
    (again,) = await Run(tmp_path, config()).run(flow, [{"id": 1}])
    assert again.value == first.value and sorted(started) == [0, 1, 2]


async def test_a_spread_fails_when_any_item_does(tmp_path):
    def item(i: int) -> int:
        if i == 1:
            raise ValueError("bad item")
        return i

    async def flow(ctx: Ctx, row) -> dict[int, int]:
        return await ctx.spread("sq", [fn(item, i) for i in range(3)])

    (result,) = await Run(tmp_path, config()).run(flow, [{"id": 1}])
    assert result.state == "failed" and "1/3 items failed" in result.error


async def test_commands_share_a_runtime_scope_and_attach_on_resume(tmp_path):
    marker = tmp_path / "marker.txt"

    class Cfg(FlowConfig):
        worker: vf.AgentConfig = vf.AgentConfig(
            harness={"id": "null"}, runtime=vf.SubprocessConfig()
        )

    async def flow(ctx: Ctx, row) -> tuple[str, int]:
        async with ctx.runtime("worker") as box:
            await ctx.step(
                "write", command(["sh", "-c", f"echo shared > {marker}"], runtime=box)
            )
            read = await ctx.step("read", command(["cat", str(marker)], runtime=box))
        return read.stdout.strip(), read.exit_code

    (first,) = await Run(tmp_path / "run", Cfg()).run(flow, [{"id": 1}])
    assert first.state == "ok" and first.value == ("shared", 0)
    marker.unlink()
    (again,) = await Run(tmp_path / "run", Cfg()).run(flow, [{"id": 1}])
    assert again.value == ("shared", 0)  # both commands attached; nothing ran


async def test_drain_stops_before_the_next_step_and_a_resume_finishes(tmp_path):
    async def flow(ctx: Ctx, row) -> int:
        a = await ctx.step("a", fn(one))
        ctx.run.drain()
        return a + await ctx.step("b", fn(two))

    async def undrained(ctx: Ctx, row) -> int:
        return await ctx.step("a", fn(one)) + await ctx.step("b", fn(two))

    run = Run(tmp_path, config())
    (stopped,) = await run.run(flow, [{"id": 1}])
    assert stopped.state == "stopped" and run.rows == {stopped.row: "stopped"}
    (done,) = await Run(tmp_path, config()).run(undrained, [{"id": 1}])
    assert done.state == "ok" and done.value == 3
    assert (
        run.ledger.get(done.row, "a#0") is not None
        and run.ledger.get(done.row, "b#0") is not None
    )


async def test_concurrent_branches_keep_their_own_scopes(tmp_path):
    calls: list[str] = []

    def work(tag: str) -> str:
        calls.append(tag)
        return tag

    async def flow(ctx: Ctx, row) -> list[str]:
        async def branch(label: str) -> str:
            with ctx.scope(label):
                await asyncio.sleep(0.01 if label == "a" else 0)
                return await ctx.step("x", fn(work, label))

        return list(await asyncio.gather(branch("a"), branch("b")))

    (first,) = await Run(tmp_path, config()).run(flow, [{"id": 1}])
    (again,) = await Run(tmp_path, config()).run(flow, [{"id": 1}])
    assert first.value == again.value == ["a", "b"] and sorted(calls) == ["a", "b"]


async def test_stream_yields_rows_as_they_finish(tmp_path):
    async def flow(ctx: Ctx, row) -> float:
        await asyncio.sleep(row["t"])
        return row["t"]

    order = [
        r.value
        async for r in Run(tmp_path, config()).stream(flow, [{"t": 0.05}, {"t": 0.0}])
    ]
    assert order == [0.0, 0.05]


async def test_events_stream_each_step_including_the_ones_in_flight(tmp_path):
    release = asyncio.Event()

    async def blocked() -> int:
        await release.wait()
        return 1

    async def flow(ctx: Ctx, row) -> int:
        return await ctx.step("long", fn(blocked))

    run = Run(tmp_path, config())
    task = asyncio.create_task(run.run(flow, [{"id": 1}]))
    for _ in range(100):
        if any(e["type"] == "step_started" for e in run.ledger.events()):
            break
        await asyncio.sleep(0.01)
    kinds = [e["type"] for e in run.ledger.events()]
    assert kinds == ["run", "row_started", "step_started"]  # started, not finished
    release.set()
    (result,) = await task
    events = run.ledger.events()
    assert result.state == "ok" and [e["type"] for e in events[3:]] == [
        "step_completed",
        "row_finished",
    ]
    assert events[2]["row"] == result.row and events[2]["path"] == "long#0"
    assert all("at" in e for e in events) and events[0]["label"] == run.label


async def test_a_resume_emits_attached_events_and_a_torn_last_line_is_skipped(tmp_path):
    async def flow(ctx: Ctx, row) -> list[int]:
        return [await ctx.step("a", fn(one)), await ctx.step("b", fn(two))]

    (first,) = await Run(tmp_path, config()).run(flow, [{"id": 1}])
    with (tmp_path / "events.jsonl").open("a") as file:
        file.write('{"type": "step_star')  # a SIGKILL mid-write
    before = len(Ledger(tmp_path).events())  # a live reader skips the torn tail ...
    (again,) = await Run(tmp_path, config()).run(
        flow, [{"id": 1}]
    )  # ... a new run drops it
    assert again.state == "ok" and again.value == first.value == [1, 2]
    resumed = Ledger(tmp_path).events()[before:]
    assert [e["type"] for e in resumed] == [
        "run",
        "row_started",
        "step_attached",
        "step_attached",
        "row_finished",
    ]
    assert [e["path"] for e in resumed if e["type"] == "step_attached"] == [
        "a#0",
        "b#0",
    ]


async def test_an_oversized_step_value_fails_at_once(tmp_path):
    calls: list[int] = []

    def big() -> str:
        calls.append(1)
        return "x" * 100

    async def flow(ctx: Ctx, row) -> str:
        return await ctx.step("big", fn(big), retries=3)

    (result,) = await Run(tmp_path, config(payload_cap=50)).run(flow, [{"id": 1}])
    assert (
        result.state != "ok" and "over payload_cap" in result.error and len(calls) == 1
    )


async def test_sweep_kills_the_subprocesses_a_dead_launch_left(tmp_path):
    import os

    from verifiers.v1.runtimes.subprocess import RUN_LABEL_VAR, sweep_subprocesses

    run = Run(tmp_path, config())
    orphan = await asyncio.create_subprocess_exec(
        "sleep",
        "600",
        env={**os.environ, RUN_LABEL_VAR: run.label},
        start_new_session=True,
    )
    try:
        await asyncio.sleep(0.2)
        assert sweep_subprocesses(run.label) >= 1
        assert await asyncio.wait_for(orphan.wait(), 5) != 0
    finally:
        if orphan.returncode is None:
            orphan.kill()
