"""The flow core on host functions and subprocess commands: memoized steps, scopes,
retry classification, spreads, runtime scopes, drain, and streaming."""

import asyncio

import verifiers.v1 as vf
from verifiers.v1.errors import ProviderError, SandboxError
from verifiers.v1.flow import Ctx, FlowConfig, Run, command, fn


def config(**kw) -> FlowConfig:
    kw.setdefault("outage_backoff_s", 0.01)
    kw.setdefault("outage_hold_s", 0.5)
    return FlowConfig(**kw)


def one() -> int:
    return 1


def two() -> int:
    return 2


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
    assert first.ok and first.value == ["start", "build0", "build1", "again"]
    (again,) = await Run(tmp_path, config()).run(flow, [{"id": 1}])
    assert again.value == first.value and len(calls) == 4


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
    failed = [r for r in results if not r.ok]
    assert (
        len(attempts) == 3 and len(failed) == 1 and "ValueError: no" in failed[0].error
    )
    assert [r.value for r in results if r.ok] == [2]


async def test_infrastructure_failures_hold_and_retry_without_consuming_retries(
    tmp_path,
):
    seen = 0

    def flapping_box() -> int:
        nonlocal seen
        seen += 1
        if seen < 3:
            raise SandboxError("box gone")
        return seen

    async def flow(ctx: Ctx, row) -> int:
        return await ctx.step("s", fn(flapping_box))

    run = Run(tmp_path, config())
    (result,) = await run.run(flow, [{"id": 1}])
    assert result.ok and result.value == 3 and not run.status()["holding"]


async def test_infrastructure_hold_budget_and_timeouts_fail_the_step(tmp_path):
    def down() -> None:
        raise SandboxError("down")

    async def slow() -> None:
        await asyncio.sleep(1)

    async def flow(ctx: Ctx, row) -> None:
        if row["id"] == 1:
            await ctx.step("down", fn(down))
        await ctx.step("slow", fn(slow), timeout=0.01)

    results = await Run(tmp_path, config(outage_hold_s=0.05)).run(
        flow, [{"id": 1}, {"id": 2}]
    )
    errors = sorted(r.error for r in results)
    assert all("held 0.05s" in e for e in errors)
    assert any("SandboxError: down" in e for e in errors) and any(
        "TimeoutError" in e for e in errors
    )


async def test_permanent_failures_stop_at_once(tmp_path):
    calls: list[int] = []

    def denied() -> None:
        calls.append(1)
        raise ProviderError("forbidden", status_code=403)

    async def flow(ctx: Ctx, row) -> None:
        await ctx.step("s", fn(denied), retries=3)

    (result,) = await Run(tmp_path, config()).run(flow, [{"id": 1}])
    assert not result.ok and len(calls) == 1


async def test_spread_starts_only_what_the_quorum_needs_and_resumes_the_same_quorum(
    tmp_path,
):
    started: list[int] = []

    async def item(i: int) -> int:
        started.append(i)
        await asyncio.sleep(0.01 * (i + 1))
        if i == 0:
            raise ValueError("bad item")
        return i * i

    async def flow(ctx: Ctx, row) -> dict[int, int]:
        return await ctx.spread("sq", [fn(item, i) for i in range(5)], at_least=2)

    (first,) = await Run(tmp_path, config()).run(flow, [{"id": 1}])
    assert first.ok and first.value == {1: 1, 2: 4} and sorted(started) == [0, 1, 2]
    (again,) = await Run(tmp_path, config()).run(flow, [{"id": 1}])
    assert again.value == first.value and len(started) == 3


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

    (first,) = await Run(tmp_path / "run", Cfg(outage_backoff_s=0.01)).run(
        flow, [{"id": 1}]
    )
    assert first.ok and first.value == ("shared", 0)
    marker.unlink()
    (again,) = await Run(tmp_path / "run", Cfg(outage_backoff_s=0.01)).run(
        flow, [{"id": 1}]
    )
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
    assert (
        stopped.stopped
        and not stopped.ok
        and run.status()["rows"] == {stopped.row: "stopped"}
    )
    (done,) = await Run(tmp_path, config()).run(undrained, [{"id": 1}])
    assert done.ok and done.value == 3
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
