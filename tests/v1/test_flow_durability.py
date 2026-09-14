"""Engine durability: cycles, quorums, resume, cache identity, crash isolation.

Model-free: fn/run nodes plus, for expand, a fabricated rollout (the ledger and join
bookkeeping are what is under test). Each test names the failure it pins.
"""

from __future__ import annotations

import asyncio
import tempfile
from pathlib import Path
from typing import Any, Literal

import pytest
from pydantic import BaseModel

import verifiers.v1 as vf
from verifiers.v1.flow import (
    ALL,
    END,
    Engine,
    Flow,
    FlowConfig,
    FlowError,
    NodeRecord,
    Upstream,
    at_least,
    expand,
    fn,
    run,
)
from verifiers.v1.flow.engine import _Row


def noop(up: Upstream) -> None:
    return None


async def arun(flow: Flow, rows: Any, run_dir: Path | None = None) -> list[Any]:
    """Run a flow over rows in a fresh (or given) run dir."""
    with tempfile.TemporaryDirectory() as d:
        engine = Engine(flow, Path(run_dir or d))
        return await engine.run(rows)


# -- cycle entry and re-entry -----------------------------------------------------


async def test_cycle_entry_node_with_only_backedge_preds_fires():
    """A node whose predecessors are all back-edges (a cycle back through it) must
    fire on first activation — including the flow entry node."""

    calls: list[str] = []

    def draft_fn(up: Upstream) -> str:
        calls.append("draft")
        return "ok"

    def review_fn(up: Upstream) -> Literal["revise", "accept"]:
        calls.append("review")
        return "revise" if calls.count("draft") < 2 else "accept"

    class Revise(Flow):
        draft = fn(draft_fn, then="review", max_visits=3)
        review = fn(review_fn, outcomes={"revise": "draft", "accept": END}, join=at_least(1))

    (result,) = await arun(Revise(), [{"id": 1}])
    assert result.ok, result.error
    assert calls == ["draft", "review", "draft", "review"], calls
    assert result.records["draft"].visit == 2
    assert result.records["review"].visit == 2


async def test_unbounded_entry_node_on_cycle_fires():
    visits = {"x": 0, "b": 0}

    def x_fn(up: Upstream) -> Literal["again", "done"]:
        visits["x"] += 1
        return "again" if visits["x"] < 2 else "done"

    def b_fn(up: Upstream) -> Literal["again", "done"]:
        visits["b"] += 1
        return "again"

    class Cycle(Flow):
        x = fn(x_fn, outcomes={"again": "b", "done": END}, join=at_least(1))
        b = fn(b_fn, outcomes={"again": "x", "done": END}, max_visits=3)

    (result,) = await arun(Cycle(), [{"id": 1}])
    assert result.ok, result.error
    assert visits == {"x": 2, "b": 1}


async def test_self_loop_entry_node_fires():
    count = [0]

    def unstable(up: Upstream) -> str:
        count[0] += 1
        if count[0] < 3:
            raise RuntimeError("transient")
        return "ok"

    class SelfLoop(Flow):
        a = fn(unstable, then=END, max_visits=5, on_error="a")

    (result,) = await arun(SelfLoop(), [{"id": 1}])
    assert result.ok, result.error
    assert result.records["a"].terminal == "completed"
    assert count[0] == 3


async def test_cycle_backedge_retriggers_without_joining():
    """Staged semantics: a bounded node's join is over its NON-cycle predecessors;
    a back-edge predecessor firing re-triggers it (visit increments)."""
    visits: list[Any] = []

    def gate_fn(up: Upstream) -> Literal["again", "done"]:
        visits.append(("gate", up.outcome("slow")))
        return "again" if len(visits) < 2 else "done"

    class Gate(Flow):
        start = fn(lambda up: 1, then=("slow", "fan"))
        slow = fn(lambda up: "slow", then="gate")
        fan = fn(lambda up: "fan", then="gate")
        gate = fn(gate_fn, outcomes={"again": ("slow",), "done": END}, max_visits=3)

    (result,) = await arun(Gate(), [{"id": 1}])
    assert result.ok, result.error
    # gate@1 joined over fan only (slow is a back-edge); slow's firing re-triggered it
    assert result.records["gate"].visit == 2
    assert [v[1] for v in visits] == [None, "completed"]


# -- dead predecessors and rejection gates ----------------------------------------


async def test_dead_predecessor_is_not_awaited():
    class Gate(Flow):
        start = fn(lambda up: 1, then=("good", "bad"))
        good = fn(lambda up: "ok", outcomes={"ok": END})
        bad = fn(lambda up: "reject", outcomes={"reject": "gate"})
        gate = fn(lambda up: up.outcome("bad"))

    (result,) = await arun(Gate(), [{"id": 1}])
    assert result.ok, result.error
    # good ENDed its branch without firing: dead, not awaited
    assert result.records["gate"].payload == "reject"


async def test_gate_waits_while_predecessor_still_live():
    async def slow_fn(up: Upstream) -> int:
        await asyncio.sleep(0.05)
        return 41

    class Gate(Flow):
        start = fn(lambda up: 1, then=("slow", "fast"))
        slow = fn(slow_fn, then="gate")
        fast = fn(lambda up: "fast", then="gate")
        gate = fn(lambda up: [up.slow, up.fast])

    (result,) = await arun(Gate(), [{"id": 1}])
    assert result.ok, result.error
    assert result.records["gate"].payload == [41, "fast"]


@pytest.mark.xfail(
    reason="PROPOSAL (not landed): error records keep outcome=None while "
    "exhausted records carry outcome='exhausted'; landing outcome='error' would "
    "make downstream up.outcome() distinguish error from not-fired, at a small "
    "compatibility cost for code reading None today",
    strict=True,
)
async def test_error_record_carries_error_outcome():
    def boom(up: Upstream) -> str:
        raise RuntimeError("kaboom")

    class Fail(Flow):
        a = fn(boom, retries=1, on_error="b")
        b = fn(lambda up: up.outcome("a"))

    (result,) = await arun(Fail(), [{"id": 1}])
    assert result.ok, result.error
    assert result.records["a"].terminal == "error"
    assert result.records["a"].attempt == 2
    assert result.records["b"].payload == "error"


# -- dynamic quorum ----------------------------------------------------------------


async def test_at_least_callable_quorum_reads_upstream():
    def quorum(up: Upstream) -> int:
        return 2 if up.config.need_two else 1

    class QuorumCfg(FlowConfig):
        need_two: bool = False

    class Quorum(Flow[QuorumCfg]):
        start = fn(lambda up: 1, then=("a", "b"))
        a = fn(lambda up: "a", then="pick")
        b = fn(lambda up: "b", then="pick")
        pick = fn(
            lambda up: sorted(n for n in ("a", "b") if up.outcome(n)),
            join=at_least(quorum),
        )

    (fast,) = await arun(Quorum(QuorumCfg(need_two=False)), [{"id": 1}])
    assert fast.ok, fast.error
    (full,) = await arun(Quorum(QuorumCfg(need_two=True)), [{"id": 1}])
    assert full.ok, full.error
    assert full.records["pick"].payload == ["a", "b"]


async def test_late_arrival_retriggers_the_join():
    """Current semantics: a straggler predecessor firing re-runs an at_least join
    node (its visit increments). Pinned here; any change is a semantic decision."""

    async def slow_fn(up: Upstream) -> str:
        await asyncio.sleep(0.05)
        return "slow"

    class Late(Flow):
        start = fn(lambda up: 1, then=("slow", "fast"))
        slow = fn(slow_fn, then="gate")
        fast = fn(lambda up: "fast", then="gate")
        gate = fn(
            lambda up: [up.outcome("slow"), up.outcome("fast")],
            then="after",
            join=at_least(1),
        )
        after = fn(lambda up: "after")

    (result,) = await arun(Late(), [{"id": 1}])
    assert result.ok, result.error
    assert result.records["gate"].visit == 2
    assert result.records["after"].visit == 2


# -- pools --------------------------------------------------------------------------


async def test_fn_node_pools_are_held():
    """`pools` on a fn node is a declared capacity limit; the engine must hold it
    (was: field silently ignored, unbounded concurrent fn bodies)."""
    inflight = [0]
    peak = [0]

    async def burst_fn(up: Upstream) -> str:
        inflight[0] += 1
        peak[0] = max(peak[0], inflight[0])
        await asyncio.sleep(0.02)
        inflight[0] -= 1
        return "done"

    class Pooled(Flow):
        start = fn(lambda up: "x", then=("a", "b", "c"))
        a = fn(burst_fn, pools=("api",))
        b = fn(burst_fn, pools=("api",))
        c = fn(burst_fn, pools=("api",))

    import tempfile

    with tempfile.TemporaryDirectory() as d:
        engine = Engine(
            Pooled(FlowConfig(pools={"api": 1})), Path(d)
        )
        results = await engine.run([{"id": i} for i in range(8)])
    assert all(r.ok for r in results)
    assert peak[0] <= 1, f"pool of 1 exceeded: {peak[0]} concurrent fn bodies"


# -- partial resume and inherited runtimes -----------------------------------------


async def test_resume_cannot_restore_inherited_runtime_fails_loudly(tmp_path):
    """EXPLICIT LIMIT: a ledger attach skips execution, so the live box a holder
    leaves its inheritors is gone after an interrupted run. The engine must NOT
    invent state (a fresh box would lose the holder's side effects); it fails with
    an error that says what to do (delete the holder's record to re-run it)."""
    stop = tmp_path / "stop"
    stop.write_text("x")

    def gate_fn(up: Upstream) -> str:
        if stop.exists():
            raise RuntimeError("simulated interruption")
        return "continue"

    class Inh(Flow):
        holder = run(["sh", "-c", "echo held"], runtime=vf.SubprocessConfig(), then="gate")
        gate = fn(gate_fn, then="reader")
        reader = run(["sh", "-c", "echo inherited-ok"], runtime="inherit:holder", then="done")
        done = fn(lambda up: up.reader.stdout.strip())

    run_dir = tmp_path / "run"
    (first,) = await Engine(Inh(), run_dir).run([{"id": 1}])
    assert not first.ok  # gate failed; holder completed and was recorded
    stop.unlink()
    (second,) = await Engine(Inh(), run_dir).run([{"id": 1}])
    assert not second.ok
    error = second.error or ""
    assert "restored from the ledger" in error, error
    assert "delete" in error  # names the recovery
    assert "deliberate operator choice" in error  # and its side-effect warning
    # the recovery path the error points at: re-run the holder (operator choice:
    # external effects repeat — safe here because `holder` is side-effect free)
    (run_dir / "nodes" / first.records["holder"].row / "holder@1.json").unlink()
    (third,) = await Engine(Inh(), run_dir).run([{"id": 1}])
    assert third.ok, third.error
    assert third.records["done"].payload == "inherited-ok"


# -- crash isolation ----------------------------------------------------------------


async def test_poisoned_row_does_not_sink_sibling_rows():
    def quorum(up: Upstream) -> int:
        if up.row.get("poison"):
            raise RuntimeError("bad quorum fn")
        return 1

    class Mixed(Flow):
        start = fn(lambda up: "x", then=("b1", "b2"))
        b1 = fn(noop, then="c")
        b2 = fn(noop, then="c")
        c = fn(noop, join=at_least(quorum))

    results = await arun(Mixed(), [{"poison": True}, {"poison": False}])
    assert len(results) == 2
    assert not results[0].ok and "bad quorum fn" in (results[0].error or "")
    assert results[1].ok, results[1].error  # the healthy row survived


async def test_producer_error_drains_started_rows():
    async def slow_fn(up: Upstream) -> str:
        await asyncio.sleep(0.5)
        return "slow"

    class Slow(Flow):
        a = fn(slow_fn, then="b")
        b = fn(noop)

    async def producer():
        yield {"id": 1}
        raise RuntimeError("producer died")

    with tempfile.TemporaryDirectory() as d:
        with pytest.raises(RuntimeError, match="producer died"):
            await Engine(Slow(), Path(d)).run(producer())
        # no leaked row tasks remain
        assert len([t for t in asyncio.all_tasks() if t is not asyncio.current_task()]) == 0


# -- cache identity ------------------------------------------------------------------


async def test_record_key_binds_upstream_payload(tmp_path):
    """An upstream value change must invalidate a downstream attach."""
    state = tmp_path / "state"
    run_dir = tmp_path / "run"
    state.write_text("1")

    class Chain(Flow):
        a = fn(lambda up: state.read_text(), then="b")
        b = fn(lambda up: "saw " + up.a)

    (first,) = await Engine(Chain(), run_dir).run([{"id": 1}])
    assert first.ok and first.records["b"].payload == "saw 1"
    # delete only a's record: a re-runs with a new payload; b must NOT attach stale
    (run_dir / "nodes" / first.records["a"].row / "a@1.json").unlink()
    state.write_text("2")
    (second,) = await Engine(Chain(), run_dir).run([{"id": 1}])
    assert second.ok, second.error
    assert second.records["a"].payload == "2"
    assert second.records["b"].payload == "saw 2", "downstream attached a stale result"


# -- expand: item identity and resume ----------------------------------------------


@pytest.fixture
def fake_rollout(monkeypatch):
    """Fabricated rollouts (no model): the ledger and join bookkeeping are under test."""
    rolls: list[Any] = []

    async def _rollout(self: Any, node: Any, task: Any) -> Any:
        rolls.append(task)
        await asyncio.sleep(0)
        trace = vf.Trace(
            agent=vf.AgentInfo(config=vf.AgentConfig()),
            task=vf.TraceTask(type="Task", data=vf.TaskData(idx=0, prompt=str(task))),
            ok=True,
        )
        await self.e.ledger.append(trace, env=self.g.name)
        return trace

    monkeypatch.setattr(_Row, "_rollout", _rollout)
    return rolls


class SeatCfg(FlowConfig):
    seat: vf.AgentConfig = vf.AgentConfig(model="fake")


class Expand(Flow[SeatCfg]):
    src = fn(lambda up: list(up.row["items"]), then="fan")
    fan = expand(
        "seat",
        lambda up, item: {"solve": item},
        over=lambda up: up.src,
        join=at_least(2),
        then="done",
    )
    done = fn(lambda up: len(up.fan))


def solved(rolls: list[Any]) -> list[str]:
    return [r["solve"] for r in rolls]


ITEMS: dict[str, Any] = {"items": []}
"""Module holder: the flows below read items through it so tests can change the
upstream value while keeping the row (and so the ledger namespace) identical."""


STATE: dict[str, Any] = {"payload": "v1"}


class FanEmbed(Flow[SeatCfg]):
    src = fn(lambda up: STATE["payload"], then="fan")
    fan = expand(
        "seat",
        lambda up, item: {"solve": item, "folder": up.src},
        over=lambda up: [0, 1],  # the item list NEVER changes
        join=at_least(2),
        then="done",
    )
    done = fn(lambda up: len(up.fan))


class ExpandAll(Flow[SeatCfg]):
    src = fn(lambda up: list(ITEMS["items"]), then="fan")
    fan = expand(
        "seat",
        lambda up, item: {"solve": item},
        over=lambda up: up.src,
        join=ALL,
        then="done",
    )
    done = fn(lambda up: len(up.fan))


async def test_expand_resume_attaches_without_new_rollouts(tmp_path, fake_rollout):
    """Invariant: resume (even with the parent record gone) attaches completed items
    without re-rolling them."""
    ITEMS["items"] = ["a", "b", "c"]
    rows = [{"id": 1}]
    (first,) = await Engine(ExpandAll(SeatCfg()), tmp_path).run(rows)
    assert first.ok, first.error
    assert len(fake_rollout) == 3
    # delete the parent record: the node re-runs and per-item attach must kick in
    (tmp_path / "nodes" / first.records["fan"].row / "fan@1.json").unlink()
    n = len(fake_rollout)
    (second,) = await Engine(ExpandAll(SeatCfg()), tmp_path).run(rows)
    assert second.ok, second.error
    assert len(fake_rollout) == n, "resume re-rolled completed items"


async def test_expand_config_change_reruns_every_item(tmp_path, fake_rollout):
    """A config change changes the run identity: every item must re-roll; per-item
    records from the old identity must not attach (blind index attach = stale)."""
    ITEMS["items"] = ["a", "b", "c"]
    rows = [{"id": 1}]
    (first,) = await Engine(ExpandAll(SeatCfg()), tmp_path).run(rows)
    assert first.ok and len(fake_rollout) == 3
    n = len(fake_rollout)
    (second,) = await Engine(ExpandAll(SeatCfg(model="other-model")), tmp_path).run(rows)
    assert second.ok, second.error
    assert sorted(solved(fake_rollout[n:])) == ["a", "b", "c"], (
        "items from the old config were reused"
    )


async def test_expand_upstream_value_change_reruns_changed_items(tmp_path, fake_rollout):
    """Items derive from upstream values: an upstream value change must re-roll the
    changed items and never alias a different item by index."""
    ITEMS["items"] = ["a", "b", "c"]
    (first,) = await Engine(ExpandAll(SeatCfg()), tmp_path).run([{"id": 1}])
    assert first.ok
    assert solved(fake_rollout) == ["a", "b", "c"]
    # upstream re-runs with different items (its record was interrupted away)
    (tmp_path / "nodes" / first.records["src"].row / "src@1.json").unlink()
    n = len(fake_rollout)
    ITEMS["items"] = ["x", "b", "z"]
    (second,) = await Engine(ExpandAll(SeatCfg()), tmp_path).run([{"id": 1}])
    assert second.ok, second.error
    # per-item keys bind the upstream material: an upstream value change re-rolls
    # every item (item 'b' included — make_task may embed upstream values)
    assert sorted(solved(fake_rollout[n:])) == ["b", "x", "z"], "stale item attach"
    assert second.records["done"].payload == 3


async def test_expand_item_records_have_item_bound_keys(tmp_path, fake_rollout):
    ITEMS["items"] = ["a", "b", "c"]
    (first,) = await Engine(ExpandAll(SeatCfg()), tmp_path).run([{"id": 1}])
    assert first.ok, first.error
    row_dir = tmp_path / "nodes" / first.records["fan"].row
    keys = set()
    for i in range(3):
        record = NodeRecord.model_validate_json(
            (row_dir / f"fan@1.{i}.json").read_text()
        )
        keys.add(record.key)
    assert len(keys) == 3, "per-item records share one key"




# -- rejection gates on a real accept-pipeline topology --------------------------------


async def test_review_gate_blocks_judge_in_sequenced_topology():
    """Gating that HOLDS today: the accept edge is the only way to reach the judge,
    so a rejected review never activates it (judge/bank never run)."""
    ran: list[str] = []

    def tracked(name: str, value: str) -> Any:
        def _fn(up: Upstream) -> str:
            ran.append(name)
            return value

        return _fn

    class Pipeline(Flow):
        prescreen = fn(
            lambda up: "accept",
            outcomes={"accept": ("build",), "reject": END},
            then=None,
        )
        build = fn(tracked("build", "built"), then="review_folder")
        review_folder = fn(
            lambda up: "reject",
            outcomes={"accept": "judge", "reject": END},
        )
        judge = fn(tracked("judge", "judged"), then="bank")
        bank = fn(tracked("bank", "banked"))

    (result,) = await arun(Pipeline(), [{"id": 1}])
    assert result.ok, result.error
    assert ran == ["build"], ran  # judge and bank never ran after rejection


async def test_review_gate_bypassed_in_parallel_topology():
    """OBSERVED LIMIT (semantic gap): when the judge ALSO has a direct edge from
    solve, its ALL join counts the rejected review as dead — the judge runs after
    rejection. Dead-branch semantics cannot express a strict rejection gate over
    node-level predecessors; see findings (proposal: strict join)."""
    ran: list[str] = []

    def tracked(name: str, value: str) -> Any:
        def _fn(up: Upstream) -> str:
            ran.append(name)
            return value

        return _fn

    class Pipeline(Flow):
        prescreen = fn(
            lambda up: "accept",
            outcomes={"accept": ("build", "solve")},
        )
        build = fn(tracked("build", "built"), then="review_folder")
        review_folder = fn(
            lambda up: "reject",
            outcomes={"accept": "judge", "reject": END},
        )
        solve = fn(tracked("solve", "solved"), then="judge")
        judge = fn(tracked("judge", "judged"), then="bank")
        bank = fn(tracked("bank", "banked"))

    (result,) = await arun(Pipeline(), [{"id": 1}])
    assert result.ok, result.error
    assert ran == ["build", "solve", "judge", "bank"], ran


# -- expand fan-out (partial success) ------------------------------------------------


class ExpandQuorum(Flow[SeatCfg]):
    src = fn(lambda up: list(up.row["items"]), then="fan")
    fan = expand(
        "seat",
        lambda up, item: {"solve": item},
        over=lambda up: up.src,
        join=at_least(2),
        then="done",
    )
    done = fn(lambda up: len(up.fan))


async def test_expand_quorum_partial_success_and_cancellation(tmp_path, fake_rollout):
    """at_least(2) of 3: the quorum fires, the straggler is cancelled, the compact
    result carries only the arrived traces, and resume attaches without re-roll."""
    (first,) = await Engine(ExpandQuorum(SeatCfg()), tmp_path).run(
        [{"id": 1, "items": ["a", "b", "c"]}]
    )
    assert first.ok, first.error
    assert len(first.records["fan"].trace_ids) == 2
    assert first.records["done"].payload == 2
    (second,) = await Engine(ExpandQuorum(SeatCfg()), tmp_path).run(
        [{"id": 1, "items": ["a", "b", "c"]}]
    )
    assert second.ok, second.error
    assert second.records["fan"].trace_ids == first.records["fan"].trace_ids


# -- external cancellation (independent-review counterexample 1) ---------------------


async def test_external_cancel_drains_node_tasks():
    """External cancel of Engine.run must cancel the row's in-flight NODE tasks:
    an orphan node finishing later would write the ledger (and touch runtimes)
    after the caller saw CancelledError."""

    async def slow_fn(up: Upstream) -> str:
        await asyncio.sleep(2.0)
        return "slow-done"

    class Slow(Flow):
        fast = fn(lambda up: "fast", then="slow")
        slow = fn(slow_fn, then="end")
        end = fn(lambda up: "end")

    with tempfile.TemporaryDirectory() as d:
        run_dir = Path(d)
        engine = Engine(Slow(), run_dir)
        task = asyncio.create_task(engine.run([{"id": 1}]))
        await asyncio.sleep(0.2)  # the row has started the slow node
        assert task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        for _ in range(3):
            await asyncio.sleep(0)
        others = [t for t in asyncio.all_tasks() if t is not asyncio.current_task()]
        assert not others, f"leaked tasks: {[t.get_name() for t in others]}"
        await asyncio.sleep(0.3)
        records = list((run_dir / "nodes").rglob("*.json"))
        assert not any("slow" in p.name for p in records), (
            "slow node wrote a ledger record after external cancellation"
        )


# -- per-item keys bind the parent's upstream material (counterexample 2) ----------


async def test_expand_item_keys_bind_upstream_material(tmp_path, fake_rollout):
    """An upstream value change that leaves the ITEM LIST identical must still
    re-roll every item: per-item keys bind the parent's upstream material, and the
    rollouts embed upstream values via make_task(up, item)."""
    STATE["payload"] = "v1"
    run_dir = tmp_path / "run"
    (first,) = await Engine(FanEmbed(SeatCfg()), run_dir).run([{"id": 1}])
    assert first.ok, first.error
    assert all("v1" in str(r) for r in fake_rollout)
    # upstream re-runs with a NEW payload (its record was interrupted away);
    # items stay [0, 1]; every item must re-roll with the new upstream
    (run_dir / "nodes" / first.records["src"].row / "src@1.json").unlink()
    n = len(fake_rollout)
    STATE["payload"] = "v2"
    (second,) = await Engine(FanEmbed(SeatCfg()), run_dir).run([{"id": 1}])
    assert second.ok, second.error
    new = fake_rollout[n:]
    assert len(new) == 2, f"expected 2 re-rolled items, got {len(new)}"
    assert all("v2" in str(r) for r in new)


# -- declarative validation: no silent hangs ----------------------------------------


def test_config_rejects_zero_row_concurrency():
    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        FlowConfig(max_concurrent_rows=0)
    with pytest.raises(ValidationError):
        FlowConfig(max_concurrent_rows=-2)


def test_config_rejects_zero_capacity_pools():
    from pydantic import ValidationError

    with pytest.raises(ValidationError, match="pool sizes must be >= 1"):
        FlowConfig(pools={"runtimes": 0})


def test_compile_rejects_degenerate_counts():
    with pytest.raises(FlowError, match="k >= 1"):

        class Zero(Flow):
            a = fn(noop, then="b")
            b = fn(noop, join=at_least(0))

    with pytest.raises(FlowError, match="negative"):

        class NegRetry(Flow):
            a = fn(noop, retries=-1)

    with pytest.raises(FlowError, match="never allows a visit"):

        class ZeroVisits(Flow):
            a = fn(noop, outcomes={"again": "a", "done": END}, max_visits=0)

    with pytest.raises(FlowError, match="not a capacity"):

        class ZeroActive(Flow):
            a = expand(
                "seat", lambda up, item: item, over=lambda up: [0], max_active=0
            )


# -- late-arrival retrigger: explicit semantics and the duplicate side effect -------


async def test_late_arrival_duplicate_side_effect_is_pinned_semantics():
    """DOCUMENTED SEMANTICS: an at_least quorum fires on the first k preds; a
    straggler firing later re-triggers the node (visit 2) and everything `then`
    executes AGAIN — here the side-effectful consumer runs twice. Consumers of a
    quorum must be idempotent or guard on `up.outcome(...)`; the engine does not
    deduplicate visits."""
    effects: list[int] = []

    async def slow_straggler(up: Upstream) -> str:
        await asyncio.sleep(0.05)
        return "slow"

    def consumer_fn(up: Upstream) -> str:
        effects.append(len(effects) + 1)
        return "ran"

    class Fan(Flow):
        start = fn(lambda up: 1, then=("slow", "fast"))
        slow = fn(slow_straggler, then="gate")
        fast = fn(lambda up: "fast", then="gate")
        gate = fn(lambda up: "quorum", then="consume", join=at_least(1))
        consume = fn(consumer_fn, then="bank")
        bank = fn(lambda up: len(effects))

    (result,) = await arun(Fan(), [{"id": 1}])
    assert result.ok, result.error
    assert effects == [1, 2], "consumer must run twice: quorum then straggler"
    assert result.records["consume"].visit == 2
    assert result.records["bank"].payload == 2


# -- stress-fuzz findings ------------------------------------------------------------


async def test_at_least_callable_k_waits_for_straggler_upstream():
    """k may read an upstream that has not landed yet (a straggler): that is
    NOT-READY, not a crashed row (was: AttributeError poisons a legal row)."""

    async def slow_fn(up: Upstream) -> str:
        await asyncio.sleep(0.05)
        return "slow"

    def quorum(up: Upstream) -> int:
        return 2 if up.slow == "slow" else 1

    class Fan(Flow):
        start = fn(lambda up: 1, then=("slow", "fast"))
        slow = fn(slow_fn, then="gate")
        fast = fn(lambda up: "fast", then="gate")
        gate = fn(lambda up: len(up.outcome("slow") or "") + len("x"), join=at_least(quorum))

    (result,) = await arun(Fan(), [{"id": 1}])
    assert result.ok, result.error  # the row is legal: k resolved once slow landed


def test_compile_rejects_node_named_entry():
    with pytest.raises(FlowError, match="may not be named 'entry'"):

        class Bad(Flow):
            entry = fn(noop, then="a", outcomes={"x": "a"})  # type: ignore[assignment]
            a = fn(noop)


# -- fn(over=): model-free dynamic fan-out (dsl-ergonomics P1/P2, reconciled) --------


async def test_fn_over_fans_out_and_joins_all():
    calls: list[str] = []

    def fan(up: Upstream, item: str) -> str:
        calls.append(item)
        return f"done:{item}"

    class Fan(Flow):
        start = fn(lambda up: ["a", "b"], then="scatter")
        scatter = fn(fan, over=lambda up: up.start, join=ALL, then="end")
        end = fn(lambda up: sorted(up.scatter))

    (result,) = await arun(Fan(), [{"id": 1}])
    assert result.ok, result.error
    assert sorted(calls) == ["a", "b"]
    assert result.records["end"].payload == ["done:a", "done:b"]


async def test_fn_over_quorum_drops_failed_items_and_up_items_keeps_indices():
    def maybe_fail(up: Upstream, item: int) -> str:
        if item == 1:
            raise RuntimeError(f"item {item} exploded")
        return f"ok{item}"

    class Fan(Flow):
        start = fn(lambda up: [0, 1, 2], then="scatter")
        scatter = fn(maybe_fail, over=lambda up: up.start, join=at_least(2), then="end")
        end = fn(lambda up: (sorted(up.scatter), sorted(up.items("scatter").items())))

    (result,) = await arun(Fan(), [{"id": 1}])
    assert result.ok, result.error
    compact, by_index = result.records["end"].payload
    assert compact == ["ok0", "ok2"]  # failed item dropped from the compact list
    assert by_index == [[0, "ok0"], [2, "ok2"]]  # original-index identity


async def test_fn_over_resume_attaches_items_without_rerun(tmp_path):
    runs: list[str] = []

    def fan(up: Upstream, item: str) -> str:
        runs.append(item)
        return f"done:{item}"

    class Fan(Flow):
        start = fn(lambda up: ["a", "b", "c"], then="scatter")
        scatter = fn(fan, over=lambda up: up.start, join=ALL, then="end")
        end = fn(lambda up: len(up.scatter))

    rows = [{"id": 1}]
    (first,) = await Engine(Fan(), tmp_path).run(rows)
    assert first.ok and len(runs) == 3
    # delete the parent record: per-item attach must still work, keyed by item
    (tmp_path / "nodes" / first.records["scatter"].row / "scatter@1.json").unlink()
    (second,) = await Engine(Fan(), tmp_path).run(rows)
    assert second.ok, second.error
    assert len(runs) == 3, "resume re-rolled completed fn items"


async def test_fn_over_config_change_invalidates_items(tmp_path):
    runs: list[str] = []

    def fan(up: Upstream, item: str) -> str:
        runs.append(item)
        return f"done:{item}"

    class Fan(Flow):
        start = fn(lambda up: ["a", "b"], then="scatter")
        scatter = fn(fan, over=lambda up: up.start, join=ALL, then="end")
        end = fn(lambda up: len(up.scatter))

    rows = [{"id": 1}]
    (first,) = await Engine(Fan(), tmp_path).run(rows)
    assert first.ok and len(runs) == 2
    (second,) = await Engine(Fan(FlowConfig(model="other")), tmp_path).run(rows)
    assert second.ok, second.error
    assert len(runs) == 4, "fn items from the old config were blindly reused"


async def test_fn_over_items_hold_declared_pools():
    inflight = [0]
    peak = [0]

    async def burst(up: Upstream, item: int) -> str:
        inflight[0] += 1
        peak[0] = max(peak[0], inflight[0])
        await asyncio.sleep(0.02)
        inflight[0] -= 1
        return "ok"

    class Fan(Flow):
        start = fn(lambda up: list(range(6)), then="scatter")
        scatter = fn(burst, over=lambda up: up.start, join=ALL, pools=("api",))

    with tempfile.TemporaryDirectory() as d:
        engine = Engine(Fan(FlowConfig(pools={"api": 2})), Path(d))
        results = await engine.run([{"id": 1}])
    assert all(r.ok for r in results)
    assert peak[0] <= 2, f"pool of 2 exceeded: {peak[0]}"


async def test_fn_over_compile_rejects_outcomes():
    with pytest.raises(FlowError, match="routes with `then`"):

        class Bad(Flow):
            a = fn(
                lambda up, item: item,
                over=lambda up: [0],
                outcomes={"x": END},
            )


# -- fn(over=) adoption regressions (parent-required before any merge) --------------


async def test_fn_over_none_outputs_preserved_across_partial_resume(tmp_path):
    """A legitimate None item result is a VALUE: it must survive the compact list,
    attach on resume (parent record gone), and not re-run the item (side effects)."""
    runs: list[int] = []

    def fan(up: Upstream, item: int) -> str | None:
        runs.append(item)
        return None if item == 0 else f"ok{item}"

    class Fan(Flow):
        start = fn(lambda up: [0, 1], then="scatter")
        scatter = fn(fan, over=lambda up: up.start, join=ALL, then="end")
        end = fn(lambda up: (up.scatter, up.items("scatter")))

    rows = [{"id": 1}]
    (first,) = await Engine(Fan(), tmp_path).run(rows)
    assert first.ok, first.error
    compact, by_index = first.records["end"].payload
    assert compact == [None, "ok1"]
    # payload is the JSON serialization of up.items(): int keys serialize as strings
    assert by_index == {"0": None, "1": "ok1"}
    (tmp_path / "nodes" / first.records["scatter"].row / "scatter@1.json").unlink()
    n = len(runs)
    (second,) = await Engine(Fan(), tmp_path).run(rows)
    assert second.ok, second.error
    assert len(runs) == n, "None-valued items were re-run on resume"
    compact2, by_index2 = second.records["end"].payload
    assert compact2 == [None, "ok1"] and by_index2 == {"0": None, "1": "ok1"}


async def test_fn_over_indices_stay_original_integers_under_failures():
    """Failed items leave HOLES: indices are never renumbered and stay integers."""

    def fail_odd(up: Upstream, item: int) -> str:
        if item % 2:
            raise RuntimeError(f"item {item}")
        return f"ok{item}"

    class Fan(Flow):
        start = fn(lambda up: [0, 1, 2, 3, 4], then="scatter")
        scatter = fn(fail_odd, over=lambda up: up.start, join=at_least(2), then="end")
        end = fn(lambda up: sorted(up.items("scatter").items()))

    (result,) = await arun(Fan(), [{"id": 1}])
    assert result.ok, result.error
    keys, values = zip(*result.records["end"].payload)
    assert list(keys) == [0, 2, 4]  # integer holes, never renumbered
    assert list(values) == ["ok0", "ok2", "ok4"]


async def test_fn_over_items_result_mutation_cannot_alter_row_state():
    seen: list[Any] = []

    def mutate(up: Upstream) -> str:
        items = up.items("scatter")
        for k in items:
            items[k] = "TAMPERED"
        seen.append(dict(items))  # our local copy holds the tampering
        return "done"

    class Fan(Flow):
        start = fn(lambda up: ["a", "b"], then="scatter")
        scatter = fn(lambda up, item: f"ok:{item}", over=lambda up: up.start, then="mid")
        mid = fn(mutate, then="end")
        end = fn(lambda up: sorted(up.items("scatter").items()))

    (result,) = await arun(Fan(), [{"id": 1}])
    assert result.ok, result.error
    assert seen == [{0: "TAMPERED", 1: "TAMPERED"}]  # the copy is ours to abuse
    assert result.records["end"].payload == [
        [0, "ok:a"],
        [1, "ok:b"],
    ], "mutating up.items() leaked into row state"


async def test_fn_over_duplicate_items_get_distinct_instances(tmp_path):
    """Equal input items are distinct instances: both run, both record, and resume
    attaches both without re-run (identity = ledger path index, not item value)."""
    runs: list[str] = []

    def fan(up: Upstream, item: str) -> str:
        runs.append(item)
        return f"ok:{item}:{len(runs)}"

    class Fan(Flow):
        start = fn(lambda up: ["a", "a"], then="scatter")
        scatter = fn(fan, over=lambda up: up.start, join=ALL, then="end")
        end = fn(lambda up: up.scatter)

    rows = [{"id": 1}]
    (first,) = await Engine(Fan(), tmp_path).run(rows)
    assert first.ok, first.error
    assert len(runs) == 2, "duplicate items must both run as distinct instances"
    assert len(set(first.records["scatter"].payload)) == 2, "distinct instance outputs"
    row_dir = tmp_path / "nodes" / first.records["scatter"].row
    keys = {
        NodeRecord.model_validate_json((row_dir / f"scatter@1.{i}.json").read_text()).key
        for i in range(2)
    }
    assert len(keys) == 1, "equal items share one key (identity is the path index)"
    (row_dir / "scatter@1.json").unlink()
    n = len(runs)
    (second,) = await Engine(Fan(), tmp_path).run(rows)
    assert second.ok, second.error
    assert len(runs) == n, "resume re-rolled duplicate items"
    assert second.records["scatter"].payload == first.records["scatter"].payload


# -- callable-k runtime guard (separate patch branch) -------------------------------


async def test_at_least_callable_k_below_one_fails_the_row_loudly():
    """A callable k that returns < 1 at runtime is a data error: the row fails with
    a clear message instead of a join that fires with zero preds."""

    def zero_k(up: Upstream) -> int:
        return 0

    class Fan(Flow):
        start = fn(lambda up: 1, then=("a", "b"))
        a = fn(noop, then="pick")
        b = fn(noop, then="pick")
        pick = fn(noop, join=at_least(zero_k))

    (result,) = await arun(Fan(), [{"id": 1}])
    assert not result.ok
    assert "k=0 < 1" in (result.error or ""), result.error


async def test_expand_callable_k_below_one_fails_the_node(tmp_path, fake_rollout):
    class SeatCfg(FlowConfig):
        seat: vf.AgentConfig = vf.AgentConfig(model="fake")

    class Bad(Flow[SeatCfg]):
        src = fn(lambda up: [0, 1], then="fan")
        fan = expand(
            "seat",
            lambda up, item: item,
            over=lambda up: up.src,
            join=at_least(lambda up: 0),
            then="done",
        )
        done = fn(noop)

    (result,) = await Engine(Bad(SeatCfg()), tmp_path).run([{"id": 1}])
    assert not result.ok
    assert "k=0 < 1" in (result.error or ""), result.error
    assert not fake_rollout, "no item may roll before the quorum guard fires"


# -- quorum cohort identity (parent-required before any landed() change) ------------


async def test_expand_quorum_cohort_is_exactly_the_selected_attempt_set(
    tmp_path, fake_rollout
):
    """The node RECORD (trace_ids / compact up.<name>) is the SELECTED quorum
    cohort — the first `need` items that arrived — never the superset that may
    also have completed. up.items() exposes ALL completed items at their original
    indices and MAY exceed the cohort: ports mapping attempts (flywheel landed())
    must read the cohort, not up.items."""

    class SeatCfg(FlowConfig):
        seat: vf.AgentConfig = vf.AgentConfig(model="fake")

    class Fan(Flow[SeatCfg]):
        start = fn(lambda up: [0, 1, 2], then="fan")
        fan = expand(
            "seat",
            lambda up, item: {"solve": item},
            over=lambda up: up.start,
            join=at_least(2),
            then="cohort",
        )
        cohort = fn(lambda up: (up.fan, up.items("fan")))

    (result,) = await Engine(Fan(SeatCfg()), tmp_path).run([{"id": 1}])
    assert result.ok, result.error
    compact, items = result.records["cohort"].payload
    record = result.records["fan"]
    # the cohort: exactly the selected `need` items, in original index order
    assert len(compact) == 2
    assert len(record.trace_ids) == 2, "record cohort must be exactly the selection"
    # late successes may exist in up.items beyond the cohort (all-instant rollouts
    # can complete all three before the break) — never in the record's cohort
    assert len(items) >= 2
    cohort_ids = sorted(t["id"] for t in compact)
    assert cohort_ids == sorted(record.trace_ids)


# -- engine-review round: regression tests for the 7 proven findings -----------------


async def test_unkeyable_row_fails_without_poisoning_siblings():
    class Ok(Flow):
        a = fn(lambda up: "x")

    results = await arun(Ok(), [object(), {"id": 1}])
    assert len(results) == 2
    assert not results[0].ok and "cannot be keyed" in (results[0].error or "")
    assert results[1].ok, results[1].error  # sibling survived


async def test_attach_requires_resolvable_trace(tmp_path, fake_rollout):
    """An agent record whose trace is gone (traces.jsonl lost/edited) must NOT
    attach as a silent None downstream: the node re-runs."""
    class SeatCfg(FlowConfig):
        seat: vf.AgentConfig = vf.AgentConfig(model="fake")

    class Agent(Flow[SeatCfg]):
        a = expand("seat", lambda up, item: item, over=lambda up: [0, 1], then="done")
        done = fn(lambda up: len(up.a))

    rows = [{"id": 1}]
    (first,) = await Engine(Agent(SeatCfg()), tmp_path).run(rows)
    assert first.ok and len(fake_rollout) == 2
    # drop ONE trace line from traces.jsonl; the parent record must not attach
    lines = (tmp_path / "traces.jsonl").read_text().splitlines()
    (tmp_path / "traces.jsonl").write_text("\n".join(lines[:1]) + "\n")
    (tmp_path / "nodes" / first.records["a"].row / "a@1.json").unlink()
    n = len(fake_rollout)
    (second,) = await Engine(Agent(SeatCfg()), tmp_path).run(rows)
    assert second.ok, second.error
    assert len(fake_rollout) > n, "attached despite a missing trace"


async def test_row_keys_with_sets_are_seed_independent():
    """A set-valued row — plain and MODEL-EMBEDDED — must key identically across
    processes (PYTHONHASHSEED); the review residual had model sets unordered."""
    import subprocess
    import sys

    code = (
        "from pydantic import BaseModel\n"
        "from verifiers.v1.flow.ledger import row_key\n"
        "class Row(BaseModel):\n"
        "    tags: set[str]\n"
        "print(row_key({'tags': {'alpha', 'beta', 'gamma'}}))\n"
        "print(row_key(Row(tags={'alpha', 'beta', 'gamma'})))"
    )
    venv = sys.executable
    out = []
    for seed in ("1", "2"):
        res = subprocess.run(  # noqa: ASYNC221
            [venv, "-c", code],
            capture_output=True,
            text=True,
            env={"PYTHONHASHSEED": seed, "PATH": "/usr/bin:/bin"},
            check=False,
        )
        out.append(res.stdout.split())
    plain = [lines[0] for lines in out]
    model = [lines[1] for lines in out]
    assert plain[0] == plain[1], f"hash-seed-dependent plain-set row key: {out}"
    assert model[0] == model[1], f"hash-seed-dependent model-set row key: {out}"


async def test_torn_final_trace_line_does_not_block_resume(tmp_path):
    """A SIGKILL can leave a torn final line in traces.jsonl; resume must still
    resolve the traces written before it."""
    from verifiers.v1.flow.ledger import Ledger

    run_dir = tmp_path / "run"
    run_dir.mkdir()
    trace = vf.Trace(
        agent=vf.AgentInfo(config=vf.AgentConfig()),
        task=vf.TraceTask(type="Task", data=vf.TaskData(idx=0, prompt="p")),
        ok=True,
    )
    ledger = Ledger(run_dir)
    await ledger.append(trace, env="e")
    with (run_dir / "traces.jsonl").open("a") as f:
        f.write('{"torn"')  # the process died mid-write
    fresh = Ledger(run_dir)
    assert fresh.trace(trace.id) is not None, "torn tail blocked trace resolution"


def test_engine_rejects_undeclared_pool_names():
    class Pooled(Flow):
        a = fn(lambda up: "x", pools=("runtmes",))  # typo

    with pytest.raises(FlowError, match="unknown pools"):
        Engine(Pooled(FlowConfig(pools={"runtimes": 4})), Path(tempfile.mkdtemp()))


async def test_fn_attach_survives_unresolvable_string_annotation(tmp_path):
    class Local(BaseModel):
        n: int = 0

    def make(up: Upstream) -> "Local":  # noqa: UP037 — the string annotation is the point
        return Local(n=1)

    class Guard(Flow):
        a = fn(make, then="b")
        b = fn(lambda up: up.a.n)

    rows = [{"id": 1}]
    (first,) = await Engine(Guard(), tmp_path).run(rows)
    assert first.ok and first.records["b"].payload == 1
    (second,) = await Engine(Guard(), tmp_path).run(rows)
    assert second.ok, second.error
    assert second.records["b"].payload == 1  # attached; no NameError row crash


# -- digest stability: set-free models digest byte-identically to the base ------------


def test_set_free_model_digests_are_byte_identical_to_base():
    """Set canonicalization must not reshape how SET-FREE models digest: aliases,
    serialization aliases, excludes, computed fields, and field/model serializers
    included — or every existing ledger key for such rows would silently change.
    The pinned constants are the digests computed on the ACCEPTED BASE
    (5c85ac42d) for plain/seralias/excluded/computed/fieldser/modelser shapes."""
    from pydantic import (
        BaseModel,
        Field,
        computed_field,
        field_serializer,
        model_serializer,
    )

    from verifiers.v1.flow.ledger import digest

    class Plain(BaseModel):
        name: str = "x"
        n: int = 2

    class SerAlias(BaseModel):
        name: str = Field(default="x", serialization_alias="NAME")

    class Excluded(BaseModel):
        keep: str = "v"
        drop: str = Field(default="secret", exclude=True)

    class Computed(BaseModel):
        a: int = 2

        @computed_field
        @property
        def double(self) -> int:
            return self.a * 2

    class FieldSer(BaseModel):
        when: str = "2026-09-14T00:00:00"

        @field_serializer("when")
        def _ser(self, v: str) -> str:
            return v[:10]

    class ModelSer(BaseModel):
        a: int = 1
        b: int = 2

        @model_serializer
        def _ms(self) -> dict:
            return {"b": self.b, "a": self.a}

    expected = {
        "plain": "ba3506b995a589c10ab96ad0",
        "seralias": "21364e629709530b49b96971",
        "excluded": "f6a6f220abd1135f272b10df",
        "computed": "c4a96b700f848d7cf3ddc59c",
        "fieldser": "59e28701ea41b0fa948c3fd3",
        "modelser": "1176b831c374ed0092e76938",
    }
    for name, model in (
        ("plain", Plain()),
        ("seralias", SerAlias()),
        ("excluded", Excluded()),
        ("computed", Computed()),
        ("fieldser", FieldSer()),
        ("modelser", ModelSer()),
    ):
        assert digest(model)[:24] == expected[name], f"{name} key drifted from base"


# -- prime teardown: a failed delete must not permanently leak a paid box -------------


async def _sleep_none():
    return None


def _prime_runtime_with(delete_attempts: list[Exception | None]):
    """A PrimeRuntime shell around a fake client; `delete_attempts` are raised in
    order (None = success on that attempt)."""
    import asyncio

    from verifiers.v1.runtimes import prime as prime_mod

    calls = []

    class FakeClient:
        async def delete(self, box_id):
            calls.append(box_id)
            outcome = delete_attempts.pop(0)
            if outcome is not None:
                raise outcome

        async def aclose(self):
            pass

    runtime = prime_mod.PrimeRuntime.__new__(prime_mod.PrimeRuntime)
    runtime.env = {}
    runtime._client = FakeClient()
    runtime.info = prime_mod.PrimeRuntimeInfo(id="box-123")
    loop = asyncio.get_running_loop()
    shared = prime_mod._shared_clients.get(loop)
    if shared is None:
        shared = prime_mod._shared_clients[loop] = prime_mod._SharedClient(FakeClient())
    shared.leases += 1
    return runtime, calls


async def test_prime_teardown_retries_a_failed_delete(tmp_path, caplog, monkeypatch):
    from verifiers.v1.runtimes import prime as prime_mod

    monkeypatch.setattr(prime_mod.asyncio, "sleep", lambda _: _sleep_none())
    runtime, calls = _prime_runtime_with([RuntimeError("transient 5xx"), None])
    await runtime.stop()
    assert calls == ["box-123", "box-123"], "delete must retry after a transient failure"
    assert runtime._client is None, "idempotency guard consumed on success"


async def test_prime_teardown_final_failure_is_loud(caplog, monkeypatch):
    import logging

    from verifiers.v1.runtimes import prime as prime_mod

    async def _no_sleep(_):
        return None

    monkeypatch.setattr(prime_mod.asyncio, "sleep", _no_sleep)
    runtime, calls = _prime_runtime_with(
        [RuntimeError("e1"), RuntimeError("e2"), RuntimeError("e3")]
    )
    with caplog.at_level(logging.ERROR, logger="verifiers.v1.runtimes.prime"):
        await runtime.stop()
    assert len(calls) == 3, "delete must be attempted exactly 3 times"
    assert "NOT CONFIRMED" in caplog.text and "box-123" in caplog.text, (
        "the unconfirmed box id must be in the error log for ops verification"
    )
    assert "box still live" not in caplog.text, "must not claim false certainty on 404/lost response"


async def test_prime_teardown_hung_delete_is_time_bounded(caplog, monkeypatch):
    import asyncio
    import logging
    import time

    from verifiers.v1.runtimes import prime as prime_mod

    # per-attempt bound shrunk so the test is fast
    monkeypatch.setattr(prime_mod, "_DELETE_TIMEOUT_S", 0.05)

    cancels = []

    class HangingClient:
        """Mimics the SDK delete: internal retry loop that awaits a hung call.
        CancelledError must NOT be swallowed by `except Exception` retries."""

        async def delete(self, box_id):
            for _ in range(3):  # SDK-style internal retries
                try:
                    await asyncio.sleep(30)  # hung connection, never returns
                except asyncio.CancelledError:
                    cancels.append(box_id)
                    raise
                except Exception:  # noqa: BLE001, S112 - SDK-style blind retry
                    continue

        async def aclose(self):
            pass

    runtime, _calls = _prime_runtime_with([])
    runtime._client = HangingClient()
    t0 = time.monotonic()
    with caplog.at_level(logging.ERROR, logger="verifiers.v1.runtimes.prime"):
        await runtime.stop()
    wall = time.monotonic() - t0
    assert wall < 3.0, f"teardown must not stall on a hung delete (took {wall:.1f}s)"
    assert len(cancels) == 3, "each attempt's hung await must be cancelled"
    assert "NOT CONFIRMED" in caplog.text and "box-123" in caplog.text
