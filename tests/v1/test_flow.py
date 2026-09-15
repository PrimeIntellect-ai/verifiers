"""The flow engine on model-free nodes: compile checks, routing, fan-out and joins,
bounded cycles, ledger resume, command nodes, and inherited runtimes."""

from typing import Literal

import pytest
from pydantic import BaseModel

import verifiers.v1 as vf
from verifiers.v1.flow import (
    END,
    Engine,
    Flow,
    FlowError,
    Upstream,
    at_least,
    expand,
    fn,
    run,
)


def noop(up: Upstream) -> None:
    return None


def test_compile_rejects_unknown_target():
    with pytest.raises(FlowError, match="unknown node"):

        class Bad(Flow):
            a = fn(noop, then="missing")


def test_compile_rejects_unreachable_node():
    with pytest.raises(FlowError, match="unreachable"):

        class Bad(Flow):
            a = fn(noop)
            b = fn(noop)


def test_compile_rejects_cycle_without_outcome_edge():
    with pytest.raises(FlowError, match="no outcome edge"):

        class Bad(Flow):
            a = fn(noop, then="b", max_visits=3)
            b = fn(noop, then="a", max_visits=3)


def test_compile_rejects_cycle_without_max_visits():
    with pytest.raises(FlowError, match="no node with max_visits"):

        class Bad(Flow):
            a = fn(noop, outcomes={"again": "a", "done": END})


def test_compile_checks_inherited_runtimes():
    with pytest.raises(FlowError, match="not on every path"):

        class Bad(Flow):
            a = fn(noop, outcomes={"x": "b", "y": "c"})
            b = run(["true"], runtime=vf.SubprocessConfig(), then="c")
            c = run(["true"], runtime="inherit:b")

    with pytest.raises(FlowError, match="run node needs a runtime"):

        class Bad2(Flow):
            a = run(["true"], runtime="fresh")


def test_compile_checks_fn_literal_outcomes():
    def decide(up: Upstream) -> Literal["x", "y"]:
        return "x"

    with pytest.raises(FlowError, match="have no outcome edge"):

        class Bad(Flow):
            a = fn(decide, outcomes={"x": END})


def test_compile_rejects_degenerate_counts():
    with pytest.raises(FlowError, match="allows no visit"):

        class Bad(Flow):
            a = fn(noop, outcomes={"again": "a", "done": END}, max_visits=0)


class Score(BaseModel):
    total: int


async def test_walk_routes_fans_out_joins_bounds_cycles_and_resumes(tmp_path):
    calls: list[str] = []

    def start_fn(up: Upstream) -> str:
        calls.append("start")
        return "go"

    def left_fn(up: Upstream) -> int:
        calls.append("left")
        return 1

    def right_fn(up: Upstream) -> Literal["skip", "go"]:
        calls.append("right")
        return "skip"

    def merge_fn(up: Upstream) -> Score:
        calls.append("merge")
        return Score(total=up.left)

    def loop_fn(up: Upstream) -> Literal["again", "done"]:
        calls.append("loop")
        return "again"

    def wrap_fn(up: Upstream) -> dict:
        calls.append("wrap")
        return {"total": up.merge.total, "loop": up.outcome("loop")}

    class Demo(Flow):
        start = fn(start_fn, then=("left", "right"))
        left = fn(left_fn, then="merge")
        right = fn(right_fn, outcomes={"skip": END, "go": "merge"})
        merge = fn(merge_fn, then="loop")  # all-join: right ends without firing
        loop = fn(
            loop_fn,
            outcomes={"again": "loop", "done": END},
            max_visits=2,
            on_exhausted="wrap",
        )
        wrap = fn(wrap_fn)

    (result,) = await Engine(Demo(), tmp_path / "run").run([{"id": 1}])
    assert result.ok, result.error
    assert calls == ["start", "left", "right", "merge", "loop", "loop", "wrap"]
    assert result.records["merge"].payload == {"total": 1}
    assert result.records["loop"].terminal == "exhausted"
    assert result.records["wrap"].payload == {"total": 1, "loop": "exhausted"}

    # A second run over the same ledger attaches to every instance and runs nothing.
    (again,) = await Engine(Demo(), tmp_path / "run").run([{"id": 1}])
    assert again.ok and len(calls) == 7
    assert again.records["wrap"].payload == {"total": 1, "loop": "exhausted"}


async def test_at_least_join_fires_before_all_predecessors(tmp_path):
    def const(value: str):
        return lambda up: value

    class Quorum(Flow):
        start = fn(const("x"), then=("a", "b"))
        a = fn(const("a"), then="pick")
        b = fn(const("b"), then="pick")
        pick = fn(
            lambda up: sorted(n for n in ("a", "b") if up.outcome(n)),
            join=at_least(1),
        )

    (result,) = await Engine(Quorum(), tmp_path / "run").run([{"id": 1}])
    assert result.ok and result.records["pick"].payload in (["a"], ["b"], ["a", "b"])


async def test_revise_edge_re_enters_a_bounded_node_without_waiting_on_it(tmp_path):
    """build ∥ solve, review revises build once, judge joins review and solve."""
    calls: list[str] = []

    def review_fn(up: Upstream) -> Literal["revise", "accept"]:
        calls.append("review")
        return "revise" if calls.count("review") == 1 else "accept"

    def judge_fn(up: Upstream) -> dict:
        return {"review": up.outcome("review"), "solve": up.solve}

    class Pipeline(Flow):
        screen = fn(lambda up: "ok", then=("build", "solve"))
        build = fn(lambda up: calls.append("build"), then="review", max_visits=3)
        review = fn(review_fn, outcomes={"revise": "build", "accept": "judge"})
        solve = fn(lambda up: "solved", then="judge")
        judge = fn(judge_fn)

    (result,) = await Engine(Pipeline(), tmp_path / "run").run([{"id": 1}])
    assert result.ok, result.error
    assert calls == ["build", "review", "build", "review"]
    assert result.records["judge"].payload == {"review": "accept", "solve": "solved"}


async def test_fn_expand_keeps_item_indices_and_resumes(tmp_path):
    seen: list[int] = []

    def square(up: Upstream, item: int) -> int:
        seen.append(item)
        if item == 2:
            raise ValueError("bad item")
        return item * item

    class Squares(Flow):
        each = expand(square, over=lambda up: [1, 2, 3], join=at_least(2), then="total")
        total = fn(lambda up: {"list": up.each, "items": up.items("each")})

    (result,) = await Engine(Squares(), tmp_path / "run").run([{"id": 1}])
    assert result.ok, result.error
    assert result.records["total"].payload == {
        "list": [1, 9],
        "items": {"0": 1, "2": 9},
    }
    (again,) = await Engine(Squares(), tmp_path / "run").run([{"id": 1}])
    assert again.ok and sorted(seen) == [1, 2, 3]


async def test_run_node_routes_on_outcome_line_or_exit_code(tmp_path):
    class Commands(Flow):
        say = run(
            ["sh", "-c", "echo hello; echo 'Outcome: loud'"],
            runtime=vf.SubprocessConfig(),
            outcomes={"loud": "check", "*": END},
        )
        check = run(
            ["sh", "-c", "exit 3"],
            runtime=vf.SubprocessConfig(),
            outcomes={"completed": END, "failed": "note"},
        )
        note = fn(lambda up: up.check.exit_code)

    (result,) = await Engine(Commands(), tmp_path / "run").run([{"id": 1}])
    assert result.ok, result.error
    assert result.records["say"].outcome == "loud"
    assert result.records["check"].outcome == "failed"
    assert result.records["note"].payload == 3


async def test_run_node_failure_routes_on_error(tmp_path):
    class Failing(Flow):
        boom = run(
            ["sh", "-c", "exit 1"],
            runtime=vf.SubprocessConfig(),
            then=END,
            on_error="recover",
        )
        recover = fn(lambda up: up.outcome("boom"))

    (result,) = await Engine(Failing(), tmp_path / "run").run([{"id": 1}])
    assert result.ok, result.error
    assert result.records["boom"].terminal == "error"
    assert result.records["recover"].payload is None


async def test_crashed_row_is_a_failed_row(tmp_path):
    class Crash(Flow):
        a = fn(lambda up: 1 / 0)

    (result,) = await Engine(Crash(), tmp_path / "run").run([{"id": 1}])
    assert not result.ok and "ZeroDivisionError" in (result.error or "")
    assert result.fault == "turn"


async def test_inherited_runtime_is_the_same_live_runtime(tmp_path):
    marker = tmp_path / "marker.txt"

    class Shared(Flow):
        write = run(
            ["sh", "-c", f"echo shared > {marker}"],
            runtime=vf.SubprocessConfig(),
            then="read",
        )
        read = run(["cat", str(marker)], runtime="inherit:write")

    (result,) = await Engine(Shared(), tmp_path / "run").run([{"id": 1}])
    assert result.ok, result.error
    assert result.records["read"].payload["stdout"].strip() == "shared"
