"""The flow engine on model-free nodes: compile checks, routing, fan-out and joins,
bounded cycles, ledger resume, command nodes, and the git state bus."""

from typing import Literal

import pytest
from pydantic import BaseModel

import verifiers.v1 as vf
from verifiers.v1.flow import END, Engine, Flow, FlowError, Upstream, at_least, fn, run
from verifiers.v1.flow.snapshot import GitBus, SnapshotError
from verifiers.v1.runtimes import provision_runtime


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


def test_compile_rejects_fork_of_unsnapshotted_node_and_boxless_run():
    with pytest.raises(FlowError, match="declares no snapshot"):

        class Bad(Flow):
            a = run(["true"], runtime=vf.SubprocessConfig(), then="b")
            b = run(["true"], runtime="fork:a")

    with pytest.raises(FlowError, match="run node needs a runtime"):

        class Bad2(Flow):
            a = run(["true"], runtime="fresh")


def test_compile_checks_fn_literal_outcomes():
    def decide(up: Upstream) -> Literal["x", "y"]:
        return "x"

    with pytest.raises(FlowError, match="have no outcome edge"):

        class Bad(Flow):
            a = fn(decide, outcomes={"x": END})


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
        merge = fn(
            merge_fn, then="loop"
        )  # all-join: waits for left, sees right end without firing
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

    # A second run over the same ledger attaches to every finished instance and runs nothing.
    (again,) = await Engine(Demo(), tmp_path / "run").run([{"id": 1}])
    assert again.ok and calls == [
        "start",
        "left",
        "right",
        "merge",
        "loop",
        "loop",
        "wrap",
    ]
    assert again.records["wrap"].payload == {"total": 1, "loop": "exhausted"}


async def test_at_least_join_fires_before_all_predecessors(tmp_path):
    def const(value: str):
        return lambda up: value

    class Quorum(Flow):
        start = fn(const("x"), then=("a", "b"))
        a = fn(const("a"), then="pick")
        b = fn(const("b"), then="pick")
        pick = fn(
            lambda up: sorted(n for n in ("a", "b") if up.outcome(n)), join=at_least(1)
        )

    (result,) = await Engine(Quorum(), tmp_path / "run").run([{"id": 1}])
    assert result.ok and result.records["pick"].payload in (["a"], ["b"], ["a", "b"])


async def test_run_node_routes_on_outcome_file_and_exit_code(tmp_path):
    class Commands(Flow):
        say = run(
            ["sh", "-c", 'printf \'{"outcome": "loud"}\' > "$FLOW_OUTCOME"'],
            runtime=vf.SubprocessConfig(),
            outcomes={"loud": "check", "*": END},
        )
        check = run(
            ["sh", "-c", "exit 3"],
            runtime=vf.SubprocessConfig(),
            exit_codes={0: "pass", "*": "fail"},
            outcomes={"pass": END, "fail": "note"},
        )
        note = fn(lambda up: up.check.exit_code)

    (result,) = await Engine(Commands(), tmp_path / "run").run([{"id": 1}])
    assert result.ok, result.error
    assert result.records["say"].outcome == "loud"
    assert result.records["check"].outcome == "fail"
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


async def test_git_bus_round_trip_and_write_once_refs(tmp_path):
    bus = GitBus(tmp_path / "repo.git")
    src = tmp_path / "src"
    src.mkdir()
    (src / "a.txt").write_text("one")
    async with provision_runtime(vf.SubprocessConfig()) as box:
        first = await bus.snapshot(
            box, workdir=str(src), ref="refs/heads/flow/r/build@1"
        )
        (src / "a.txt").write_text("two")
        (src / "b.txt").write_text("new")
        second = await bus.snapshot(
            box,
            workdir=str(src),
            ref="refs/heads/flow/r/build@2",
            base_sha=first.head_sha,
        )
        assert second.base_sha == first.head_sha and second.head_sha != first.head_sha
        with pytest.raises(SnapshotError, match="push_raced"):
            await bus.snapshot(box, workdir=str(src), ref="refs/heads/flow/r/build@2")

        fresh = tmp_path / "fresh"
        await bus.restore(box, second, workdir=str(fresh))
        assert (fresh / "a.txt").read_text() == "two" and (
            fresh / "b.txt"
        ).read_text() == "new"

        # A box that holds the base commit receives only the delta.
        await bus.restore(box, first, workdir=str(fresh))
        assert (fresh / "a.txt").read_text() == "one" and not (fresh / "b.txt").exists()

        moved = second.model_copy(update={"head_sha": first.head_sha})
        with pytest.raises(SnapshotError, match="source_moved"):
            await bus.restore(box, moved, workdir=str(tmp_path / "other"))
