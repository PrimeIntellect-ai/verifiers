"""Trace construction + wire round-trip: a dumped trace re-validates via `from_wire` (branches,
task fields, and recomputed derived fields survive; transient `state` does not), and the permissive
`WireTrace` loads one without importing the originating taskset."""

import json

import verifiers.v1 as vf
from verifiers.v1.graph import MessageNode
from verifiers.v1.types import AssistantMessage, UserMessage


class _Task(vf.Task):
    answer: str = ""  # a taskset-specific field WireTask must absorb


class _State(vf.State):
    score: int = 0


def test_bare_trace_round_trip():
    # The minimal trace: a base task, no nodes, no extras — to_wire and back into a plain Trace.
    tr = vf.Trace(task=vf.Task(idx=3, instruction="hello"))
    rt = vf.Trace.from_wire(tr.to_wire())
    assert rt.id == tr.id
    assert rt.task.idx == 3 and rt.task.instruction == "hello"
    assert rt.num_turns == 0 and rt.num_branches == 0
    assert rt.reward == 0.0 and rt.errors == []


def test_custom_task_state_round_trip():
    # A custom Task + State, round-tripped into the same parameterization. Custom task fields are
    # typed (not just `model_extra`); `state` is runtime-only and never crosses the wire.
    tr = vf.Trace[_Task, _State](
        task=_Task(idx=0, instruction="q", answer="gold"),
        state=_State(score=7),
        nodes=[
            MessageNode(parent=None, message=UserMessage(content="q"), sampled=False),
            MessageNode(parent=0, message=AssistantMessage(content="a"), sampled=True),
        ],
    )
    tr.record_reward("r", 0.5)
    wire = tr.to_wire()
    assert "state" not in wire  # transient state is excluded from the dump

    rt = vf.Trace[_Task, _State].from_wire(wire)
    assert isinstance(rt.task, _Task) and rt.task.answer == "gold"  # typed custom field
    assert rt.num_turns == 1 and rt.num_branches == 1
    assert rt.reward == 0.5


def test_wire_trace_round_trip():
    # Two leaves off one root → 2 branches (a compaction-shaped trace), so the round-trip has to
    # carry node `parent` links for `num_branches` to survive.
    tr = vf.Trace[_Task, vf.State](
        task=_Task(idx=0, instruction="q", answer="a"),
        nodes=[
            MessageNode(parent=None, message=UserMessage(content="q"), sampled=False),
            MessageNode(parent=0, message=AssistantMessage(content="a1"), sampled=True),
            MessageNode(parent=0, message=AssistantMessage(content="a2"), sampled=True),
        ],
    )
    tr.record_reward("r", 1.0)
    tr.info = {"build": "ok"}
    tr.stop("done")

    # exactly what `results.jsonl` stores (the derived fields ride along on disk)
    data = json.loads(tr.model_dump_json(exclude_none=True))
    assert {
        "reward",
        "is_truncated",
    } <= data.keys()  # derived fields present, would break a strict load

    rt = vf.WireTrace.from_wire(data)
    assert rt.num_branches == tr.num_branches == 2  # branch topology survived
    assert rt.num_turns == tr.num_turns == 2
    assert rt.reward == 1.0  # recomputed from `rewards`
    assert rt.stop_condition == "done"
    assert rt.info == {"build": "ok"}
    assert rt.task.model_extra == {
        "answer": "a"
    }  # taskset extras preserved on WireTask

    # the env-server wire format (derived already excluded) loads too
    assert vf.WireTrace.from_wire(tr.to_wire()).num_branches == 2
