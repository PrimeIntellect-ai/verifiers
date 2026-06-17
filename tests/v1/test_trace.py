"""A dumped trace round-trips back into a model via `WireTrace.from_wire` — branches, task
extras, and recomputed derived fields all survive, without importing the originating taskset."""

import json

import verifiers.v1 as vf
from verifiers.v1.graph import MessageNode
from verifiers.v1.types import AssistantMessage, UserMessage


class _Task(vf.Task):
    answer: str = ""  # a taskset-specific field WireTask must absorb


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
