"""Trace construction + serialization round-trip: a dumped trace re-validates with plain pydantic
(derived values — reward/is_truncated/error/duration — are properties, not serialized, so they just
recompute on load), transient `state` never crosses the wire, and the permissive `WireTrace` loads a
dump without importing the originating taskset."""

import json

import pytest

import verifiers.v1 as vf
from verifiers.v1.graph import MessageNode
from verifiers.v1.types import AssistantMessage, UserMessage


class MyTask(vf.Task):
    answer: str = ""  # a taskset-specific field WireTask must absorb


class MyState(vf.State):
    score: int = 0


def test_bare_trace_round_trip():
    # The minimal trace: a base task, no nodes, no extras — dump and back into a plain Trace.
    tr = vf.Trace(task=vf.Task(idx=3, prompt="hello"))
    rt = vf.Trace.model_validate(tr.model_dump())
    assert rt.id == tr.id
    assert rt.task.idx == 3 and rt.task.prompt == "hello"
    assert rt.num_turns == 0 and rt.num_branches == 0
    assert rt.reward == 0.0 and rt.errors == []


def test_custom_task_state_round_trip():
    # A custom Task + State, round-tripped into the same parameterization. Custom task fields are
    # typed (not just `model_extra`); `state` is runtime-only and never crosses the wire.
    tr = vf.Trace[MyTask, MyState](
        task=MyTask(idx=0, prompt="q", answer="gold"),
        state=MyState(score=7),
        nodes=[
            MessageNode(parent=None, message=UserMessage(content="q"), sampled=False),
            MessageNode(parent=0, message=AssistantMessage(content="a"), sampled=True),
        ],
    )
    tr.record_reward("r", 0.5)
    wire = tr.model_dump()
    assert "state" not in wire  # transient state is excluded from the dump

    rt = vf.Trace[MyTask, MyState].model_validate(wire)
    assert (
        isinstance(rt.task, MyTask) and rt.task.answer == "gold"
    )  # typed custom field
    assert rt.num_turns == 1 and rt.num_branches == 1
    assert rt.reward == 0.5  # property recomputed from `rewards`


def test_wire_trace_round_trip():
    # Two leaves off one root → 2 branches (a compaction-shaped trace), so the round-trip has to
    # carry node `parent` links for `num_branches` to survive.
    tr = vf.Trace[MyTask, vf.State](
        task=MyTask(idx=0, prompt="q", answer="a"),
        nodes=[
            MessageNode(parent=None, message=UserMessage(content="q"), sampled=False),
            MessageNode(parent=0, message=AssistantMessage(content="a1"), sampled=True),
            MessageNode(parent=0, message=AssistantMessage(content="a2"), sampled=True),
        ],
    )
    tr.record_reward("r", 1.0)
    tr.info = {"build": "ok"}
    tr.stop("done")

    # the dump is plain pydantic — derived values are properties, so they're not serialized
    data = json.loads(tr.model_dump_json(exclude_none=True))
    assert "reward" not in data and "is_truncated" not in data

    rt = vf.WireTrace.model_validate(data)
    assert rt.num_branches == tr.num_branches == 2  # branch topology survived
    assert rt.num_turns == tr.num_turns == 2
    assert rt.reward == 1.0  # property recomputed from `rewards`
    assert rt.stop_condition == "done"
    assert rt.info == {"build": "ok"}
    assert rt.task.model_extra == {
        "answer": "a"
    }  # taskset extras preserved on WireTask

    # the env-server wire form (a plain model_dump) loads too
    assert vf.WireTrace.model_validate(tr.model_dump()).num_branches == 2


def test_to_record_excludes_node_tensors_and_round_trips():
    """`to_record` is the disk/W&B record: it strips the per-node training tensors that can't
    round-trip JSON (the plain json dump crashes on real expert ids) and the result reloads as
    a WireTrace, while token_ids/mask/logprobs/usage are kept."""
    import numpy as np
    from renderers.base import MultiModalData

    from verifiers.v1.types import Usage

    # uint8 expert ids with a real (>0x7F) value, plus a multimodal numpy item — both ride the
    # wire as raw bytes but cannot UTF-8 decode for JSON.
    routed = np.array([[[200]], [[3]]], dtype=np.uint8)
    mmd = MultiModalData(
        mm_items={"image": [{"pixel_values": np.array([200], np.uint8)}]}
    )
    tr = vf.Trace(
        task=MyTask(idx=0, prompt="q", answer="a"),
        nodes=[
            MessageNode(parent=None, message=UserMessage(content="q"), sampled=False),
            MessageNode(
                parent=0,
                message=AssistantMessage(content="a"),
                sampled=True,
                token_ids=[1, 2],
                mask=[True, True],
                logprobs=[-0.1, -0.2],
                usage=Usage(prompt_tokens=1, completion_tokens=2),
                routed_experts=routed,
                multi_modal_data=mmd,
            ),
        ],
    )

    # The load-bearing reason for the exclude: a plain json dump crashes on the raw tensor bytes.
    with pytest.raises(UnicodeDecodeError):
        tr.model_dump(mode="json")

    rec = tr.to_record()
    json.dumps(rec)  # genuinely JSON-serializable
    node = rec["nodes"][1]
    assert "routed_experts" not in node and "multi_modal_data" not in node
    # Training payload that DOES round-trip is kept.
    assert node["token_ids"] == [1, 2] and node["mask"] == [True, True]
    assert node["usage"]["completion_tokens"] == 2
    assert "state" not in rec  # transient, never dumped
    # Reloads as a WireTrace with the graph intact.
    rt = vf.WireTrace.model_validate(rec)
    assert rt.num_branches == 1 and rt.num_turns == 1 and rt.completion_len == 2
