"""`trace_to_sample`: a v1 Trace -> the platform's (v0) eval-sample dict."""

import verifiers.v1 as vf
from verifiers.v1.graph import MessageNode
from verifiers.v1.samples import drop_non_finite, trace_to_sample
from verifiers.v1.types import AssistantMessage, UserMessage


class MyTask(vf.Task):
    answer: str = ""


def _trace(**rewards: float):
    trace = vf.Trace[MyTask, vf.State](
        task=MyTask(idx=2, prompt="q", answer="gold"),
        nodes=[
            MessageNode(parent=None, message=UserMessage(content="q"), sampled=False),
            MessageNode(parent=0, message=AssistantMessage(content="a"), sampled=True),
        ],
    )
    for name, value in (rewards or {"correct": 1.0}).items():
        trace.record_reward(name, value)
    trace.stop("done")
    return trace


def test_trace_to_sample():
    sample = trace_to_sample(_trace(), rollout_number=3)

    assert sample["example_id"] == 2
    assert sample["rollout_number"] == 3
    assert sample["answer"] == "gold"
    assert sample["reward"] == 1.0
    assert sample["is_completed"] and sample["stop_condition"] == "done"
    assert sample["num_steps"] == 1
    # `completion` is the final branch's messages; one branch -> one trajectory entry.
    assert [m["content"] for m in sample["completion"]] == ["q", "a"]
    assert len(sample["trajectory"]) == 1
    assert sample["error"] is None


def test_non_finite_rewards_dropped():
    # NaN/Inf have no JSON literal; they must become null, not break the upload.
    sample = trace_to_sample(_trace(bad=float("nan")))
    assert sample["reward"] is None
    assert drop_non_finite({"a": float("inf"), "b": [float("nan"), 1.0]}) == {
        "a": None,
        "b": [None, 1.0],
    }
