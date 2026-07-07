"""`trace_to_sample`: a v1 Trace -> the platform's (v0) eval-sample dict."""

import verifiers.v1 as vf
from verifiers.v1.graph import MessageNode
from verifiers.v1.push import trace_to_sample
from verifiers.v1.types import AssistantMessage, UserMessage


class MyTask(vf.Task):
    answer: str = ""


def test_trace_to_sample():
    trace = vf.Trace[MyTask, vf.State](
        task=MyTask(idx=2, prompt="q", answer="gold"),
        nodes=[
            MessageNode(parent=None, message=UserMessage(content="q"), sampled=False),
            MessageNode(parent=0, message=AssistantMessage(content="a"), sampled=True),
        ],
    )
    trace.record_reward("correct", 1.0)
    trace.stop("done")

    sample = trace_to_sample(trace, rollout_number=3)

    assert sample["sample_id"] == trace.id
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
