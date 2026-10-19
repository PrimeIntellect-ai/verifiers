"""One task, one reply: the model echoes a phrase back (echo), across the harness x
runtime grid."""

import pytest
from grid import RUNTIMES, grid, skip_harness

# mini-swe-agent's loop requires a bash tool call in every response and ends only on
# `echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT`, so it cannot answer a chat-only task.
SINGLE_TURN = skip_harness(
    grid(runtimes=RUNTIMES),
    "mini-swe-agent",
    "mini-swe-agent requires a bash call every turn; it cannot answer in text",
)


@pytest.mark.parametrize("harness,harness_runtime", SINGLE_TURN, indirect=True)
async def test_single_turn(run, harness, harness_runtime):
    (episode,) = await run(
        "echo",
        harness=harness,
        runtime=harness_runtime,
        env={"agent": {"sampling": {"temperature": 0.0}}},
        max_turns=2,
    )
    (trace,) = episode.traces
    assert trace.num_branches == 1
    assert trace.ok, [error.message for error in trace.errors]
    assert trace.num_turns == 1
    assert trace.stop_condition == "agent_completed"
    assert trace.reward == 1.0, trace.last_reply
    assert trace.task.key == f"echo:{trace.task.data.answer}"
    assert trace.task.hash is not None
    # The seat's resolved identity rides the trace (policy metadata for trainers).
    assert trace.agent is not None
    assert trace.agent.config.sampling.max_tokens == 2048
    assert trace.agent.config.sampling.temperature == 0.0
    # Every sampled turn has one per-call record, linked to its assistant node.
    sampled = [i for i, n in enumerate(trace.nodes) if n.sampled]
    assert [c.node for c in trace.calls if c.error is None] == sampled
    for call in trace.calls:
        assert call.model and call.sampling is not None
        assert call.time.duration > 0
