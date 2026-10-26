"""A retry resumes from the box's newest filesystem checkpoint (checkpoint-resume-v1: the
first attempt dies as turn 3 goes upstream; the retry restores the turn-2 snapshot,
keeps turn 1, and finishes). One cell: prime is the one runtime with checkpoints, and
of the harnesses that relaunch on a transcript (`SUPPORTS_RESUME`) bash is the one
the fixture can drive — null has no shell and takes one turn, browser-use needs a
browser image. The ACP harnesses take no checkpoints (they start a fresh process
on user turns, not a handed conversation)."""

import pytest
from checkpoint_resume_v1 import FAIL_AFTER_TURNS
from grid import mark

RETRIES = {"max_retries": 0, "rules": [{"type": "ProviderError", "max_retries": 1}]}


@pytest.mark.parametrize(
    "harness,harness_runtime",
    [
        pytest.param(
            "bash", "prime", marks=[mark("bash"), mark("prime")], id="bash-in-prime"
        )
    ],
    indirect=True,
)
async def test_retry_resumes_from_checkpoint(run, harness, harness_runtime):
    """The box restores to the state before the turn that failed, the turns up to
    there stay sampled, and the task's `setup` (a nonce the prompt never reveals) is
    not rerun — the resumed attempt builds on turn 1's file."""
    (episode,) = await run(
        "checkpoint-resume-v1",
        harness=harness,
        runtime=harness_runtime,
        env={"agent": {"retries": RETRIES}},
        max_turns=8,
        max_tokens=4096,
    )
    (trace,) = episode.traces
    assert trace.ok, [error.message for error in trace.errors]
    assert trace.reward == 1.0
    assert [error.type for error in trace.errors] == ["ProviderError"]
    # The failure hit as turn FAIL_AFTER_TURNS + 1 went upstream, so the newest
    # checkpoint is the one taken as turn FAIL_AFTER_TURNS was asked for: the box then
    # held the earlier turns' effects, and the trace kept those turns.
    assert trace.agent.runtime is not None
    restored_from = trace.agent.runtime.checkpoint
    assert restored_from is not None
    assert restored_from in [checkpoint.id for checkpoint in trace.checkpoints]
    assert trace.num_turns >= FAIL_AFTER_TURNS
    assert len(trace.checkpoints) >= trace.num_turns
    # The error belongs to the first attempt, in the agent stage; the retry resumed
    # from the restored checkpoint.
    assert trace.has_resumed
    assert [c.id for c in trace.checkpoints if c.resumed] == [restored_from]
    assert [(error.attempt, error.stage) for error in trace.errors] == [(0, "agent")]
    # Every turn kept has its call; the failure never went upstream, so no call
    # records it.
    assert [call.node for call in trace.calls] == [
        i for i, node in enumerate(trace.nodes) if node.sampled
    ]
