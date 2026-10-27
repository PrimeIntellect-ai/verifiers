"""A retry restores the box's newest filesystem checkpoint (checkpoint-restore-v1: the
first attempt dies as turn 3 goes upstream; the retry restores the turn-2 snapshot,
keeps the turns before it, and finishes in a new harness process). Every harness that
declares `SUPPORTS_CHECKPOINT_RESTORE` runs it on prime, the one runtime with
checkpoints; the others are collected as skips."""

import pytest
from checkpoint_restore_v1 import FAIL_AFTER_TURNS
from grid import grid

from verifiers.v1.harness import RESTORE_PROMPT
from verifiers.v1.types import content_text

RETRIES = {"max_retries": 0, "rules": [{"type": "ProviderError", "max_retries": 1}]}


@pytest.mark.parametrize(
    "harness,harness_runtime",
    grid(needs=("restore", "checkpoint", "mcp")),
    indirect=True,
)
async def test_retry_restores_checkpoint(run, harness, harness_runtime):
    """The box restores to the state before the turn that failed, the turns up to
    there stay sampled, the task's `setup` (a nonce the prompt never reveals) is not
    rerun, and the harness is told about the restore — the restored attempt builds on
    the earlier tool call's file."""
    (episode,) = await run(
        "checkpoint-restore-v1",
        harness=harness,
        runtime=harness_runtime,
        env={"agent": {"retries": RETRIES}},
        max_turns=12,
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
    assert [c.id for c in trace.checkpoints if c.restored] == [restored_from]
    assert trace.has_restored
    assert trace.num_turns >= FAIL_AFTER_TURNS
    # The error belongs to the first attempt, in the agent stage.
    assert [(error.attempt, error.stage) for error in trace.errors] == [(0, "agent")]
    # The restored process got the restore notice as a user turn.
    assert any(
        node.message.role == "user"
        and RESTORE_PROMPT in content_text(node.message.content)
        for node in trace.nodes
    )
    # Every turn kept has its call; the dropped turns' calls stay, unlinked, so usage
    # counts them.
    assert [call.node for call in trace.calls if call.node is not None] == [
        i for i, node in enumerate(trace.nodes) if node.sampled
    ]
    assert any(call.node is None for call in trace.calls)
