"""A retry restores the box's newest snapshot (checkpoint-restore-v1: the first attempt
dies as turn 3 goes upstream; the retry restores the newest snapshot, keeps the turns
before it, and finishes in a new harness process).

Every case runs once per snapshot type (`SnapshotType`), on each harness and runtime
that both declare `SUPPORTS_<TYPE>_SNAPSHOTS`; the other cells are collected as skips.
A new snapshot type adds its cases without a change here. The retention policies run
on one harness: what they keep or delete does not depend on it."""

from collections.abc import Sequence
from typing import get_args

import pytest
from checkpoint_restore_v1 import FAIL_AFTER_TURNS
from grid import grid
from prime_sandboxes import AsyncSandboxClient
from skill_codeword import SKILL

from verifiers.v1.configs.runtime import SnapshotType
from verifiers.v1.harness import RESTORE_PROMPT
from verifiers.v1.rollout import kept
from verifiers.v1.types import content_text

RETRIES = {"max_retries": 0, "rules": [{"type": "ProviderError", "max_retries": 1}]}


def snapshot_grid(
    harnesses: Sequence[str] | None = None, needs: tuple[str, ...] = ("mcp",)
) -> list:
    """`(snapshot_type, harness, harness_runtime)` cells: the harness grid once per
    snapshot type, each cell needing that type's snapshots."""
    return [
        pytest.param(kind, *cell.values, marks=cell.marks, id=f"{kind}-{cell.id}")
        for kind in get_args(SnapshotType)
        for cell in grid(
            *([harnesses] if harnesses else []), needs=(f"{kind}-snapshots", *needs)
        )
    ]


def checkpoint_env(snapshot_type: str, **checkpoint) -> dict:
    return {
        "agent": {
            "retries": RETRIES,
            "checkpoint": {"type": snapshot_type, **checkpoint},
        }
    }


async def on_platform(checkpoint: str) -> bool:
    """Whether prime still holds `checkpoint`, restorable."""
    async with AsyncSandboxClient() as client:
        try:
            return (await client.get_checkpoint(checkpoint)).state == "DURABLE"
        except Exception:  # noqa: BLE001 - a deleted checkpoint is not found
            return False


@pytest.mark.parametrize(
    "snapshot_type,harness,harness_runtime",
    snapshot_grid(),
    indirect=["harness", "harness_runtime"],
)
async def test_retry_restores_checkpoint(run, snapshot_type, harness, harness_runtime):
    """The box restores to the state before the turn that failed, the turns up to
    there stay sampled, the task's `setup` (a nonce the prompt never reveals) is not
    rerun, and the harness is told about the restore — the restored attempt builds on
    the earlier tool call's file. A snapshot every turn, so the failure at turn 3
    restores a snapshot with turns behind it."""
    env = checkpoint_env(snapshot_type, interval=1)
    trace = await restored_trace(run, harness, harness_runtime, env)
    assert trace.num_turns >= FAIL_AFTER_TURNS


@pytest.mark.parametrize(
    "snapshot_type,harness,harness_runtime",
    snapshot_grid(["bash"]),
    indirect=["harness", "harness_runtime"],
)
async def test_default_config_restores(run, snapshot_type, harness, harness_runtime):
    """With the default interval (every 10th turn), the retry restores the turn-1
    snapshot: the box as `setup` left it, before any tool call."""
    env = checkpoint_env(snapshot_type)
    trace = await restored_trace(run, harness, harness_runtime, env)
    assert len(trace.checkpoints) >= 1


@pytest.mark.parametrize(
    "snapshot_type,harness,harness_runtime",
    snapshot_grid(["rlm"], needs=("mcp", "skills")),
    indirect=["harness", "harness_runtime"],
)
async def test_rlm_with_skills_restores(run, snapshot_type, harness, harness_runtime):
    """rlm keeps its session in a per-run skill home when it has skills: the retry
    must load the session from the same home the restored box holds."""
    config = harness.model_copy(update={"skills": [SKILL]})
    env = checkpoint_env(snapshot_type, interval=1)
    trace = await restored_trace(run, config, harness_runtime, env)
    assert trace.num_turns >= FAIL_AFTER_TURNS


async def restored_trace(run, harness, harness_runtime, env):
    (episode,) = await run(
        "checkpoint-restore-v1",
        harness=harness,
        runtime=harness_runtime,
        env=env,
        max_turns=12,
        max_tokens=4096,
    )
    (trace,) = episode.traces
    assert trace.ok, [error.message for error in trace.errors]
    assert trace.reward == 1.0
    assert [error.type for error in trace.errors] == ["ProviderError"]
    # The newest snapshot before the failed turn boots the retry's box.
    assert trace.agent.runtime is not None
    restored_from = trace.agent.runtime.checkpoint
    assert restored_from is not None
    assert [c.id for c in trace.checkpoints if c.restored] == [restored_from]
    assert trace.has_restored
    assert trace.attempt == 2
    # The error belongs to the first attempt, in the agent stage.
    assert [(error.attempt, error.stage) for error in trace.errors] == [(1, "agent")]
    # The restored process made no side request of its own (it would start a second
    # root), e.g. a model probe on session load.
    assert sum(node.parent is None for node in trace.nodes) == 1
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
    return trace


@pytest.mark.parametrize(
    "keep,after",
    [("last", "delete"), ("last", "keep"), ("all", "keep"), (2, "keep")],
    ids=["keep-last-delete", "keep-last", "keep-all", "keep-every-2nd"],
)
@pytest.mark.parametrize(
    "snapshot_type,harness,harness_runtime",
    snapshot_grid(["bash"]),
    indirect=["harness", "harness_runtime"],
)
async def test_checkpoint_retention(
    run, snapshot_type, harness, harness_runtime, keep, after
):
    """Snapshots the policy drops are deleted while the rollout runs, without breaking
    the newest one: the retry still restores it. When the run is final, `after` deletes
    the rest or leaves exactly what `keep` holds."""
    (episode,) = await run(
        "checkpoint-restore-v1",
        harness=harness,
        runtime=harness_runtime,
        env=checkpoint_env(snapshot_type, interval=1, keep=keep, after=after),
        max_turns=12,
        max_tokens=4096,
    )
    (trace,) = episode.traces
    assert trace.ok, [error.message for error in trace.errors]
    assert trace.reward == 1.0
    assert trace.has_restored
    live = [c.id for c in trace.checkpoints if not c.deleted]
    try:
        if after == "delete":
            assert live == []
        else:
            assert set(live) == kept(trace.checkpoints, keep)
            if keep == "last":
                assert len(live) == 1
            if keep == "all":
                # Snapshots still uploading when the run succeeded were never restorable.
                assert live == [c.id for c in trace.checkpoints if c.time.end]
        for checkpoint in trace.checkpoints:
            assert await on_platform(checkpoint.id) == (not checkpoint.deleted)
    finally:
        async with AsyncSandboxClient() as client:
            for checkpoint in live:
                await client.delete_checkpoint(checkpoint)
