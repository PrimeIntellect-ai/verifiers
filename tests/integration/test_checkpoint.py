"""A retry restores the box's newest snapshot (checkpoint-restore-v1: the first attempt
dies as turn 3 goes upstream; the retry restores the newest snapshot, keeps the turns
before it, and finishes in a new harness process).

Every harness that can continue in a new process (`Harness.can_restore`) runs it on
prime, the one runtime with snapshots; the others are collected as skips. The retention policies run
on one harness: what they keep or delete does not depend on it. Every case checks that
the run leaks no snapshot: each one its boxes took is deleted on the platform once the
run is final."""

import pytest
from checkpoint_restore_v1 import FAIL_AFTER_TURNS
from grid import grid
from prime_sandboxes import AsyncSandboxClient
from skill_codeword import SKILL

from verifiers.v1.harness import RESTORE_PROMPT
from verifiers.v1.runtimes import PrimeBackend
from verifiers.v1.types import content_text

RETRIES = {"max_retries": 0, "rules": [{"type": "ProviderError", "max_retries": 1}]}
NEEDS = ("snapshots", "mcp")


def checkpoint_env(**checkpoint) -> dict:
    return {"agent": {"retries": RETRIES, "checkpoint": checkpoint}}


@pytest.fixture
def taken(monkeypatch) -> list[str]:
    """The ids of every snapshot the run's boxes take, in every attempt: also the
    ones a restore rewinds past and the ones that never become restorable."""
    ids: list[str] = []
    snapshot = PrimeBackend.snapshot

    async def recording(self):
        result = await snapshot(self)
        ids.append(result.id)
        return result

    monkeypatch.setattr(PrimeBackend, "snapshot", recording)
    return ids


async def platform_state(snapshot_id: str) -> str | None:
    """The snapshot's state on prime, or None once it is gone."""
    async with AsyncSandboxClient() as client:
        try:
            return (await client.get_checkpoint(snapshot_id)).state
        except Exception:  # noqa: BLE001 - a deleted snapshot is not found
            return None


async def assert_no_leaks(taken: list[str]) -> None:
    """Every snapshot the run took is deleted."""
    states = {snapshot_id: await platform_state(snapshot_id) for snapshot_id in taken}
    leaked = {
        snapshot_id: state
        for snapshot_id, state in states.items()
        if state not in (None, "DELETING")
    }
    assert leaked == {}


@pytest.mark.parametrize("harness,harness_runtime", grid(needs=NEEDS), indirect=True)
async def test_retry_restores_checkpoint(run, taken, harness, harness_runtime):
    """The box restores to the state before the turn that failed, the turns up to
    there stay sampled, the task's `setup` (a nonce the prompt never reveals) is not
    rerun, and the harness is told about the restore — the restored attempt builds on
    the earlier tool call's file. A snapshot every turn, so the failure at turn 3
    restores a snapshot with turns behind it."""
    trace = await restored_trace(run, harness, harness_runtime, interval=1)
    assert trace.num_turns >= FAIL_AFTER_TURNS
    await assert_no_leaks(taken)


@pytest.mark.parametrize(
    "harness,harness_runtime", grid(["bash"], needs=NEEDS), indirect=True
)
async def test_a_short_rollout_takes_no_snapshot(run, taken, harness, harness_runtime):
    """With the default interval (every 10th turn), a rollout that fails at turn 3
    has taken no snapshot: its retry starts over and still succeeds."""
    (episode,) = await run(
        "checkpoint-restore-v1",
        harness=harness,
        runtime=harness_runtime,
        env=checkpoint_env(),
        max_turns=12,
        max_tokens=4096,
    )
    (trace,) = episode.traces
    assert trace.ok, [error.message for error in trace.errors]
    assert trace.reward == 1.0
    assert [error.type for error in trace.errors] == ["ProviderError"]
    assert trace.attempt == 2
    assert not trace.has_restored
    assert trace.checkpoints == []
    assert taken == []


@pytest.mark.parametrize(
    "harness,harness_runtime",
    grid(["rlm"], needs=(*NEEDS, "skills")),
    indirect=True,
)
async def test_rlm_with_skills_restores(run, taken, harness, harness_runtime):
    """rlm keeps its session in a per-run skill home when it has skills: the retry
    must load the session from the same home the restored box holds."""
    config = harness.model_copy(update={"skills": [SKILL]})
    trace = await restored_trace(run, config, harness_runtime, interval=1)
    assert trace.num_turns >= FAIL_AFTER_TURNS
    await assert_no_leaks(taken)


@pytest.mark.parametrize("keep", ["last", 2], ids=["keep-last", "keep-every-2nd"])
@pytest.mark.parametrize(
    "harness,harness_runtime", grid(["bash"], needs=NEEDS), indirect=True
)
async def test_checkpoint_retention(run, taken, harness, harness_runtime, keep):
    """Snapshots the policy drops are deleted while the rollout runs, without breaking
    the newest one: the retry still restores it. Every snapshot is deleted once the
    run is final."""
    trace = await restored_trace(run, harness, harness_runtime, interval=1, keep=keep)
    assert all(checkpoint.deleted for checkpoint in trace.checkpoints)
    await assert_no_leaks(taken)


async def restored_trace(run, harness, harness_runtime, **checkpoint):
    (episode,) = await run(
        "checkpoint-restore-v1",
        harness=harness,
        runtime=harness_runtime,
        env=checkpoint_env(**checkpoint),
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
    # Each checkpoint records what its snapshot captured.
    assert {c.snapshot.type for c in trace.checkpoints} == {"filesystem"}
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
