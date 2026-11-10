"""A retry restores the box's newest memory snapshot (memory-restore-v1: the reply of one
model turn fails once; the retry boots a new box from the memory snapshot of that turn,
and the harness program the box still runs gets the reply and finishes).

Every harness that continues from a memory snapshot (`Harness.SUPPORTS_MEMORY_RESTORE`)
runs it on prime, the one runtime with memory snapshots. A process the task's `setup`
started holds a nonce only in its memory: the reward reads it from the restored box, so
a box that booted from the snapshot's files alone would fail. Every case checks that the
run leaks no snapshot."""

import pytest
from grid import grid
from prime_sandboxes import AsyncSandboxClient

from verifiers.v1.harness import RESTORE_PROMPT
from verifiers.v1.runtimes import PrimeBackend
from verifiers.v1.types import content_text
from verifiers.v1.utils.loaders import harness_class

RETRIES = {"max_retries": 0, "rules": [{"type": "ProviderError", "max_retries": 1}]}
NEEDS = ("memory",)
SHELLS = ["bash", "rlm"]


@pytest.fixture
def taken(monkeypatch) -> list[tuple[str, str]]:
    """The id and type of every snapshot the run's boxes take, in every attempt."""
    snapshots: list[tuple[str, str]] = []
    snapshot = PrimeBackend.snapshot

    async def recording(self, **kwargs):
        result = await snapshot(self, **kwargs)
        snapshots.append((result.id, result.type))
        return result

    monkeypatch.setattr(PrimeBackend, "snapshot", recording)
    return snapshots


async def assert_no_leaks(taken: list[tuple[str, str]]) -> None:
    """Every snapshot the run took is deleted on the platform."""
    leaked = {}
    async with AsyncSandboxClient() as client:
        for snapshot_id, _ in taken:
            try:
                state = (await client.get_checkpoint(snapshot_id)).state
            except Exception:  # noqa: BLE001 - a deleted snapshot is not found
                continue
            if state != "DELETING":
                leaked[snapshot_id] = state
    assert leaked == {}


async def restored_trace(
    run, harness, harness_runtime, *, fail_turn: int, **checkpoint
):
    (episode,) = await run(
        "memory-restore-v1",
        harness=harness,
        runtime=harness_runtime,
        taskset_overrides={"task": {"fail_turn": fail_turn}},
        env={"agent": {"retries": RETRIES, "checkpoint": checkpoint}},
        max_turns=12,
        max_tokens=4096,
    )
    (trace,) = episode.traces
    assert trace.ok, [error.message for error in trace.errors]
    # The process `setup` started kept its memory across the restore.
    assert trace.reward == 1.0
    assert [error.type for error in trace.errors] == ["ProviderError"]
    assert [(error.attempt, error.stage) for error in trace.errors] == [(1, "agent")]
    assert trace.attempt == 2
    assert trace.has_restored
    assert trace.agent.runtime is not None
    restored_from = trace.agent.runtime.checkpoint
    assert [c.id for c in trace.checkpoints if c.restored] == [restored_from]
    assert {c.snapshot.type for c in trace.checkpoints} == {"memory"}
    # The harness program never restarted: one root, and no restore notice.
    assert sum(node.parent is None for node in trace.nodes) == 1
    assert not any(
        node.message.role == "user"
        and RESTORE_PROMPT in content_text(node.message.content)
        for node in trace.nodes
    )
    # Every turn kept has its call; the failed turn's call stays, unlinked.
    assert [call.node for call in trace.calls if call.node is not None] == [
        i for i, node in enumerate(trace.nodes) if node.sampled
    ]
    assert any(call.node is None for call in trace.calls)
    return trace


@pytest.mark.parametrize("harness,harness_runtime", grid(needs=NEEDS), indirect=True)
async def test_retry_restores_memory_checkpoint(run, taken, harness, harness_runtime):
    """A snapshot at every turn. A harness with a shell fails at its second turn, so the
    restored box holds the first turn's command and its harness goes on to the second;
    a harness without one has a single turn, which fails."""
    shell = harness_class(harness.id).EXECUTES_CODE
    trace = await restored_trace(
        run, harness, harness_runtime, fail_turn=2 if shell else 1, interval=1
    )
    if shell:
        assert trace.num_turns >= 2
        assert trace.metrics["copied"] == 1.0
    assert {kind for _, kind in taken} == {"memory"}
    await assert_no_leaks(taken)


@pytest.mark.parametrize(
    "harness,harness_runtime", grid(["bash"], needs=NEEDS), indirect=True
)
async def test_filesystem_snapshots_stay_selectable(
    run, taken, harness, harness_runtime
):
    """`snapshot = "filesystem"` takes no memory snapshot: the retry boots from the
    snapshot's files, where the process that held the nonce is gone."""
    (episode,) = await run(
        "memory-restore-v1",
        harness=harness,
        runtime=harness_runtime,
        env={
            "agent": {
                "retries": RETRIES,
                "checkpoint": {"interval": 1, "snapshot": "filesystem"},
            }
        },
        max_turns=12,
        max_tokens=4096,
    )
    (trace,) = episode.traces
    assert trace.has_restored
    assert {kind for _, kind in taken} == {"filesystem"}
    assert trace.reward == 0.0
    await assert_no_leaks(taken)
