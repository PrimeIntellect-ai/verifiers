"""A retry restores the box's newest memory snapshot (memory-restore-v1: the reply of one
model turn fails once; the retry boots a new box from the memory snapshot of that turn,
and the harness program the box still runs gets the reply and finishes).

Every harness that continues from a memory snapshot (`Harness.SUPPORTS_MEMORY_RESTORE`)
runs it on prime, the one runtime with memory snapshots. A process the task's `setup`
started holds a nonce only in its memory: the reward reads it from the restored box, so
a box that booted from the snapshot's files alone would fail. Every case checks that the
run leaks no snapshot."""

import asyncio
import pickle

import pytest
from conftest import _eval_config
from grid import grid
from prime_sandboxes import AsyncSandboxClient

from verifiers.v1.clients import ModelContext
from verifiers.v1.env import RunSlot
from verifiers.v1.harness import RESTORE_PROMPT
from verifiers.v1.restore import save_all
from verifiers.v1.runtimes import PrimeBackend, Snapshot
from verifiers.v1.types import content_text
from verifiers.v1.utils.loaders import harness_class, load_environment

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


async def platform_state(client: AsyncSandboxClient, snapshot_id: str) -> str | None:
    """The snapshot's state on prime, or None once it is gone."""
    try:
        return (await client.get_checkpoint(snapshot_id)).state
    except Exception:  # noqa: BLE001 - a deleted snapshot is not found
        return None


async def assert_no_leaks(taken: list[tuple[str, str]]) -> None:
    """Every snapshot the run took is deleted on the platform."""
    leaked = {}
    async with AsyncSandboxClient() as client:
        for snapshot_id, _ in taken:
            state = await platform_state(client, snapshot_id)
            if state not in (None, "DELETING"):
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
    if not trace.errors:
        # The harness handled the failed reply itself and finished its run: nothing
        # failed, so nothing was restored.
        pytest.skip(f"{harness.id} recovers from a failed model call by itself")
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


@pytest.mark.parametrize(
    "harness,harness_runtime", grid(SHELLS, needs=NEEDS), indirect=True
)
async def test_another_owner_continues_a_saved_rollout(taken, harness, harness_runtime):
    """`save_all` checkpoints a running rollout for its owner. The rollout is then
    cancelled with everything it holds, and its restore point goes through a pickle. A
    second env, with a server, a tunnel and secrets of its own, continues the rollout
    from the point: in a new box, with the harness program and the task's process
    still running."""
    config = _eval_config(
        "memory-restore-v1",
        harness=harness,
        runtime=harness_runtime,
        taskset_overrides={"task": {"fail_turn": 99}},
        max_turns=12,
        max_tokens=4096,
    )
    ctx = ModelContext(
        client=config.client, model=config.model, sampling=config.sampling
    )

    first = load_environment(config.env)
    (task,) = list(first.taskset)
    async with first.serving():
        (slot,) = first.slots(task)
        running = asyncio.ensure_future(first.run_slot(slot, ctx))
        try:
            # Past its first turn: the box holds work the trace recorded.
            async with asyncio.timeout(600):
                while not (slot.traces and slot.traces[0].num_turns >= 1):
                    assert not running.done(), "the rollout ended before the save"
                    await asyncio.sleep(0.2)
            (point,) = await save_all()
        finally:
            running.cancel()
            await asyncio.gather(running, return_exceptions=True)
    turns_saved = len([node for node in point.trace["nodes"] if node["sampled"]])
    point = pickle.loads(pickle.dumps(point))

    second = load_environment(config.env)
    (task,) = list(second.taskset)
    async with second.serving():
        episode = await second.run_slot(RunSlot(task, id=slot.id, restore=point), ctx)
    (trace,) = episode.traces
    assert trace.ok, [error.message for error in trace.errors]
    assert trace.id == point.id
    assert trace.reward == 1.0
    assert trace.metrics["copied"] == 1.0, [
        (n.message.role, n.sampled, str(n.message.model_dump(exclude_none=True))[:600])
        for n in trace.nodes
    ]
    assert trace.has_restored and trace.attempt == 2
    assert trace.num_turns > turns_saved >= 1
    assert sum(node.parent is None for node in trace.nodes) == 1
    # The owner holds the snapshot: no rollout deleted it.
    (snapshot_id,) = [c.id for c in trace.checkpoints if c.pinned]
    assert snapshot_id == point.checkpoint
    async with AsyncSandboxClient() as client:
        assert (await client.get_checkpoint(snapshot_id)).state == "DURABLE"
    assert await PrimeBackend.discard([Snapshot(id=snapshot_id, type="memory")]) == [
        snapshot_id
    ]
    await assert_no_leaks(taken)
