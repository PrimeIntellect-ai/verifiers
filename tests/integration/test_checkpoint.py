"""A retry restores the box's newest filesystem checkpoint (checkpoint-restore-v1: the
first attempt dies as turn 3 goes upstream; the retry restores the turn-2 snapshot,
keeps the turns before it, and finishes in a new harness process). Every harness that
declares `SUPPORTS_CHECKPOINT_RESTORE` runs it on prime, the one runtime with
checkpoints; the others are collected as skips. The retention policies run on one
harness: what they keep or delete does not depend on it."""

import asyncio
import contextlib

import pytest
from checkpoint_restore_v1 import FAIL_AFTER_TURNS
from grid import grid, mark, runtime_config
from prime_sandboxes import AsyncSandboxClient

from verifiers.v1.harness import RESTORE_PROMPT
from verifiers.v1.rollout import kept
from verifiers.v1.types import content_text

RETRIES = {"max_retries": 0, "rules": [{"type": "ProviderError", "max_retries": 1}]}
CHECKPOINT = {"agent": {"retries": RETRIES, "checkpoint": {}}}


async def on_platform(checkpoint: str) -> bool:
    """Whether prime still holds `checkpoint`, restorable."""
    async with AsyncSandboxClient() as client:
        try:
            return (await client.get_checkpoint(checkpoint)).state == "DURABLE"
        except Exception:  # noqa: BLE001 - a deleted checkpoint is not found
            return False


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
        env=CHECKPOINT,
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


@pytest.mark.parametrize(
    "keep,after",
    [("last", "delete"), ("last", "keep"), ("all", "keep"), (2, "keep")],
    ids=["keep-last-delete", "keep-last", "keep-all", "keep-every-2nd"],
)
@pytest.mark.parametrize(
    "harness,harness_runtime",
    grid(["bash"], needs=("restore", "checkpoint", "mcp")),
    indirect=True,
)
async def test_checkpoint_retention(run, harness, harness_runtime, keep, after):
    """Snapshots the policy drops are deleted while the rollout runs, without breaking
    the newest one: the retry still restores it. When the run is final, `after` deletes
    the rest or leaves exactly what `keep` holds."""
    (episode,) = await run(
        "checkpoint-restore-v1",
        harness=harness,
        runtime=harness_runtime,
        env={
            "agent": {"retries": RETRIES, "checkpoint": {"keep": keep, "after": after}}
        },
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


@pytest.mark.parametrize(
    "harness,runtime,checkpoint",
    [
        pytest.param("null", {"type": "subprocess"}, {}, id="null-boxless"),
        pytest.param(
            "bash",
            runtime_config("prime", "bash"),
            {"keep": "all", "after": "keep"},
            marks=[mark("prime")],
            id="bash-in-prime",
        ),
    ],
)
async def test_restore_in_another_process(harness, runtime, checkpoint):
    """An orchestrator's resume: a client runs a rollout through an env server, holding
    the stream's partial trace and newest restore point (`EpisodeAssembly`); the run is
    cut (it fails once the counter reached 2); a new run restores the trace from that
    point, the counter at 1, and finishes it. The counter's
    typed state rides the restore point. The null harness on the subprocess runtime
    keeps nothing in its box, so its checkpoints are box-less. Bash on prime restores a
    box snapshot the client owns (`after = "keep"`): the cut run leaves its snapshots,
    and the client deletes them when done (`discard`)."""
    from conftest import _eval_config

    from verifiers.v1.serve import EnvClient, EnvServer, EpisodeAssembly, RestoreRequest

    config = _eval_config(
        "counter-tool",
        harness=harness,
        runtime=runtime,
        env={
            "agent": {
                "checkpoint": checkpoint,
                "retries": {"max_retries": 0, "rules": []},
            },
            "retries": {"max_retries": 0, "rules": []},
        },
        taskset_overrides={
            "task": {
                "stops": {
                    "fail_once_after_bumps": {
                        "fn": "counter_tool:fail_once_after_bumps"
                    }
                }
            }
        },
        max_turns=8,
    )
    server = EnvServer(config.env, address="tcp://127.0.0.1:0")
    serving = asyncio.create_task(server.run())
    client = EnvClient(address=server.address)
    task = next(iter(server.env.taskset))
    task_data = task.data.model_dump(mode="json")
    kwargs = {
        "client": config.client,
        "model": config.model,
        "sampling": config.sampling,
    }
    try:
        await client.wait_for_server_startup(timeout=30)
        # The first run fails once the counter moved; a failed attempt waits for its
        # snapshots, so its newest restore point streams before the run ends.
        assembly = EpisodeAssembly()
        cut = await client.run(task_data=task_data, assembly=assembly, **kwargs)
        assert not cut.ok
        ((trace_id, point),) = assembly.restores.items()
        assert point["state"]["count"] == 1
        restore = RestoreRequest(trace=assembly.traces[trace_id], point=point)
        episode = await client.run(task_data=task_data, restore=restore, **kwargs)
        snapshots = [c.id for c in episode.traces[0].checkpoints if c.box]
        if snapshots:
            assert sorted(await client.discard(snapshots)) == sorted(snapshots)
    finally:
        await client.close()
        serving.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await serving

    (trace,) = episode.traces
    assert trace.id == trace_id
    assert trace.ok, [error.message for error in trace.errors]
    assert trace.reward == 1.0
    assert trace.has_restored
    assert [c.id for c in trace.checkpoints if c.restored] == [
        point["checkpoint"]["id"]
    ]
    assert all(c.box == (harness == "bash") for c in trace.checkpoints)
    if harness == "bash":
        assert trace.agent.runtime.checkpoint == point["checkpoint"]["id"]
