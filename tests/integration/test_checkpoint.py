"""A retry restores the box's newest memory snapshot (memory-restore-v1: the reply of one
model turn fails once and the box is snapshotted before the harness gets the error; the
retry continues a new box from that snapshot, and the harness program the box still runs
gets the reply and finishes).

Every harness that continues from a memory snapshot (`Harness.SUPPORTS_MEMORY_RESTORE`)
runs it on prime, the one runtime with memory snapshots. A process the task's `setup`
started holds a nonce only in its memory: the reward reads it from the restored box, so
a box that booted from files alone would fail. Every case checks that the run leaks no
snapshot."""

import asyncio
import json
import pickle

import pytest
from conftest import _eval_config
from grid import grid
from prime_sandboxes import AsyncSandboxClient
from subagent_restore import sub_agent_turns
from tick_restore import COMMAND, TURNS

from verifiers.v1.clients import ModelContext
from verifiers.v1.dialects.chat import ChatDialect
from verifiers.v1.env import RunSlot
from verifiers.v1.restore import restore_entry, save_all
from verifiers.v1.runtimes import PrimeBackend, Snapshot
from verifiers.v1.utils.loaders import harness_class, load_environment

RETRIES = {"max_retries": 0, "rules": [{"type": "ProviderError", "max_retries": 1}]}
NEEDS = ("memory",)
SHELLS = ["bash", "rlm"]
# rlm with its guidance on how to hand work to sub-agents.
SUB_AGENTS = {"id": "rlm", "delegation_prompt": True}


@pytest.fixture
def taken(monkeypatch) -> list[str]:
    """The id of every snapshot the run's boxes take, in every attempt."""
    snapshots: list[str] = []
    snapshot = PrimeBackend.snapshot

    async def recording(self):
        result = await snapshot(self)
        snapshots.append(result.id)
        return result

    monkeypatch.setattr(PrimeBackend, "snapshot", recording)
    return snapshots


@pytest.fixture
async def command_model():
    """A scripted model for the bash harness: its k-th reply asks for the tick task's
    command, and the reply after the last command says done. A box that asks again for
    a turn gets the same reply. Yields a client config that points at it."""
    from aiohttp import web

    dialect = ChatDialect()

    async def answer(request: web.Request) -> web.StreamResponse:
        body = await request.json()
        turn = sum(m.get("role") == "assistant" for m in body["messages"]) + 1
        if turn <= TURNS:
            call = {
                "id": f"call_{turn}",
                "type": "function",
                "function": {
                    "name": "bash",
                    "arguments": json.dumps({"command": COMMAND}),
                },
            }
            message = {"role": "assistant", "content": None, "tool_calls": [call]}
        else:
            message = {"role": "assistant", "content": "done"}
        raw = {
            "id": f"chatcmpl-{turn}",
            "object": "chat.completion",
            "created": 0,
            "model": str(body.get("model", "scripted")),
            "choices": [
                {
                    "index": 0,
                    "message": message,
                    "finish_reason": "tool_calls" if turn <= TURNS else "stop",
                }
            ],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
        }
        if not dialect.streaming(body):
            return web.json_response(raw)
        response = web.StreamResponse(headers={"Content-Type": "text/event-stream"})
        await response.prepare(request)
        for event in dialect.stream_events(raw):
            await response.write(event)
        await response.write_eof()
        return response

    app = web.Application()
    app.router.add_post("/{tail:.*}", answer)
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, "127.0.0.1", 0)
    await site.start()
    port = site._server.sockets[0].getsockname()[1]
    try:
        yield {"base_url": f"http://127.0.0.1:{port}/v1"}
    finally:
        await runner.cleanup()


async def platform_state(client: AsyncSandboxClient, snapshot_id: str) -> str | None:
    """The snapshot's state on prime, or None once it is gone."""
    try:
        return (await client.get_checkpoint(snapshot_id)).state
    except Exception:  # noqa: BLE001 - a deleted snapshot is not found
        return None


async def assert_no_leaks(taken: list[str]) -> None:
    """Every snapshot the run took is deleted on the platform."""
    leaked = {}
    async with AsyncSandboxClient() as client:
        for snapshot_id in taken:
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
        env={
            "agent": {
                "retries": RETRIES,
                "checkpoint": checkpoint,
            }
        },
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
    # The harness program never restarted: one root.
    assert sum(node.parent is None for node in trace.nodes) == 1
    # Every turn kept has its call; the failed turn's call stays, unlinked.
    assert [call.node for call in trace.calls if call.node is not None] == [
        i for i, node in enumerate(trace.nodes) if node.sampled
    ]
    assert any(call.node is None for call in trace.calls)
    return trace


async def saved_rollout(config, ctx, ready=lambda trace: trace.num_turns >= 1):
    """Run the config's one rollout until `ready(trace)`, checkpoint it with `save_all`,
    and cancel it with everything it holds. Returns its slot and its restore point. By
    default the rollout is past its first turn: the box holds work the trace recorded."""
    env = load_environment(config.env)
    (task,) = list(env.taskset)
    async with env.serving():
        (slot,) = env.slots(task)
        running = asyncio.ensure_future(env.run_slot(slot, ctx))
        try:
            async with asyncio.timeout(600):
                while not (slot.traces and ready(slot.traces[0])):
                    assert not running.done(), "the rollout ended before the save"
                    await asyncio.sleep(0.2)
            (point,) = await save_all()
        finally:
            running.cancel()
            await asyncio.gather(running, return_exceptions=True)
    return slot, point


@pytest.mark.parametrize("harness,harness_runtime", grid(needs=NEEDS), indirect=True)
async def test_retry_restores_checkpoint(run, taken, harness, harness_runtime):
    """The box is snapshotted at every model turn, and the retry restores the snapshot
    of the turn whose call failed. A harness with a shell fails at its second turn, so
    the restored box holds the first turn's command and its harness goes on to the
    second; a harness without one has a single turn, which fails. The calls that fail
    again at that turn take no further snapshot."""
    shell = harness_class(harness.id).EXECUTES_CODE
    fail_turn = 2 if shell else 1
    trace = await restored_trace(
        run, harness, harness_runtime, fail_turn=fail_turn, interval=1
    )
    if shell:
        assert trace.num_turns >= 2
        assert trace.metrics["copied"] == 1.0
    assert [c.id for c in trace.checkpoints if c.restored] == [taken[fail_turn - 1]]
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

    slot, point = await saved_rollout(config, ctx)
    turns_saved = len([node for node in point["nodes"] if node["sampled"]])
    point = pickle.loads(pickle.dumps(point))

    second = load_environment(config.env)
    (task,) = list(second.taskset)
    async with second.serving():
        episode = await second.run_slot(RunSlot(task, id=slot.id, restore=point), ctx)
    (trace,) = episode.traces
    assert trace.ok, [error.message for error in trace.errors]
    assert trace.id == point["id"]
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
    assert snapshot_id == restore_entry(point)["id"]
    async with AsyncSandboxClient() as client:
        assert (await client.get_checkpoint(snapshot_id)).state == "DURABLE"
    assert await PrimeBackend.discard([Snapshot(id=snapshot_id)]) == [snapshot_id]
    await assert_no_leaks(taken)


@pytest.mark.parametrize(
    "harness,harness_runtime", grid(SHELLS, needs=NEEDS), indirect=True
)
async def test_a_restore_point_that_leads_nowhere_starts_the_rollout_over(
    taken, harness, harness_runtime
):
    """A restore point whose snapshot the platform no longer restores costs the rollout
    no retry: it starts over under the same trace id and finishes."""
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

    slot, point = await saved_rollout(config, ctx)
    snapshot = restore_entry(point)["snapshot"]
    snapshot_id, snapshot["id"] = snapshot["id"], "no-such-checkpoint"

    second = load_environment(config.env)
    (task,) = list(second.taskset)
    async with second.serving():
        episode = await second.run_slot(RunSlot(task, id=slot.id, restore=point), ctx)
    (trace,) = episode.traces
    assert trace.ok, [error.message for error in trace.errors]
    assert trace.id == point["id"]
    assert trace.reward == 1.0
    assert trace.attempt == 3

    async with AsyncSandboxClient() as client, asyncio.timeout(300):
        while (await client.get_checkpoint(snapshot_id)).state != "DURABLE":
            await asyncio.sleep(2)
    assert await PrimeBackend.discard([Snapshot(id=snapshot_id)]) == [snapshot_id]
    await assert_no_leaks(taken)


@pytest.mark.parametrize(
    "harness,harness_runtime", grid([SUB_AGENTS], needs=NEEDS), indirect=True
)
async def test_another_owner_continues_running_sub_agents(
    taken, harness, harness_runtime
):
    """A rollout is saved while two sub-agents of its harness are at work (subagent-
    restore). The second owner's box continues both: neither is started again, each
    takes its next turns, and each reads its nonce from a process that only a box with
    its memory still runs."""
    config = _eval_config(
        "subagent-restore",
        harness=harness,
        runtime=harness_runtime,
        max_turns=40,
        max_tokens=4096,
        rollout_timeout=600,
    )
    ctx = ModelContext(
        client=config.client, model=config.model, sampling=config.sampling
    )
    slot, point = await saved_rollout(
        config, ctx, ready=lambda trace: all(sub_agent_turns(trace).values())
    )
    saved = len(point["nodes"])
    point = pickle.loads(pickle.dumps(point))

    second = load_environment(config.env)
    (task,) = list(second.taskset)
    async with second.serving():
        episode = await second.run_slot(RunSlot(task, id=slot.id, restore=point), ctx)
    (trace,) = episode.traces
    assert trace.ok, [error.message for error in trace.errors]
    assert trace.has_restored and trace.attempt == 2
    assert trace.reward == 1.0
    for name, turns in sub_agent_turns(trace).items():
        # At work when the box was saved, and at work again in the restored box.
        assert [turn for turn in turns if turn < saved], name
        assert [turn for turn in turns if turn >= saved], name
    # The coordinator and each sub-agent started once: three conversations.
    assert sum(node.parent is None for node in trace.nodes) == 3
    (snapshot_id,) = [c.id for c in trace.checkpoints if c.pinned]
    assert await PrimeBackend.discard([Snapshot(id=snapshot_id)]) == [snapshot_id]
    await assert_no_leaks(taken)


@pytest.mark.parametrize(
    "harness,harness_runtime", grid(["bash"], needs=NEEDS), indirect=True
)
async def test_every_saved_turn_continues_to_the_same_result(
    taken, harness, harness_runtime, command_model
):
    """One rollout is saved at its first, second and third turn and then finishes. A
    second owner continues each saved trace, and every continuation ends with the
    messages of the first run: a restore from any turn loses no command and repeats
    none. The model is scripted, so the turns after a restore are the same ones."""
    config = _eval_config(
        "tick-restore",
        harness=harness,
        runtime=harness_runtime,
        max_turns=12,
        max_tokens=4096,
        rollout_timeout=300,
        client=command_model,
    )
    ctx = ModelContext(
        client=config.client, model=config.model, sampling=config.sampling
    )
    env = load_environment(config.env)
    (task,) = list(env.taskset)
    points = []
    async with env.serving():
        (slot,) = env.slots(task)
        running = asyncio.ensure_future(env.run_slot(slot, ctx))
        try:
            for turn in range(1, TURNS):
                async with asyncio.timeout(300):
                    while not (slot.traces and slot.traces[0].num_turns >= turn):
                        assert not running.done(), "the rollout ended before the save"
                        await asyncio.sleep(0.1)
                (point,) = await save_all()
                points.append(pickle.loads(pickle.dumps(point)))
            episode = await running
        finally:
            running.cancel()
            await asyncio.gather(running, return_exceptions=True)
    (reference,) = episode.traces
    assert reference.ok, [error.message for error in reference.errors]
    assert reference.reward == 1.0

    def messages(trace):
        return [node.message.model_dump(exclude_none=True) for node in trace.nodes]

    async def continued(point):
        second = load_environment(config.env)
        (task,) = list(second.taskset)
        async with second.serving():
            episode = await second.run_slot(
                RunSlot(task, id=slot.id, restore=point), ctx
            )
        (trace,) = episode.traces
        return trace

    # The saves are at three different turns.
    saved_turns = [sum(node["sampled"] for node in p["nodes"]) for p in points]
    assert saved_turns == sorted(set(saved_turns)), saved_turns
    for point, trace in zip(points, await asyncio.gather(*map(continued, points))):
        assert trace.ok, [error.message for error in trace.errors]
        assert trace.has_restored
        assert trace.reward == 1.0
        assert sum(node.parent is None for node in trace.nodes) == 1
        assert messages(trace) == messages(reference)
    snapshots = [Snapshot(id=restore_entry(point)["id"]) for point in points]
    assert sorted(await PrimeBackend.discard(snapshots)) == sorted(
        snapshot.id for snapshot in snapshots
    )
    await assert_no_leaks(taken)
