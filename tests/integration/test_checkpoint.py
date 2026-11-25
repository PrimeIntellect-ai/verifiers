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
import time
from contextlib import asynccontextmanager

import pytest
from conftest import _eval_config
from grid import grid
from prime_sandboxes import AsyncSandboxClient
from subagent_restore import sub_agent_turns
from tick_restore import COMMAND, TURNS

from verifiers.v1.clients import ModelContext
from verifiers.v1.dialects.chat import ChatDialect
from verifiers.v1.env import RunSlot
from verifiers.v1.restore import LIVE, release_all, restore_entry, save_all
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
async def command_model(request):
    """Replay one tool call per turn, then finish. Retries receive the same reply."""
    from aiohttp import web

    calls = getattr(request, "param", [("bash", {"command": COMMAND})] * TURNS)
    dialect = ChatDialect()

    async def answer(request: web.Request) -> web.StreamResponse:
        body = await request.json()
        turn = sum(m.get("role") == "assistant" for m in body["messages"]) + 1
        if turn <= len(calls):
            name, arguments = calls[turn - 1]
            call = {
                "id": f"call_{turn}",
                "type": "function",
                "function": {
                    "name": name,
                    "arguments": json.dumps(arguments),
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
                    "finish_reason": "tool_calls" if turn <= len(calls) else "stop",
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
    default the rollout is past its first turn: the box holds work the trace recorded.
    `save_all` returns nothing until the harness program runs, so it is asked again."""
    env = load_environment(config.env)
    (task,) = list(env.taskset)
    async with env.serving():
        (slot,) = env.slots(task)
        running = asyncio.ensure_future(env.run_slot(slot, ctx))
        try:
            async with asyncio.timeout(600):
                while True:
                    assert not running.done(), "the rollout ended before the save"
                    if slot.traces and ready(slot.traces[0]):
                        points = await save_all()
                        if points:
                            break
                    await asyncio.sleep(0.2)
            (point,) = points
        finally:
            running.cancel()
            await asyncio.gather(running, return_exceptions=True)
    return slot, point


@pytest.mark.parametrize("harness,harness_runtime", grid(needs=NEEDS), indirect=True)
async def test_retry_restores_checkpoint(run, taken, harness, harness_runtime):
    """A timed capture lets a failed harness resume in its original process."""
    shell = harness_class(harness.id).EXECUTES_CODE
    fail_turn = 2 if shell else 1
    trace = await restored_trace(
        run, harness, harness_runtime, fail_turn=fail_turn, interval=15
    )
    if shell:
        assert trace.num_turns >= 2
        assert trace.metrics["copied"] == 1.0
    assert trace.agent.runtime.checkpoint in taken
    await assert_no_leaks(taken)


@pytest.mark.parametrize("harness,harness_runtime", grid(needs=NEEDS), indirect=True)
async def test_another_owner_continues_a_saved_rollout(taken, harness, harness_runtime):
    """`save_all` checkpoints a running rollout for its owner. The rollout is then
    cancelled with everything it holds, and its restore point goes through a pickle. A
    second env, with a server, a tunnel and secrets of its own, continues the rollout
    from the point: in a new box, with the harness program and the task's process
    still running. A harness with a shell is saved after its first command and takes
    more turns in the new box; one without has a single turn and is saved during it."""
    shell = harness_class(harness.id).EXECUTES_CODE
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

    slot, point = await saved_rollout(
        config, ctx, ready=(lambda trace: trace.num_turns >= 1) if shell else bool
    )
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
    if shell:
        assert trace.metrics["copied"] == 1.0, [
            (
                n.message.role,
                n.sampled,
                str(n.message.model_dump(exclude_none=True))[:600],
            )
            for n in trace.nodes
        ]
        assert trace.num_turns > turns_saved >= 1
    assert trace.has_restored and trace.attempt == 2
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


@asynccontextmanager
async def running_slot(env, slot, ctx):
    """Keep the rollout alive through the assertions and tear it down on any exit."""
    async with env.serving(), asyncio.timeout(600):
        running = asyncio.create_task(env.run_slot(slot, ctx))
        try:
            yield running
        finally:
            release_all()
            running.cancel()
            await asyncio.gather(running, return_exceptions=True)


async def live_rollout(slot, running, ready):
    while not running.done():
        for rollout in list(LIVE):
            if slot.traces and rollout.trace is slot.traces[0] and ready(rollout.trace):
                return rollout
        await asyncio.sleep(0.25)
    pytest.fail(f"Rollout ended before the expected state: {running.result().errors}")


BLOCKED_CELL = """from pathlib import Path
import time
with open("nonce.fifo") as f:
    nonce = f.read()
with open("cell-started", "x") as f:
    f.write("started")
while not Path("continue-cell").exists():
    time.sleep(0.25)
cell_finished = True"""
RLM_CALLS = [
    ("ipython", {"code": BLOCKED_CELL, "timeout": 600}),
    ("ipython", {"code": 'assert cell_finished; Path("done.txt").write_text(nonce)'}),
]


@pytest.mark.parametrize(
    "harness,harness_runtime", grid(["rlm"], needs=NEEDS), indirect=True
)
@pytest.mark.parametrize("command_model", [RLM_CALLS], indirect=True, ids=["ipython"])
@pytest.mark.parametrize(
    "already_owned", [False, True], ids=["recent", "already-pinned"]
)
async def test_held_save_inside_rlm_cell(
    taken, harness, harness_runtime, command_model, already_owned
):
    """Fork a timed snapshot with an IPython cell on the stack. Its continuation
    must finish that cell once and read the same kernel variable in the next cell."""
    config = _eval_config(
        "memory-restore-v1",
        harness=harness,
        runtime=harness_runtime,
        taskset_overrides={"task": {"fail_turn": 99}},
        env={"agent": {"checkpoint": {"interval": 150}, "retries": {"max_retries": 0}}},
        client=command_model,
        rollout_timeout=600,
    )
    ctx = ModelContext(
        client=config.client, model=config.model, sampling=config.sampling
    )
    env = load_environment(config.env)
    (task,) = list(env.taskset)
    owned = []
    try:
        (slot,) = env.slots(task)
        async with running_slot(env, slot, ctx) as running:
            rollout = await live_rollout(
                slot, running, lambda trace: trace.num_turns == 1
            )
            while (
                await rollout.runtime.run(["test", "-s", "cell-started"], {})
            ).exit_code:
                assert not running.done(), (
                    "the first IPython cell exited before the save"
                )
                await asyncio.sleep(1)
            started = time.time()
            while not (
                recent := [
                    c
                    for c in rollout.trace.checkpoints
                    if c.time.start >= started and c.time.end
                ]
            ):
                if running.done():
                    pytest.fail(
                        f"Rollout ended before its timed snapshot: {running.result().traces[0].errors}"
                    )
                await asyncio.sleep(0.25)
            assert rollout.trace.num_turns == 1
            assert not recent[-1].pinned
            before = taken.copy()
            (point,) = await save_all(hold=True)
            entry = restore_entry(point)
            owned.append(Snapshot(id=entry["id"]))
            assert entry["pinned"] and entry["id"] == recent[-1].id
            assert taken == before
            if already_owned:
                # A second trainer checkpoint needs independent ownership. Either
                # trainer checkpoint can expire while the other still owns its VM.
                release_all()
                (point,) = await save_all(hold=True)
                entry = restore_entry(point)
                owned.append(Snapshot(id=entry["id"]))
                assert entry["pinned"] and taken == [*before, entry["id"]]
                assert entry["id"] != recent[-1].id
                async with AsyncSandboxClient() as client:
                    # A save returns before the upload; it becomes durable after.
                    durable = await client.wait_for_checkpoint(
                        entry["id"], timeout_seconds=600
                    )
                    assert durable.state == "DURABLE"

        second = load_environment(config.env)
        (task,) = list(second.taskset)
        fork = RunSlot(task, id=slot.id, restore=pickle.loads(pickle.dumps(point)))
        async with running_slot(second, fork, ctx) as running:
            restored = await live_rollout(
                fork,
                running,
                lambda trace: trace.has_restored and trace.agent.runtime.id,
            )
            await restored.runtime.write("continue-cell", b"go")
            (trace,) = (await running).traces
        assert trace.ok, trace.errors
        assert trace.agent.runtime.checkpoint == entry["id"]
        assert trace.reward == trace.metrics["copied"] == 1.0
        assert trace.num_turns == 3
    finally:
        if owned:
            await PrimeBackend.discard(owned)
    await assert_no_leaks(taken)
