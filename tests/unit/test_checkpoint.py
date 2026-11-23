"""Rollout checkpointing: when a rollout takes checkpoints, what `rewind` restores, and
what a restored attempt must not redo."""

import asyncio
import contextlib
import pickle
import time
from types import SimpleNamespace
from typing import Literal

import verifiers.v1 as vf
import verifiers.v1.agent as agent_module
from verifiers.v1.clients import ModelContext, ModelRoute
from verifiers.v1.configs.agent import RolloutCheckpointConfig
from verifiers.v1.configs.client import EvalClientConfig
from verifiers.v1.graph import MessageNode
from verifiers.v1.restore import open_restore, restorable, restore_entry, unsaveable
from verifiers.v1.rollout import Rollout, kept
from verifiers.v1.runtimes.backends.prime import PrimeBackend
from verifiers.v1.runtimes.base import Program
from verifiers.v1.session import IdempotentRequest, ReplayResponse, RolloutSession
from verifiers.v1.trace import (
    Checkpoint,
    InterceptRecord,
    ModelCall,
    TimeSpan,
    TraceTask,
)
from verifiers.v1.types import (
    AssistantMessage,
    ToolCall,
    ToolMessage,
    UserMessage,
)


def make_trace() -> vf.Trace:
    trace = vf.Trace(
        agent=vf.AgentInfo(config=vf.AgentConfig()),
        task=TraceTask(type="TaskData", data=vf.TaskData(idx=0, prompt="q")),
    )
    trace.nodes.append(MessageNode(parent=None, message=UserMessage(content="q")))
    trace.nodes.append(
        MessageNode(
            parent=0,
            message=AssistantMessage(
                content="",
                tool_calls=[ToolCall(id="c1", name="bash", arguments="{}")],
            ),
            sampled=True,
        )
    )
    trace.calls.append(ModelCall(node=1, time=TimeSpan(start=10.0, end=40.0)))
    return trace


def held(trace: vf.Trace, **fields) -> dict:
    """A checkpoint's session data (`RolloutSession.checkpoint`) for `trace` as is."""
    return {
        "state": trace.state.model_dump(mode="json"),
        "info": dict(trace.info),
        "rewrites": (len(trace.request_rewrites), len(trace.response_rewrites)),
        **fields,
    }


def test_rewind_restores_info_with_state():
    trace = make_trace()
    checkpoint = Checkpoint(
        id="ck",
        snapshot=vf.Snapshot(id="ck"),
        node=1,
        session=held(trace, info={"setup": "nonce"}),
    )
    trace.checkpoints.append(checkpoint)
    trace.info = {"setup": "nonce", "scoring_skipped": "stale"}
    trace.rewind(checkpoint)
    assert trace.info == {"setup": "nonce"}


def test_rewind_keeps_the_dropped_turns_calls():
    trace = make_trace()
    checkpoint = Checkpoint(id="ck", snapshot=vf.Snapshot(id="ck"), node=0)
    trace.checkpoints.append(checkpoint)
    trace.rewind(checkpoint)
    assert len(trace.nodes) == 1
    assert [call.node for call in trace.calls] == [None]


def test_rewind_drops_the_rewrite_records_of_the_turns_it_cuts():
    trace = make_trace()
    trace.request_rewrites.append(InterceptRecord(handler="before"))
    checkpoint = Checkpoint(
        id="ck1", snapshot=vf.Snapshot(id="ck1"), node=1, session=held(trace)
    )
    trace.checkpoints.append(checkpoint)
    trace.request_rewrites.append(InterceptRecord(handler="after"))
    trace.response_rewrites.append(InterceptRecord(handler="after"))
    trace.rewind(checkpoint)
    assert [r.handler for r in trace.request_rewrites] == ["before"]
    assert trace.response_rewrites == []


def test_rewind_drops_the_later_checkpoints_at_the_same_node():
    trace = make_trace()
    older = Checkpoint(id="older", node=1, snapshot=vf.Snapshot(id="older"))
    target = Checkpoint(id="target", node=1, snapshot=vf.Snapshot(id="target"))
    later = Checkpoint(id="later", node=1, snapshot=vf.Snapshot(id="later"))
    trace.checkpoints.extend([older, target, later])
    assert trace.rewind(target) == [vf.Snapshot(id="later")]
    assert [c.id for c in trace.checkpoints] == ["older", "target"]


def test_restored_span_counts_only_its_own_calls():
    trace = make_trace()
    trace.timing.agent.start, trace.timing.agent.end = 100.0, 120.0
    trace.calls.append(ModelCall(time=TimeSpan(start=105.0, end=110.0)))
    trace.split_agent_time()
    assert trace.timing.agent.model.duration == 5.0
    assert trace.timing.agent.harness.duration == 15.0


def test_agent_time_splits_off_the_checkpoint_hold():
    trace = make_trace()
    trace.timing.agent.start, trace.timing.agent.end = 100.0, 120.0
    trace.calls.append(ModelCall(time=TimeSpan(start=105.0, end=110.0)))
    trace.checkpoints += [
        Checkpoint(id="old", snapshot=vf.Snapshot(id="old"), captured=90.0, held=3.0),
        Checkpoint(id="new", snapshot=vf.Snapshot(id="new"), captured=106.0, held=1.5),
    ]
    trace.split_agent_time()
    assert trace.timing.agent.checkpoint.duration == 1.5
    assert trace.timing.agent.model.duration == 3.5
    assert trace.timing.agent.harness.duration == 15.0


class SlowBackend:
    """Stops the box for a snapshot until the test lets it run again; the snapshot
    becomes durable when the test says so."""

    def __init__(self) -> None:
        self.durable = asyncio.Event()
        self.stopped = asyncio.Event()
        """Set once a snapshot has stopped the box."""
        self.running = asyncio.Event()
        """A snapshot returns when this is set: the box runs again."""
        self.fail = False
        self.taken = 0
        self.deleted: list[str] = []

    def snapshot_pace(self):
        return contextlib.nullcontext()

    async def snapshot(self) -> vf.Snapshot:
        self.taken += 1
        self.stopped.set()
        await self.running.wait()
        return vf.Snapshot(id=f"ck{self.taken}")

    async def wait_snapshot(self, snapshot: vf.Snapshot) -> int | None:
        await self.durable.wait()
        if self.fail:
            raise RuntimeError("upload failed")
        return 1024

    async def discard_snapshots(self, snapshots: list[vf.Snapshot]) -> list[str]:
        ids = [snapshot.id for snapshot in snapshots]
        self.deleted += ids
        return ids


class FakeRelay:
    upstream = "https://tunnel.example"

    def __init__(self) -> None:
        self.state = {"dir": "/tmp/vf-relay/t", "url": "http://127.0.0.1:4000"}
        self.rerouted: list[str] = []

    async def reroute(self, upstream: str) -> None:
        self.rerouted.append(upstream)


def make_rollout(
    interval: int = 1,
    keep: Literal["last"] | int = "last",
    *,
    running: bool = True,
    stops: bool = False,
) -> tuple[Rollout, SlowBackend, RolloutSession]:
    """A rollout that takes snapshots, with or without a harness segment that runs.
    With `stops`, a snapshot keeps the box stopped until the test sets
    `backend.running`."""
    backend = SlowBackend()
    if not stops:
        backend.running.set()
    job = {"job_id": "j1"} if running else None
    run = Rollout.__new__(Rollout)
    run.trace = make_trace()  # one turn so far
    run.runtime = SimpleNamespace(backend=backend)  # type: ignore[assignment]
    session = RolloutSession(
        route=ModelRoute(ModelContext(model="m", client=EvalClientConfig())),
        trace=run.trace,
    )
    run._session = session
    run._harness_session = SimpleNamespace(  # type: ignore[assignment]
        memory_state=lambda: None if job is None else {"job": job}
    )
    run._relay = FakeRelay()  # type: ignore[assignment]
    run._secret = "box-secret"
    run._checkpointing = RolloutCheckpointConfig(interval=interval, keep=keep)
    run._durable = {}
    run._captures = set()
    run._closed = False
    run._opened = True
    run._memory = True
    run.unsaveable = None
    run.save_skipped = None
    run._held = False
    run._release_timer = None
    run._unheld = asyncio.Event()
    run._unheld.set()
    run._captured_turn = 0
    run._agent_time_spent = 0.0
    run._segment_start = None
    run._capturing = asyncio.Lock()
    run._pruning = asyncio.Lock()
    run.deleted_snapshots = set()
    run.rewound_snapshots = []
    run.restored_turns = None
    return run, backend, session


async def test_checkpoint_interval_skips_turns():
    run, backend, _ = make_rollout(interval=2)
    assert await run._checkpoint() is not None  # turn 2: the 2nd
    run.trace.nodes.append(MessageNode(parent=1, message=UserMessage(content="go")))
    run.trace.nodes.append(
        MessageNode(parent=2, message=AssistantMessage(content="ok"), sampled=True)
    )
    assert await run._checkpoint() is None  # turn 3
    assert backend.taken == 1


async def test_the_restored_turn_is_not_snapshotted_again():
    run, backend, _ = make_rollout(interval=2)
    run.restored_turns = run.trace.num_turns
    run._captured_turn = run.restored_turns + 1
    assert await run._checkpoint() is None
    assert backend.taken == 0


async def test_checkpoint_becomes_durable_in_the_background():
    run, backend, _ = make_rollout(interval=1)
    checkpoint = await run._checkpoint()
    assert checkpoint is not None and checkpoint.captured
    assert checkpoint.time.end == 0 and checkpoint.bytes is None
    backend.durable.set()
    await run._settle_checkpoints(wait=True)
    assert checkpoint.time.end and checkpoint.bytes == 1024
    assert run.trace.checkpoints == [checkpoint]


async def test_a_checkpoint_that_never_becomes_durable_is_dropped():
    run, backend, _ = make_rollout(interval=1)
    await run._checkpoint()
    backend.fail = True
    backend.durable.set()
    await run._settle_checkpoints(wait=True)
    assert run.trace.checkpoints == []


async def test_settling_without_a_retry_stops_waiting():
    run, _, _ = make_rollout(interval=1)
    checkpoint = await run._checkpoint()
    await asyncio.wait_for(run._settle_checkpoints(wait=False), timeout=1)
    assert run._durable == {}
    assert run.trace.checkpoints == [checkpoint] and checkpoint.time.end == 0


async def test_a_closing_run_takes_no_checkpoint():
    run, backend, _ = make_rollout(interval=1)
    capture = asyncio.create_task(run._checkpoint())
    run._closed = True
    await run._settle_checkpoints(wait=True)
    assert await capture is None
    assert backend.taken == 0 and run.trace.checkpoints == []


def durable(*ids: str, deleted: tuple[str, ...] = ()) -> list[Checkpoint]:
    return [
        Checkpoint(
            id=i,
            snapshot=vf.Snapshot(id=i),
            time=TimeSpan(start=1.0, end=2.0),
            deleted=i in deleted,
        )
        for i in ids
    ]


def test_keep_policies():
    checkpoints = durable("a", "b", "c", "d", "e")
    assert kept(checkpoints, "last") == {"e"}
    assert kept(checkpoints, 1) == {"a", "b", "c", "d", "e"}
    assert kept(checkpoints, 2) == {"a", "c", "e"}
    assert kept(checkpoints, 3) == {"a", "d", "e"}
    # Deleted and not yet durable snapshots are never kept.
    pending = [
        *durable("a", "b", deleted=("a",)),
        Checkpoint(id="c", snapshot=vf.Snapshot(id="c")),
    ]
    assert kept(pending, 1) == {"b"}


def test_an_owners_snapshots_do_not_shift_the_keep_policy():
    checkpoints = durable("a", "o1", "b", "o2", "c", "d", "e")
    for checkpoint in checkpoints:
        checkpoint.pinned = checkpoint.id.startswith("o")
    assert kept(checkpoints, 2) == {"a", "c", "e"}


async def test_a_durable_snapshot_prunes_the_older_ones():
    run, backend, _ = make_rollout(keep="last")
    run.trace.checkpoints = durable("old1", "old2")
    run.rewound_snapshots = [vf.Snapshot(id="rewound")]
    backend.durable.set()
    newest = await run._checkpoint()
    await run._settle_checkpoints(wait=True)
    assert sorted(backend.deleted) == ["old1", "old2", "rewound"]
    assert [c.id for c in run.trace.checkpoints if not c.deleted] == [newest.id]
    assert run.deleted_snapshots == {"old1", "old2", "rewound"}


def gated(harness: str, runtime: vf.RuntimeConfig, **checkpoint) -> vf.Agent:
    return vf.Agent(
        vf.AgentConfig(
            model="test-model",
            harness={"id": harness},
            runtime=runtime,
            checkpoint=RolloutCheckpointConfig(**checkpoint),
        )
    )


async def test_no_snapshot_at_turns_by_default():
    run, backend, _ = make_rollout()
    run._checkpointing = RolloutCheckpointConfig()
    assert await run._checkpoint() is None
    assert backend.taken == 0


async def test_a_call_asked_again_takes_no_second_snapshot():
    run, backend, _ = make_rollout(interval=1)
    first = await run._checkpoint()
    assert first is not None and backend.taken == 1
    # The same call comes again (the snapshot reset the box's connections).
    assert await run._checkpoint() is None
    assert backend.taken == 1


def test_checkpointing_is_on_by_default():
    assert vf.AgentConfig().checkpoint == RolloutCheckpointConfig()
    assert RolloutCheckpointConfig().interval is None
    assert gated("bash", vf.PrimeConfig())._checkpointing() is not None


def test_unsupported_checkpointing_warns_once(caplog, monkeypatch):
    monkeypatch.setattr(agent_module, "_checkpointing_warned", set())
    for _ in range(2):
        assert gated("bash", vf.SubprocessConfig())._checkpointing() is None
        assert gated("browser-use", vf.PrimeConfig())._checkpointing() is None
    warnings = [r.getMessage() for r in caplog.records if r.levelname == "WARNING"]
    assert warnings == [
        (
            "rollout checkpointing is off: the 'subprocess' runtime takes no memory "
            "snapshots; a failed rollout retries from the start"
        ),
        (
            "rollout checkpointing is off: harness 'browser-use' cannot continue from "
            "a memory snapshot; a failed rollout retries from the start"
        ),
    ]


async def test_a_snapshot_holds_every_reply_until_the_box_runs_again():
    run, backend, session = make_rollout(stops=True)
    session.idempotent_requests = {
        "retry:/v1/chat/completions:aa": IdempotentRequest(
            binding=("/v1/chat/completions", b"aa"),
            response=ReplayResponse(200, b"{}", "application/json"),
            completed_at=time.monotonic(),
        ),
        "old": IdempotentRequest(
            binding=("/v1/chat/completions", b"bb"),
            response=ReplayResponse(200, b"{}", "application/json"),
            completed_at=time.monotonic() - 3600,
            sent=True,
        ),
        "open": IdempotentRequest(binding=("/v1/chat/completions", b"cc")),
    }
    capture = asyncio.create_task(run._checkpoint())
    await backend.stopped.wait()
    # The box is stopped: nothing it asked for may be answered.
    assert not session.flowing.is_set()
    assert run._relay.rerouted == []
    backend.running.set()
    checkpoint = await capture
    assert session.flowing.is_set()
    # The stop killed the box's connections: its relay opens new ones.
    assert run._relay.rerouted == ["https://tunnel.example"]
    assert checkpoint.snapshot == vf.Snapshot(id="ck1")
    assert checkpoint.node == len(run.trace.nodes) - 1
    assert checkpoint.session is not None
    assert checkpoint.session["live"] == {
        "program": {"job": {"job_id": "j1"}},
        "relay": run._relay.state,
        "secret": "box-secret",
    }
    # Only a reply the box may not have received yet is kept for its restore.
    assert list(checkpoint.session["replays"]) == ["retry:/v1/chat/completions:aa"]
    assert not checkpoint.pinned


async def test_no_memory_snapshot_without_a_running_segment():
    run, backend, session = make_rollout(running=False)
    assert await run._checkpoint() is None
    assert backend.taken == 0 and session.flowing.is_set()


async def test_a_failed_memory_snapshot_lets_replies_flow_again():
    run, backend, session = make_rollout()

    async def failing():
        raise vf.SandboxError("pause refused")

    backend.snapshot = failing  # type: ignore[method-assign]
    assert await run._checkpoint() is None
    assert session.flowing.is_set() and run.trace.checkpoints == []


async def test_save_returns_a_trace_another_process_can_continue():
    run, backend, session = make_rollout()
    session.pinned_tool_results = {
        ("prefix", "c1"): ToolMessage(tool_call_id="c1", content="pinned")
    }
    run.trace.info = {"nonce": "n"}
    denied = ToolMessage(tool_call_id="c2", content="denied")
    session.tool_decisions = {("prefix", "c1"): None, ("prefix", "c2"): denied}
    session.gated_tools = {("prefix", "c1")}
    session.tool_issuers[("prefix", "c1")] = {0}
    session.tool_dialects[0] = "chat_completions"
    session.prepared_users["user-key"] += 1
    backend.running.set()
    backend.durable.set()
    saved = await run.save()
    assert saved is not None
    (checkpoint,) = run.trace.checkpoints
    assert checkpoint.pinned
    # The running rollout's own trace does not carry the session data.
    assert "session" not in run.trace.model_dump()["checkpoints"][0]

    run.trace.request_rewrites.append(InterceptRecord(handler="after"))
    saved = pickle.loads(pickle.dumps(restorable(run.trace, checkpoint)))
    assert saved["id"] == run.trace.id
    assert restore_entry(saved)["snapshot"] == {"id": "ck1"}
    task = vf.Task(vf.TaskData(idx=0, prompt="q"))
    trace, restored = open_restore(saved, task)
    assert trace.id == run.trace.id and trace.num_turns == run.trace.num_turns
    assert restored is trace.checkpoints[0] and restored.pinned
    assert restored.agent_time == checkpoint.agent_time
    # The owner offers it, so a later retry of the restored run can pick it again.
    assert restored.time.end
    assert restored.session is not None and checkpoint.session is not None
    assert restored.session["live"] == checkpoint.session["live"]
    fresh = RolloutSession(
        route=ModelRoute(ModelContext(model="m", client=EvalClientConfig())),
        trace=trace,
    )
    fresh.restore(restored.session)
    assert fresh.pinned_tool_results == session.pinned_tool_results
    assert fresh.tool_decisions == {("prefix", "c1"): None, ("prefix", "c2"): denied}
    assert fresh.gated_tools == {("prefix", "c1")}
    # A request that names no path reaches the pin's node through its issuer.
    assert fresh.tool_issuers == {("prefix", "c1"): {0}}
    assert fresh.tool_dialects == {0: "chat_completions"}
    assert fresh.prepared_users == {"user-key": 1}
    # The restored trace rewinds to the point like a retry's: the rewrite record
    # made after the save goes, the info comes back.
    trace.info = {}
    assert trace.rewind(restored) == []
    assert trace.info == {"nonce": "n"}
    assert trace.request_rewrites == []


async def test_save_returns_before_the_snapshot_is_durable():
    """The owner gets the restore point once the box runs again; the snapshot becomes
    durable in the background, and the rollout's teardown waits for it."""
    run, backend, _ = make_rollout()
    backend.running.set()
    point = await run.save()
    (checkpoint,) = run.trace.checkpoints
    assert point is not None and not checkpoint.time.end
    backend.durable.set()
    await run._settle_checkpoints(wait=False)
    assert checkpoint.time.end


async def test_a_held_save_answers_the_box_only_after_release():
    run, backend, session = make_rollout()
    backend.running.set()
    assert await run.save(hold=True) is not None
    assert not session.flowing.is_set()
    # A checkpoint of the run's own under the hold does not end it.
    await run._checkpoint()
    assert not session.flowing.is_set()
    run.release()
    assert session.flowing.is_set() and run._release_timer is None


async def test_a_release_does_not_answer_a_box_that_another_snapshot_stops():
    run, backend, session = make_rollout()
    backend.running.set()
    assert await run.save(hold=True) is not None
    # A turn later the rollout takes a snapshot of its own, and the owner releases
    # while the box is stopped for it.
    run.trace.nodes.append(MessageNode(parent=1, message=UserMessage(content="go")))
    backend.running.clear()
    backend.stopped.clear()
    capture = asyncio.create_task(run._checkpoint())
    await backend.stopped.wait()
    run.release()
    assert not session.flowing.is_set()
    backend.running.set()
    assert await capture is not None
    assert session.flowing.is_set()


async def test_a_cancelled_held_save_does_not_hold_the_box():
    run, backend, session = make_rollout()
    backend.running.clear()
    saving = asyncio.create_task(run.save(hold=True))
    await backend.stopped.wait()
    saving.cancel()
    assert (await asyncio.gather(saving, return_exceptions=True))[0] is not None
    assert saving.cancelled()
    assert session.flowing.is_set()
    assert not run._held and run._release_timer is None


async def test_a_snapshot_waits_for_a_turn_that_is_committing():
    """A reply that is on the trace can always be sent again: the snapshot records
    the trace only once the turn has stored its reply for replay."""
    run, backend, session = make_rollout()
    binding = ("/v1/chat/completions", b"digest")
    with session.committing():
        capture = asyncio.create_task(run._checkpoint())
        await asyncio.sleep(0)
        await asyncio.sleep(0)
        assert backend.taken == 0 and not session.flowing.is_set()
        run.trace.nodes.append(MessageNode(parent=1, message=UserMessage(content="go")))
        session.idempotent_requests["k"] = IdempotentRequest(
            binding=binding,
            response=ReplayResponse(200, b"{}", "application/json"),
            completed_at=0.0,
        )
    checkpoint = await capture
    assert checkpoint is not None and checkpoint.node == 2
    # The reply was committed long ago by the clock and never sent: it is kept.
    assert checkpoint.session is not None
    assert checkpoint.session["replays"] == {
        "k": (*binding, 200, b"{}", "application/json")
    }


async def test_snapshots_count_turns_from_the_newest_one():
    run, backend, _ = make_rollout(interval=2)
    assert await run._checkpoint() is not None  # turn 2
    # Two conversations step the count from 2 to 4 before the next request.
    for parent in (1, 2, 3, 4):
        run.trace.nodes.append(
            MessageNode(
                parent=parent, message=AssistantMessage(content="ok"), sampled=True
            )
        )
    assert run.trace.num_turns + 1 == 6
    assert await run._checkpoint() is not None
    assert backend.taken == 2


async def test_a_finished_segment_waits_for_the_owners_release():
    """An owner records a saved rollout as running, so the rollout does not finish
    while the owner still holds it."""
    run, _, _ = make_rollout()
    assert await run.save(hold=True) is not None

    async def turn(messages):
        return None

    run._harness_session = SimpleNamespace(turn=turn)  # type: ignore[assignment]
    segment = asyncio.create_task(run._turn(None))
    await asyncio.sleep(0.01)
    assert not segment.done()
    run.release()
    assert await segment is False


async def test_a_save_whose_program_ended_is_skipped():
    run, _, session = make_rollout()
    states = iter([{"job": {"job_id": "j1"}}, {"job": {"job_id": "j1"}}, None])
    run._harness_session = SimpleNamespace(memory_state=lambda: next(states))  # type: ignore[assignment]
    assert await run.save(hold=True) is None
    assert run.save_skipped == "its harness program ended during the save"
    (checkpoint,) = run.trace.checkpoints
    assert not checkpoint.pinned and session.flowing.is_set() and not run._held


async def test_a_failed_held_save_does_not_hold_the_box():
    run, backend, session = make_rollout()

    async def failing():
        raise vf.SandboxError("pause refused")

    backend.snapshot = failing  # type: ignore[method-assign]
    assert await run.save(hold=True) is None
    assert session.flowing.is_set()
    assert run.save_skipped == "its memory snapshot failed"


async def test_a_rollout_without_memory_snapshots_saves_nothing():
    run, backend, _ = make_rollout()
    run._memory, run.unsaveable = False, "the 'fake' runtime takes no memory snapshots"
    assert await run.save() is None
    assert backend.taken == 0
    assert run.save_skipped == "the 'fake' runtime takes no memory snapshots"


async def test_a_save_between_segments_names_its_reason():
    run, _, _ = make_rollout(running=False)
    assert await run.save() is None
    assert run.save_skipped == "its harness program is not running (setup or scoring)"


def test_unsaveable_names_what_rules_memory_snapshots_out():
    def reason(harness: str, runtime: vf.RuntimeConfig, **config) -> str | None:
        return unsaveable(
            vf.AgentConfig(harness={"id": harness}, runtime=runtime, **config)
        )

    assert reason("bash", vf.PrimeConfig()) is None
    assert reason("rlm", vf.PrimeConfig()) is None
    assert reason("bash", vf.SubprocessConfig()) == (
        "the 'subprocess' runtime takes no memory snapshots"
    )
    assert reason("browser-use", vf.PrimeConfig()) == (
        "harness 'browser-use' cannot continue from a memory snapshot"
    )
    assert reason("bash", vf.PrimeConfig(), checkpoint=None) == (
        "rollout checkpointing is off (agent.checkpoint)"
    )


async def test_an_owners_snapshot_outlives_the_rollout():
    """A pinned snapshot is not under the keep policy, is not deleted with the run's
    own, and becomes durable even when the run settles without a retry to wait for."""
    run, backend, _ = make_rollout()
    backend.running.set()
    saving = asyncio.create_task(run.save())
    await asyncio.sleep(0)
    while not run.trace.checkpoints:
        await asyncio.sleep(0)
    settling = asyncio.create_task(run._settle_checkpoints(wait=False))
    await asyncio.sleep(0)
    assert not settling.done()
    backend.durable.set()
    await settling
    assert await saving is not None
    (pinned,) = run.trace.checkpoints
    # A newer snapshot of the run's own, a turn later, does not prune the owner's.
    run.trace.nodes.append(MessageNode(parent=1, message=UserMessage(content="go")))
    newest = await run._checkpoint()
    await run._settle_checkpoints(wait=True)
    assert backend.deleted == []
    assert kept(run.trace.checkpoints, "last") == {newest.id}
    assert not pinned.deleted


async def test_a_snapshot_records_the_agent_time_spent():
    run, _, _ = make_rollout()
    run._agent_time_spent = 30.0
    run._segment_start = asyncio.get_running_loop().time() - 5.0
    checkpoint = await run._checkpoint()
    assert checkpoint is not None
    # The finished segments and the running one up to the snapshot.
    assert 35.0 <= checkpoint.agent_time < 40.0


async def test_an_attached_program_can_be_snapshotted_again():
    """A box restored from a snapshot runs the program the host attaches to: it is the
    segment's program, so the restored rollout can be saved again."""
    backend = PrimeBackend.__new__(PrimeBackend)
    backend.info = SimpleNamespace(id="sb")  # type: ignore[assignment]
    backend._awake = asyncio.Event()
    backend._awake.set()
    backend._snapshots = 0

    async def attach_process(sandbox_id: str, session: str) -> str:
        return session

    backend._client = SimpleNamespace(attach_process=attach_process)  # type: ignore[assignment]
    program = Program()
    seen = []

    async def collect(process: str) -> None:
        seen.append(program.job)

    backend._collect = collect  # type: ignore[method-assign]
    await backend.attach(None, {"session": "s1"}, program=program)  # type: ignore[arg-type]
    assert seen == [{"session": "s1"}]
    assert program.job is None


async def test_restore_applies_egress_before_connecting_relay(monkeypatch):
    """On restore, prepare_execution must allow the new tunnel host before Relay.connect
    reloads the already-running relay."""
    import verifiers.v1.rollout as rollout_module
    from verifiers.v1.runtimes.backends.prime import PrimeRuntimeInfo
    from verifiers.v1.runtimes.base import RuntimeBackend

    events: list[str] = []

    class FakeBackend(RuntimeBackend):
        SUPPORTS_MEMORY_SNAPSHOTS = True
        network_restricted = True

        def __init__(self):
            super().__init__()
            self.config = vf.PrimeConfig()
            self.info = PrimeRuntimeInfo(id="sb-restored")

        @property
        def is_local(self) -> bool:
            return False

        async def start(self):
            return None

        async def stop(self):
            self.stopped = True

        async def discard_snapshots(self, snapshots):
            return []

        async def prepare_execution(self, routes, network=None):
            events.append(f"prepare_execution:{routes}")

        async def run(self, runtime, argv, env):
            raise AssertionError("unexpected run")

        async def exec(self, runtime, argv, env):
            raise AssertionError("unexpected exec")

        async def run_background(self, runtime, argv, env, log):
            raise AssertionError("unexpected run_background")

        async def read(self, runtime, path, max_bytes=None):
            raise AssertionError("unexpected read")

        async def write(self, runtime, path, data):
            raise AssertionError("unexpected write")

        @classmethod
        def restore_config(cls, config, snapshot):
            return config

    class FakeRelayClient:
        def __init__(self, runtime, name, *, running=None):
            self.url = running["url"]
            self.upstream = ""

        async def installed(self):
            return None

        async def connect(self, upstream, tokens):
            self.upstream = upstream
            events.append(f"relay.connect:{upstream}")
            return self.url

        async def stop(self):
            return None

    @contextlib.asynccontextmanager
    async def fake_serve_interception(*args, **kwargs):
        yield ("https://new-tunnel.example", "new-secret", "state-secret")

    @contextlib.asynccontextmanager
    async def fake_serve_tools(*args, **kwargs):
        yield {}

    async def fake_resolve_agent_runtime(runtime, *args):
        return runtime

    class FakeHarness:
        SUPPORTS_MEMORY_RESTORE = True
        SUPPORTS_TOOL_INTERCEPTION = False
        replay_view = None
        config = SimpleNamespace(id="bash", name="bash")

        async def session(self, *args, **kwargs):
            return SimpleNamespace(memory=False, close=lambda: asyncio.sleep(0))

        async def cleanup(self, *args):
            return None

    monkeypatch.setattr(rollout_module, "backend_cls", lambda cfg: FakeBackend)
    monkeypatch.setattr(rollout_module, "make_backend", lambda cfg, name: FakeBackend())
    monkeypatch.setattr(rollout_module, "Relay", FakeRelayClient)
    monkeypatch.setattr(rollout_module, "serve_interception", fake_serve_interception)
    monkeypatch.setattr(rollout_module, "serve_tools", fake_serve_tools)
    monkeypatch.setattr(
        rollout_module, "resolve_agent_runtime", fake_resolve_agent_runtime
    )

    trace = make_trace()
    ckpt = Checkpoint(
        id="ck1",
        snapshot=vf.Snapshot(id="ck1"),
        node=1,
        time=TimeSpan(start=1.0, end=2.0),
        session={
            **held(trace),
            "pins": [],
            "decisions": [],
            "gated": [],
            "prepared": {},
            "replays": {},
            "live": {
                "program": {"job": {"session": "s1"}},
                "relay": {"dir": "/tmp/vf-relay/t", "url": "http://127.0.0.1:4000"},
                "secret": "old-secret",
            },
        },
    )
    trace.checkpoints.append(ckpt)

    from verifiers.v1.rollout import RolloutTimeouts
    from verifiers.v1.session import RolloutLimits

    run = Rollout(
        task=vf.Task(vf.TaskData(idx=0, prompt="q")),
        harness=FakeHarness(),  # type: ignore[arg-type]
        runtime_config=vf.PrimeConfig(),
        agent_config=vf.AgentConfig(runtime=vf.PrimeConfig()),
        route=ModelRoute(ModelContext(model="m", client=EvalClientConfig())),
        timeouts=RolloutTimeouts(),
        limits=RolloutLimits(),
        restore=(trace, ckpt),
        checkpoint=RolloutCheckpointConfig(),
    )
    try:
        opened = await run.open()
        assert opened, (run.failure, run.trace.errors)
    finally:
        await run.abort()

    assert events == [
        "prepare_execution:['http://127.0.0.1:4000/v1', 'https://new-tunnel.example']",
        "relay.connect:https://new-tunnel.example",
    ]
