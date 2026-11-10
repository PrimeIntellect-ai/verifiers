"""Rollout checkpointing: when a rollout takes checkpoints, what `rewind` restores, and
what a restored attempt must not redo."""

import asyncio
import pickle
import time
from types import SimpleNamespace
from typing import Literal

import verifiers.v1 as vf
import verifiers.v1.agent as agent_module
from verifiers.v1.clients import ModelContext
from verifiers.v1.configs.agent import RolloutCheckpointConfig
from verifiers.v1.configs.client import EvalClientConfig
from verifiers.v1.configs.retries import RetryConfig
from verifiers.v1.graph import MessageNode
from verifiers.v1.restore import RestorePoint, unsaveable
from verifiers.v1.rollout import Rollout, kept
from verifiers.v1.session import IdempotentRequest, ReplayResponse, RolloutSession
from verifiers.v1.trace import (
    Checkpoint,
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
from verifiers.v1.utils.retries import RetryState


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


def test_rewind_restores_info_with_state():
    trace = make_trace()
    checkpoint = Checkpoint(
        id="ck", snapshot=vf.Snapshot(id="ck"), node=1, info={"setup": "nonce"}
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
        ctx=ModelContext(model="m", client=EvalClientConfig()), trace=run.trace
    )
    run._session = session
    run._harness_session = SimpleNamespace(  # type: ignore[assignment]
        memory_state=lambda: None if job is None else {"job": job}
    )
    run._relay = FakeRelay()  # type: ignore[assignment]
    run._secret = "box-secret"
    run._checkpointing = RolloutCheckpointConfig(interval=interval, keep=keep)
    run._retryable = lambda error: isinstance(error, vf.ProviderError)
    run.rescued = False
    run._durable = {}
    run._captures = set()
    run._closed = False
    run._opened = True
    run._memory = True
    run.unsaveable = None
    run.save_skipped = None
    run._held = False
    run._release_timer = None
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


async def test_a_failed_model_call_that_the_run_retries_takes_a_snapshot():
    run, backend, session = make_rollout()
    run._checkpointing = RolloutCheckpointConfig()
    # The run would not retry this error: the box is not stopped for it.
    await run._rescue(vf.HarnessError("exit 1"))
    assert backend.taken == 0
    await run._rescue(vf.ProviderError("overloaded", status_code=529))
    (checkpoint,) = run.trace.checkpoints
    assert backend.taken == 1 and checkpoint.node == len(run.trace.nodes) - 1
    assert run.rescued
    assert session.flowing.is_set()
    # The same call fails again: the snapshot already holds the box that waits for it.
    await run._rescue(vf.ProviderError("overloaded", status_code=529))
    assert backend.taken == 1
    # A turn later, a new failure needs a new snapshot.
    run.trace.nodes.append(MessageNode(parent=1, message=UserMessage(content="go")))
    await run._rescue(vf.ProviderError("overloaded", status_code=529))
    assert backend.taken == 2


async def test_a_run_without_a_retry_left_takes_no_snapshot():
    run, backend, _ = make_rollout()
    run._retryable = None
    await run._rescue(vf.ProviderError("overloaded", status_code=529))
    assert backend.taken == 0


def test_retry_state_tells_whether_an_error_gets_a_retry_without_spending_it():
    state = RetryState(
        RetryConfig(max_retries=0, rules=[{"type": "ProviderError", "max_retries": 1}])
    )
    error = vf.ProviderError("overloaded", status_code=529)
    assert state.allows(error) and state.allows(error)
    assert not state.allows(vf.HarnessError("exit 1"))
    trace = make_trace()
    trace.record_error(error)
    assert state.next_error(trace.errors) is not None
    assert not state.allows(error)


def test_checkpointing_is_on_by_default():
    assert vf.AgentConfig().checkpoint == RolloutCheckpointConfig()
    assert RolloutCheckpointConfig().interval is None
    assert gated("bash", vf.PrimeConfig())._checkpointing() is not None


def test_unsupported_checkpointing_warns_once(caplog, monkeypatch):
    monkeypatch.setattr(agent_module, "_checkpointing_warned", set())
    for _ in range(2):
        assert gated("bash", vf.SubprocessConfig())._checkpointing() is None
        assert gated("terminus-2", vf.PrimeConfig())._checkpointing() is None
    warnings = [r.getMessage() for r in caplog.records if r.levelname == "WARNING"]
    assert warnings == [
        (
            "rollout checkpointing is off: the 'subprocess' runtime takes no memory "
            "snapshots; a failed rollout retries from the start"
        ),
        (
            "rollout checkpointing is off: harness 'terminus-2' cannot continue from "
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
    assert checkpoint.live == {
        "program": {"job": {"job_id": "j1"}},
        "relay": run._relay.state,
        "secret": "box-secret",
    }
    # Only a reply the box may not have received yet is kept for its restore.
    assert list(checkpoint.replays) == ["retry:/v1/chat/completions:aa"]
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


async def test_save_returns_a_restore_point_another_process_can_open():
    run, backend, session = make_rollout()
    session.pinned_tool_results = {
        ("prefix", "c1"): ToolMessage(tool_call_id="c1", content="pinned")
    }
    run.trace.info = {"nonce": "n"}
    backend.running.set()
    backend.durable.set()
    point = await run.save()
    assert isinstance(point, RestorePoint)
    (checkpoint,) = run.trace.checkpoints
    assert checkpoint.pinned

    point = pickle.loads(pickle.dumps(point))
    assert point.id == run.trace.id
    assert point.snapshot == {"id": "ck1"}
    task = vf.Task(vf.TaskData(idx=0, prompt="q"))
    trace, restored = point.open(task)
    assert trace.id == run.trace.id and trace.num_turns == run.trace.num_turns
    assert restored is trace.checkpoints[0] and restored.pinned
    assert restored.info == {"nonce": "n"}
    assert restored.pins == session.pinned_tool_results
    assert restored.live == checkpoint.live
    # The restored trace rewinds to the point like a retry's.
    assert trace.rewind(restored) == []


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
    assert reason("terminus-2", vf.PrimeConfig()) == (
        "harness 'terminus-2' cannot continue from a memory snapshot"
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
    # A newer snapshot of the run's own does not prune the owner's.
    newest = await run._checkpoint()
    await run._settle_checkpoints(wait=True)
    assert backend.deleted == []
    assert kept(run.trace.checkpoints, "last") == {newest.id}
    assert not pinned.deleted
