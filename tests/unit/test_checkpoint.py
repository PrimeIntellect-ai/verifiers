"""Rollout checkpointing: when a rollout takes a checkpoint (on a failure, of its
filesystem or memory), what `rewind` restores, and what a restored attempt must not
redo."""

import asyncio
from types import SimpleNamespace

import verifiers.v1 as vf
import verifiers.v1.agent as agent_module
from verifiers.v1.clients import ModelContext
from verifiers.v1.configs.agent import (
    MemoryCheckpointConfig,
    RolloutCheckpointConfig,
    TurnRetryConfig,
)
from verifiers.v1.configs.client import EvalClientConfig
from verifiers.v1.graph import MessageNode
from verifiers.v1.interception.server import retryable, turn_retry_delay
from verifiers.v1.rollout import Rollout
from verifiers.v1.runtimes import Snapshot
from verifiers.v1.session import RolloutSession
from verifiers.v1.trace import (
    Checkpoint,
    ModelCall,
    RestorePoint,
    TimeSpan,
    TraceTask,
)
from verifiers.v1.types import (
    AssistantMessage,
    Request,
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


def test_rewind_restores_info_with_state():
    trace = make_trace()
    checkpoint = Checkpoint(id="ck", node=1, info={"setup": "nonce"})
    trace.checkpoints.append(checkpoint)
    trace.info = {"setup": "nonce", "scoring_skipped": "stale"}
    trace.rewind(checkpoint)
    assert trace.info == {"setup": "nonce"}


def test_rewind_keeps_the_dropped_turns_calls():
    trace = make_trace()
    checkpoint = Checkpoint(id="ck", node=0)
    trace.checkpoints.append(checkpoint)
    trace.rewind(checkpoint)
    assert len(trace.nodes) == 1
    assert [call.node for call in trace.calls] == [None]


def test_restored_span_counts_only_its_own_calls():
    trace = make_trace()
    trace.timing.agent.start, trace.timing.agent.end = 100.0, 120.0
    trace.calls.append(ModelCall(time=TimeSpan(start=105.0, end=110.0)))
    trace.split_agent_time()
    assert trace.timing.agent.model.duration == 5.0
    assert trace.timing.agent.harness.duration == 15.0


async def test_restored_tail_is_not_intercepted_twice():
    trace = make_trace()
    seen: list[str] = []

    def tag(request: Request) -> Request:
        for message in request.messages:
            if isinstance(message, ToolMessage | UserMessage):
                seen.append(str(message.content))
                if isinstance(message, ToolMessage):
                    message.content = f"{message.content} [tagged]"
        return request

    session = RolloutSession(
        ctx=ModelContext(model="m", client=EvalClientConfig()),
        trace=trace,
        request_interceptors=[tag],
    )
    tail = [
        ToolMessage(tool_call_id="c1", content="out [tagged]"),
        UserMessage(content="go on"),
    ]
    session.mark_intercepted(trace.messages, tail)
    request = Request(messages=[*trace.messages, *tail])
    rewritten, records, _ = await session.rewrite_request(request)
    assert rewritten.messages == request.messages
    assert records == []
    assert seen == []


class SlowBackend:
    """Captures at once; becomes durable when the test says so."""

    def __init__(self) -> None:
        self.durable = asyncio.Event()
        self.fail = False
        self.pause_fails = False
        self.taken: list[bool] = []
        self.deleted: list[str] = []

    async def checkpoint(self, *, memory: bool = False) -> Snapshot:
        if memory and self.pause_fails:
            raise RuntimeError("pause failed")
        self.taken.append(memory)
        return Snapshot(
            id=f"ck{len(self.taken)}", memory=memory, box="box" if memory else None
        )

    async def wait_checkpoint(self, snapshot: Snapshot) -> int | None:
        await self.durable.wait()
        if self.fail:
            raise RuntimeError("upload failed")
        return 1024

    async def discard_checkpoints(self, snapshots: list[Snapshot]) -> list[str]:
        self.deleted += [snapshot.id for snapshot in snapshots]
        return [snapshot.id for snapshot in snapshots]


def make_rollout(*, memory: bool = False) -> tuple[Rollout, SlowBackend]:
    backend = SlowBackend()
    run = Rollout.__new__(Rollout)
    run.trace = make_trace()  # one turn so far
    run.runtime = SimpleNamespace(backend=backend)  # type: ignore[assignment]

    async def checkpoint_state() -> dict:
        return {"acp_session_id": "s1"}

    async def memory_state() -> dict:
        return {"program": {"job_id": "j1"}}

    run._harness_session = SimpleNamespace(  # type: ignore[assignment]
        checkpoint_state=checkpoint_state, memory_state=memory_state, frozen=False
    )
    run._session = RolloutSession(  # type: ignore[assignment]
        ctx=ModelContext(model="m", client=EvalClientConfig()), trace=run.trace
    )
    run._checkpointing = RolloutCheckpointConfig()
    run._memory = MemoryCheckpointConfig() if memory else None
    run._memory_checkpoints = memory
    run._relay = {"url": "http://127.0.0.1:1", "tokens": {"model": "m", "state": "s"}}
    run._checkpointed = None
    run._paused = False
    run._durable = {}
    run._captures = set()
    run._closed = False
    run.deleted_checkpoints = set()
    run.rewound_checkpoints = []
    run._boxless = False
    run._session.checkpointer = run._checkpoint
    return run, backend


TURN = SimpleNamespace(prefix_node_ids=[0, 1], tail=[])
ERROR = vf.ProviderError("upstream 503", status_code=503)


async def test_checkpoint_becomes_durable_in_the_background():
    run, backend = make_rollout()
    checkpoint = await run._checkpoint(TURN, ERROR)
    assert checkpoint is not None and checkpoint.captured
    assert checkpoint.reason == "ProviderError" and not checkpoint.memory
    assert checkpoint.harness == {"acp_session_id": "s1"}
    assert checkpoint.time.end == 0 and checkpoint.bytes is None
    backend.durable.set()
    await run._settle_checkpoints(wait=True)
    assert checkpoint.time.end and checkpoint.bytes == 1024
    assert run.trace.checkpoints == [checkpoint]


async def test_an_attempt_takes_one_checkpoint():
    run, backend = make_rollout()
    first = await run._checkpoint(TURN, ERROR)
    assert await run._checkpoint(TURN, ERROR) is first
    assert backend.taken == [False]


async def test_a_filesystem_checkpoint_needs_a_turn():
    run, backend = make_rollout()
    assert await run._checkpoint(None, ERROR) is None
    assert backend.taken == []


async def test_a_memory_checkpoint_freezes_the_box():
    run, backend = make_rollout(memory=True)
    checkpoint = await run._checkpoint(None, ERROR)
    assert checkpoint is not None and checkpoint.memory
    assert checkpoint.live == {
        "harness": {"program": {"job_id": "j1"}},
        "relay": run._relay,
    }
    # Without a turn, the box holds every committed node.
    assert checkpoint.node == len(run.trace.nodes) - 1
    assert run._paused and run._harness_session.frozen
    assert backend.taken == [True]


async def test_a_failed_pause_falls_back_to_the_filesystem():
    run, backend = make_rollout(memory=True)
    backend.pause_fails = True
    checkpoint = await run._checkpoint(TURN, ERROR)
    assert checkpoint is not None and not checkpoint.memory and checkpoint.box
    assert not run._paused and not run._harness_session.frozen
    assert backend.taken == [False]


async def test_a_checkpoint_that_never_becomes_durable_is_dropped():
    run, backend = make_rollout()
    await run._checkpoint(TURN, ERROR)
    backend.fail = True
    backend.durable.set()
    await run._settle_checkpoints(wait=True)
    assert run.trace.checkpoints == []


async def test_settling_without_a_retry_stops_waiting():
    run, _ = make_rollout()
    checkpoint = await run._checkpoint(TURN, ERROR)
    await asyncio.wait_for(run._settle_checkpoints(wait=False), timeout=1)
    assert run._durable == {}
    assert run.trace.checkpoints == [checkpoint] and checkpoint.time.end == 0


async def test_a_closing_run_takes_no_checkpoint():
    run, backend = make_rollout()
    capture = asyncio.create_task(run._checkpoint(TURN, ERROR))
    run._closed = True
    await run._settle_checkpoints(wait=True)
    assert await capture is None
    assert backend.taken == [] and run.trace.checkpoints == []


async def test_a_boxless_checkpoint_is_restorable_at_once():
    run, backend = make_rollout()
    run._boxless = True
    checkpoint = await run._checkpoint(TURN, ERROR)
    assert checkpoint is not None and not checkpoint.box
    assert checkpoint.time.end and backend.taken == []
    assert run.trace.restorable is checkpoint


async def test_interrupt_checkpoints_the_turn_in_flight():
    run, backend = make_rollout()
    run.trace._pending = {"k": TURN}
    await run.interrupt(timeout=1)
    assert isinstance(run._session.aborted, vf.RolloutInterrupted)
    assert run._checkpointed is not None
    assert run._checkpointed.reason == "RolloutInterrupted"
    assert backend.taken == [False]


async def test_interrupt_waits_for_the_next_turn_for_a_filesystem_checkpoint():
    run, backend = make_rollout()
    run.trace._pending = {}
    interrupt = asyncio.create_task(run.interrupt(timeout=5))
    await asyncio.sleep(0)
    error = run._session.interrupt_at_turn
    assert isinstance(error, vf.RolloutInterrupted)
    # The next model call checkpoints and ends the segment (the interception).
    await run._session.checkpointer(TURN, error)
    run._session.abort(error)
    await asyncio.wait_for(interrupt, timeout=1)
    assert backend.taken == [False]


async def test_interrupt_without_a_turn_gives_up_after_its_timeout():
    run, backend = make_rollout()
    run.trace._pending = {}
    await run.interrupt(timeout=0.01)
    assert isinstance(run._session.aborted, vf.RolloutInterrupted)
    assert backend.taken == []


async def test_save_checkpoints_at_the_next_turn_and_goes_on():
    run, backend = make_rollout()
    save = asyncio.create_task(run.save(timeout=5))
    await asyncio.sleep(0)
    # The next model call takes the save (the interception), and goes upstream.
    await run._session.take_save(TURN)
    assert await asyncio.wait_for(save, timeout=1)
    assert run._session.save_at_turn is None and run._session.aborted is None
    (checkpoint,) = run.trace.checkpoints
    assert checkpoint.reason == "Saved" and not checkpoint.memory
    # A save is not the attempt's checkpoint: a failure still takes its own.
    assert run._checkpointed is None
    assert await run._checkpoint(TURN, ERROR) is not None
    assert backend.taken == [False, False]


async def test_save_takes_a_filesystem_checkpoint_of_a_memory_run():
    run, backend = make_rollout(memory=True)
    save = asyncio.create_task(run.save(timeout=5))
    await asyncio.sleep(0)
    await run._session.take_save(TURN)
    assert await save
    assert backend.taken == [False] and not run._paused


async def test_save_without_a_turn_gives_up_after_its_timeout():
    run, backend = make_rollout()
    assert not await run.save(timeout=0.01)
    assert run._session.save_at_turn is None and backend.taken == []


async def test_save_skips_a_finishing_rollout():
    run, _ = make_rollout()
    run.trace.stop("max_turns")
    assert not await run.save(timeout=1)
    run, _ = make_rollout()
    run._checkpointing = None
    assert not await run.save(timeout=1)


def test_turn_retries_back_off_and_run_out():
    config = TurnRetryConfig(max_retries=3, initial_delay=1.0, max_delay=3.0)
    delays = [turn_retry_delay(config, ERROR, n) for n in range(4)]
    assert delays == [1.0, 2.0, 3.0, None]
    assert retryable(config, vf.ProviderError("x", status_code=429))
    # A deterministic error goes to the harness at once.
    assert turn_retry_delay(config, vf.ProviderError("x", status_code=400), 0) is None


def test_a_restore_point_round_trips():
    checkpoint = Checkpoint(
        id="ck",
        node=1,
        tail=[ToolMessage(tool_call_id="c1", content="out")],
        state=vf.State(),
        info={"nonce": "n"},
        pins={("prefix", "c1"): ToolMessage(tool_call_id="c1", content="pinned")},
        time=TimeSpan(start=1.0, end=2.0),
    )
    wire = RestorePoint.of(checkpoint).model_dump(mode="json")
    back = RestorePoint.model_validate(wire).live(vf.State)
    assert back.tail == checkpoint.tail
    assert back.info == {"nonce": "n"}
    assert back.pins == checkpoint.pins
    assert back.state == checkpoint.state


def gated(harness: str, runtime: vf.RuntimeConfig, **checkpoint) -> vf.Agent:
    return vf.Agent(
        vf.AgentConfig(
            model="test-model",
            harness={"id": harness},
            runtime=runtime,
            checkpoint=RolloutCheckpointConfig(**checkpoint),
            memory_checkpoint=MemoryCheckpointConfig(),
        )
    )


def test_filesystem_checkpoints_are_on_by_default():
    assert vf.AgentConfig().checkpoint == RolloutCheckpointConfig()
    assert vf.AgentConfig().memory_checkpoint is None
    checkpoint, memory = gated("bash", vf.PrimeConfig())._checkpointing()
    assert checkpoint is not None and memory is not None


def test_unsupported_checkpointing_warns_once(caplog, monkeypatch):
    monkeypatch.setattr(agent_module, "_checkpointing_warned", set())
    for _ in range(2):
        assert gated("bash", vf.SubprocessConfig())._checkpointing() == (None, None)
        assert gated("terminus-2", vf.PrimeConfig())._checkpointing() == (None, None)
    warnings = [r.getMessage() for r in caplog.records if r.levelname == "WARNING"]
    assert warnings == [
        (
            "filesystem checkpoints are off: the 'subprocess' runtime takes no "
            "snapshots; a failed rollout retries from the start"
        ),
        (
            "memory checkpoints are off: the 'subprocess' runtime takes no memory "
            "checkpoints; a failed rollout retries from the start"
        ),
        (
            "filesystem checkpoints are off: harness 'terminus-2' does not support it; "
            "a failed rollout retries from the start"
        ),
        (
            "memory checkpoints are off: harness 'terminus-2' does not support it; "
            "a failed rollout retries from the start"
        ),
    ]


def test_boxless_checkpointing_is_opt_in():
    assert gated("null", vf.SubprocessConfig())._checkpointing()[0] is None
    assert gated("null", vf.SubprocessConfig(), boxless=True)._checkpointing()[0]
    assert (
        gated("bash", vf.SubprocessConfig(), boxless=True)._checkpointing()[0] is None
    )


async def test_interrupt_all_interrupts_the_open_rollouts(monkeypatch):
    import verifiers.v1.rollout as rollout_module

    running, _ = make_rollout()
    running.trace._pending = {"k": TURN}
    closed, _ = make_rollout()
    closed._closed = True
    monkeypatch.setattr(rollout_module, "_LIVE", {running, closed})
    assert await rollout_module.interrupt_all(timeout=1) == (1, [running.trace.id])
    assert isinstance(running._session.aborted, vf.RolloutInterrupted)
    assert closed._session.aborted is None


async def test_interrupt_leaves_a_stopped_rollout_to_finish():
    run, backend = make_rollout()
    run.trace._pending = {"k": TURN}
    run.trace.stop("max_turns")
    await run.interrupt(timeout=1)
    assert run._session.aborted is None and backend.taken == []
