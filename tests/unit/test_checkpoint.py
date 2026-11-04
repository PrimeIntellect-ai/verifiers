"""Rollout checkpointing: when a rollout takes checkpoints, what `rewind` restores, and
what a restored attempt must not redo."""

import asyncio
from types import SimpleNamespace
from typing import Literal

import verifiers.v1 as vf
import verifiers.v1.agent as agent_module
from verifiers.v1.clients import ModelContext
from verifiers.v1.configs.agent import RolloutCheckpointConfig
from verifiers.v1.configs.client import EvalClientConfig
from verifiers.v1.graph import MessageNode
from verifiers.v1.rollout import Rollout, kept
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


def test_agent_time_splits_off_the_checkpoint_hold():
    trace = make_trace()
    trace.timing.agent.start, trace.timing.agent.end = 100.0, 120.0
    trace.calls.append(ModelCall(time=TimeSpan(start=105.0, end=110.0)))
    trace.checkpoints += [
        Checkpoint(id="old", captured=90.0, held=3.0),
        Checkpoint(id="new", captured=106.0, held=1.5),
    ]
    trace.split_agent_time()
    assert trace.timing.agent.checkpoint.duration == 1.5
    assert trace.timing.agent.model.duration == 3.5
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
        self.taken = 0
        self.deleted: list[str] = []

    async def checkpoint(self) -> str:
        self.taken += 1
        return f"ck{self.taken}"

    async def wait_checkpoint(self, checkpoint: str) -> int | None:
        await self.durable.wait()
        if self.fail:
            raise RuntimeError("upload failed")
        return 1024

    async def discard_checkpoints(self, ids: list[str]) -> list[str]:
        self.deleted += ids
        return ids


def make_rollout(
    interval: int = 1, keep: Literal["all", "last"] | int = "last"
) -> tuple[Rollout, SlowBackend]:
    backend = SlowBackend()
    run = Rollout.__new__(Rollout)
    run.trace = make_trace()  # one turn so far
    run.runtime = SimpleNamespace(backend=backend)  # type: ignore[assignment]

    async def checkpoint_state() -> dict:
        return {}

    run._harness_session = SimpleNamespace(checkpoint_state=checkpoint_state)  # type: ignore[assignment]
    run._session = SimpleNamespace(pinned_tool_results={})  # type: ignore[assignment]
    run._checkpointing = RolloutCheckpointConfig(interval=interval, keep=keep)
    run._durable = {}
    run._captures = set()
    run._closed = False
    run._pruning = asyncio.Lock()
    run.deleted_checkpoints = set()
    run.rewound_checkpoints = []
    run._boxless = False
    return run, backend


TURN = SimpleNamespace(prefix_node_ids=[0, 1], tail=[])


async def test_checkpoint_interval_skips_turns():
    run, backend = make_rollout(interval=2)
    assert await run._checkpoint(TURN) is None  # turn 2: 1 % 2
    run.trace.nodes.append(MessageNode(parent=1, message=UserMessage(content="go")))
    run.trace.nodes.append(
        MessageNode(parent=2, message=AssistantMessage(content="ok"), sampled=True)
    )
    assert await run._checkpoint(TURN) is not None  # turn 3: 2 % 2
    assert backend.taken == 1


async def test_checkpoint_becomes_durable_in_the_background():
    run, backend = make_rollout(interval=1)
    checkpoint = await run._checkpoint(TURN)
    assert checkpoint is not None and checkpoint.captured
    assert checkpoint.time.end == 0 and checkpoint.bytes is None
    backend.durable.set()
    await run._settle_checkpoints(wait=True)
    assert checkpoint.time.end and checkpoint.bytes == 1024
    assert run.trace.checkpoints == [checkpoint]


async def test_a_checkpoint_that_never_becomes_durable_is_dropped():
    run, backend = make_rollout(interval=1)
    await run._checkpoint(TURN)
    backend.fail = True
    backend.durable.set()
    await run._settle_checkpoints(wait=True)
    assert run.trace.checkpoints == []


async def test_settling_without_a_retry_stops_waiting():
    run, _ = make_rollout(interval=1)
    checkpoint = await run._checkpoint(TURN)
    await asyncio.wait_for(run._settle_checkpoints(wait=False), timeout=1)
    assert run._durable == {}
    assert run.trace.checkpoints == [checkpoint] and checkpoint.time.end == 0


async def test_a_closing_run_takes_no_checkpoint():
    run, backend = make_rollout(interval=1)
    capture = asyncio.create_task(run._checkpoint(TURN))
    run._closed = True
    await run._settle_checkpoints(wait=True)
    assert await capture is None
    assert backend.taken == 0 and run.trace.checkpoints == []


def durable(*ids: str, deleted: tuple[str, ...] = ()) -> list[Checkpoint]:
    return [
        Checkpoint(id=i, time=TimeSpan(start=1.0, end=2.0), deleted=i in deleted)
        for i in ids
    ]


def test_keep_policies():
    checkpoints = durable("a", "b", "c", "d", "e")
    assert kept(checkpoints, "last") == {"e"}
    assert kept(checkpoints, "all") == {"a", "b", "c", "d", "e"}
    assert kept(checkpoints, 2) == {"a", "c", "e"}
    assert kept(checkpoints, 3) == {"a", "d", "e"}
    # Deleted and not yet durable snapshots are never kept.
    pending = [*durable("a", "b", deleted=("a",)), Checkpoint(id="c")]
    assert kept(pending, "all") == {"b"}


async def test_a_durable_snapshot_prunes_the_older_ones():
    run, backend = make_rollout(keep="last")
    run.trace.checkpoints = durable("old1", "old2")
    run.rewound_checkpoints = ["rewound"]
    backend.durable.set()
    newest = await run._checkpoint(TURN)
    await run._settle_checkpoints(wait=True)
    assert sorted(backend.deleted) == ["old1", "old2", "rewound"]
    assert [c.id for c in run.trace.checkpoints if not c.deleted] == [newest.id]
    assert run.deleted_checkpoints == {"old1", "old2", "rewound"}


async def test_a_boxless_checkpoint_is_restorable_at_once():
    run, backend = make_rollout()
    run._boxless = True
    checkpoint = await run._checkpoint(TURN)
    assert checkpoint is not None and not checkpoint.box
    assert checkpoint.time.end and backend.taken == 0
    assert run.trace.restorable is checkpoint


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
        )
    )


def test_checkpointing_is_on_by_default():
    assert vf.AgentConfig().checkpoint == RolloutCheckpointConfig()
    assert gated("bash", vf.PrimeConfig())._checkpointing() is not None


def test_unsupported_checkpointing_warns_once(caplog, monkeypatch):
    monkeypatch.setattr(agent_module, "_checkpointing_warned", set())
    for _ in range(2):
        assert gated("bash", vf.SubprocessConfig())._checkpointing() is None
        assert gated("terminus-2", vf.PrimeConfig())._checkpointing() is None
    warnings = [r.getMessage() for r in caplog.records if r.levelname == "WARNING"]
    assert warnings == [
        (
            "rollout checkpointing is off: the 'subprocess' runtime takes no "
            "snapshots; a failed rollout retries from the start"
        ),
        (
            "rollout checkpointing is off: harness 'terminus-2' does not support it; "
            "a failed rollout retries from the start"
        ),
    ]


def test_boxless_checkpointing_is_opt_in():
    assert gated("null", vf.SubprocessConfig())._checkpointing() is None
    assert gated("null", vf.SubprocessConfig(), boxless=True)._checkpointing()
    assert gated("bash", vf.SubprocessConfig(), boxless=True)._checkpointing() is None
