"""Restoring a trace from a checkpoint: what `rewind` restores and what a restored
attempt must not redo."""

import asyncio
from types import SimpleNamespace

import verifiers.v1 as vf
from verifiers.v1.clients import ModelContext
from verifiers.v1.configs.client import EvalClientConfig
from verifiers.v1.graph import MessageNode
from verifiers.v1.rollout import Rollout
from verifiers.v1.session import RolloutSession
from verifiers.v1.trace import Checkpoint, ModelCall, TimeSpan, TraceTask
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
        self.taken = 0

    async def checkpoint(self) -> str:
        self.taken += 1
        return f"ck{self.taken}"

    async def wait_checkpoint(self, checkpoint: str) -> int | None:
        await self.durable.wait()
        if self.fail:
            raise RuntimeError("upload failed")
        return 1024


def make_rollout(interval: int) -> tuple[Rollout, SlowBackend]:
    backend = SlowBackend()
    run = Rollout.__new__(Rollout)
    run.trace = make_trace()  # one turn so far
    run.runtime = SimpleNamespace(backend=backend)  # type: ignore[assignment]

    async def checkpoint_state() -> dict:
        return {}

    run._harness_session = SimpleNamespace(checkpoint_state=checkpoint_state)  # type: ignore[assignment]
    run._session = SimpleNamespace(pinned_tool_results={})  # type: ignore[assignment]
    run._checkpoints = interval
    run._durable = {}
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
