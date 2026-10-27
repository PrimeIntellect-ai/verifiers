"""Resuming a trace from a checkpoint: what `rewind` restores and what a resumed
attempt must not redo."""

import verifiers.v1 as vf
from verifiers.v1.clients import ModelContext
from verifiers.v1.configs.client import EvalClientConfig
from verifiers.v1.graph import MessageNode
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


def test_resumed_span_counts_only_its_own_calls():
    trace = make_trace()
    trace.timing.agent.start, trace.timing.agent.end = 100.0, 120.0
    trace.calls.append(ModelCall(time=TimeSpan(start=105.0, end=110.0)))
    trace.split_agent_time()
    assert trace.timing.agent.model.duration == 5.0
    assert trace.timing.agent.harness.duration == 15.0


async def test_resumed_tail_is_not_intercepted_twice():
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
