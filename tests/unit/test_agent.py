"""`Interaction` segments: a failed segment reports its own partial reply, and an
expired agent deadline stops the rollout as a timeout."""

import asyncio
from types import SimpleNamespace

import pytest

import verifiers.v1 as vf
from verifiers.v1 import graph
from verifiers.v1.agent import Interaction
from verifiers.v1.clients import EvalClientConfig, ModelContext, ModelRoute
from verifiers.v1.graph import MessageNode
from verifiers.v1.harnesses.utils.replay import drop_blank_text_blocks
from verifiers.v1.rollout import Rollout, RolloutTimeouts
from verifiers.v1.session import RolloutSession
from verifiers.v1.types import AssistantMessage


class FailingSegmentRollout:
    ok = Rollout.ok
    closed = Rollout.closed
    fail = Rollout.fail
    step = Rollout.step
    timeout = Rollout.timeout


def make_trace() -> vf.Trace:
    return vf.Trace(
        agent=vf.AgentInfo(config=vf.AgentConfig()),
        task=vf.TraceTask(
            type="Task",
            data=vf.TaskData(idx=0, prompt=None),
            key="dataset/example-0",
            hash="content-digest",
        ),
    )


async def test_each_sample_keeps_its_tool_result_across_assistant_replays():
    replacements = iter(["First result", "Second result"])

    def replace_result(request: vf.Request) -> vf.Request | None:
        tool = request.messages[-1]
        if isinstance(tool, vf.ToolMessage) and tool.content == "":
            tool.content = next(replacements)
            return request
        return None

    trace = make_trace()
    trace._replay_view = lambda message, dialect: (
        drop_blank_text_blocks(message) if dialect == "anthropic_messages" else message
    )
    sampled = AssistantMessage(
        content="Ready\n",
        text_block_offsets=[5],
        tool_calls=[vf.ToolCall(id="call", name="lookup", arguments="{}")],
    )
    session = RolloutSession(
        ModelRoute(ModelContext(model="test", client=EvalClientConfig())),
        trace,
        request_interceptors=[replace_result],
        gates_tools=True,
    )
    replay = sampled.model_copy(update={"content": "Ready", "text_block_offsets": None})
    prompt = [vf.UserMessage(content="go")]

    def commit(messages, assistant):
        turn = graph.prepare_turn(trace, messages, dialect="anthropic_messages")
        return turn.commit(
            vf.Response(
                id="", created=0, model="test", message=assistant, finish_reason="stop"
            )
        )

    # Two samples from one prompt replay identically and reuse a call ID.
    for assistant, expected in [(sampled, "First result"), (replay, "Second result")]:
        node = commit(prompt, assistant)
        await session.gate_tool_calls(node, "anthropic_messages")
        assert (await session.decide_tool("call"))["action"] == "deny"
        request = vf.Request(
            messages=[
                *prompt,
                replay,
                vf.ToolMessage(tool_call_id="call", content="Tool unavailable"),
            ]
        )
        rewritten, _, _ = await session.rewrite_request(
            request, dialect="anthropic_messages"
        )
        assert rewritten.messages[-1].content == expected
        commit(rewritten.messages, AssistantMessage(content="done"))


def make_rollout(trace: vf.Trace, session, time_remaining=None):
    run = FailingSegmentRollout()
    run.trace = trace
    run._opened = True
    run._closed = False
    run._failed = False
    run._failure = None
    run._borrowed_runtime = None
    run.runtime = None
    run._agent_time_remaining = time_remaining
    run._timeouts = RolloutTimeouts()
    run._harness_session = session
    run.deployment = SimpleNamespace(run=lambda turn: turn)
    run._session = SimpleNamespace(
        request_interceptors=[],
        error=None,
        stopped=False,
    )
    run.deadline_at = None
    return run


@pytest.mark.asyncio
async def test_failed_segment_does_not_reuse_prior_root_reply():
    trace = make_trace()
    trace.root_reply = "previous reply"

    class FailingSession:
        async def turn(self, messages):
            trace.nodes.append(
                MessageNode(
                    parent=None,
                    message=AssistantMessage(content="current partial reply"),
                    sampled=True,
                )
            )
            raise RuntimeError("segment failed after sampling")

    run = make_rollout(trace, FailingSession())

    segment = await Interaction(run).turn("next")

    assert segment.last_reply == "current partial reply"
    assert trace.root_reply is None
    assert trace.last_reply == "current partial reply"


@pytest.mark.asyncio
async def test_expired_deadline_is_a_timeout_when_the_runtime_reports_an_error():
    """A runtime can turn the deadline's cancellation into its own error, as Prime's SDK
    does for a cancelled request. The rollout still stops on its agent timeout."""
    trace = make_trace()

    class CancelReportingSession:
        async def turn(self, messages):
            try:
                await asyncio.sleep(60)
            except asyncio.CancelledError:
                raise RuntimeError("request was cancelled") from None

    run = make_rollout(trace, CancelReportingSession(), time_remaining=0.05)

    await Interaction(run).turn("next")

    assert trace.is_timeout and not trace.errors
    assert trace.stop_condition == "agent_timeout"
