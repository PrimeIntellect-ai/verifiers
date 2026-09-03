"""Request-interceptor rewrites survive the harness replaying its own originals."""

import verifiers.v1 as vf
from verifiers.v1 import graph
from verifiers.v1.clients import EvalClientConfig, ModelContext
from verifiers.v1.session import RolloutSession


def _session(*interceptors) -> RolloutSession:
    trace = vf.Trace(
        agent=vf.AgentInfo(config=vf.AgentConfig()),
        task=vf.TraceTask(type="Task", data=vf.TaskData(idx=0, prompt="x")),
    )
    return RolloutSession(
        ctx=ModelContext(model="m", client=EvalClientConfig()),
        trace=trace,
        request_interceptors=list(interceptors),
    )


def _commit(session: RolloutSession, request: vf.Request, text: str) -> None:
    response = vf.Response(
        id="a",
        created=0,
        model="m",
        message=vf.AssistantMessage(content=text),
        finish_reason="stop",
    )
    graph.prepare_turn(session.trace, request.messages).commit(response)


async def test_tool_result_rewrite_is_restored_when_the_harness_replays_its_original():
    seen: list[vf.Message] = []

    def redact_last(request: vf.Request) -> vf.Request | None:
        last = request.messages[-1]
        seen.append(last)
        if isinstance(last, vf.ToolMessage) and last.content == "secret":
            redacted = last.model_copy(update={"content": "[redacted]"})
            return request.model_copy(
                update={"messages": [*request.messages[:-1], redacted]}
            )
        return None

    session = _session(redact_last)
    call = vf.ToolCall(id="c1", type="function", name="bash", arguments="{}")
    turn1 = vf.Request(
        messages=[
            vf.UserMessage(content="run it"),
            vf.AssistantMessage(content=None, tool_calls=[call]),
            vf.ToolMessage(tool_call_id="c1", content="secret"),
        ]
    )
    model1, records1, _ = await session.rewrite_request(turn1)
    assert model1.messages[-1].content == "[redacted]"
    assert [r.handler for r in records1] == ["redact_last"]
    _commit(session, model1, "done")

    # The harness stored `secret`, not `[redacted]`, and replays it with its next turn.
    turn2 = vf.Request(
        messages=[
            *turn1.messages,
            vf.AssistantMessage(content="done"),
            vf.UserMessage(content="next"),
        ]
    )
    model2, records2, _ = await session.rewrite_request(turn2)
    assert model2.messages[2].content == "[redacted]"
    assert records2 == []
    # Only the new message reached the interceptor, and the committed history was reused whole.
    assert seen[-1] == turn2.messages[-1]
    assert graph.prepare_turn(session.trace, model2.messages).tail_start == 4


async def test_repeated_user_originals_map_to_their_rewrites_in_order():
    def shout_last(request: vf.Request) -> vf.Request | None:
        last = request.messages[-1]
        if isinstance(last, vf.UserMessage) and isinstance(last.content, str):
            shouted = last.model_copy(update={"content": last.content.upper()})
            return request.model_copy(
                update={"messages": [*request.messages[:-1], shouted]}
            )
        return None

    session = _session(shout_last)
    turn1 = vf.Request(messages=[vf.UserMessage(content="go")])
    model1, _, _ = await session.rewrite_request(turn1)
    assert model1.messages[0].content == "GO"
    _commit(session, model1, "ok")

    # The replayed original is restored; the identical new message is intercepted afresh.
    turn2 = vf.Request(
        messages=[
            vf.UserMessage(content="go"),
            vf.AssistantMessage(content="ok"),
            vf.UserMessage(content="go"),
        ]
    )
    model2, records2, _ = await session.rewrite_request(turn2)
    assert [m.content for m in model2.messages] == ["GO", "ok", "GO"]
    assert [r.handler for r in records2] == ["shout_last"]
    assert graph.prepare_turn(session.trace, model2.messages).tail_start == 2
