"""A continuation opens on a well-formed transcript: no reply cut off at the token
limit, and one host note however often the run was cut off before the model replied."""

import verifiers.v1 as vf
from verifiers.v1.graph import MessageNode
from verifiers.v1.trace import ModelCall, TimeSpan, TraceTask
from verifiers.v1.types import AssistantMessage, ToolCall, ToolMessage, UserMessage
from verifiers.v1.utils.retries import KILLED_NOTE, RETRY_NOTE, without_unanswered_notes


class Data(vf.TaskData):
    prompt: str


def _trace(*messages, cut_off_last: bool = False) -> vf.Trace:
    trace = vf.Trace(
        agent=vf.AgentInfo(config=vf.AgentConfig()),
        task=TraceTask(type="Data", data=Data(idx=0, prompt="q")),
    )
    for index, message in enumerate(messages):
        sampled = message.role == "assistant"
        trace.nodes.append(
            MessageNode(
                parent=index - 1 if index else None, message=message, sampled=sampled
            )
        )
        if sampled:
            last = index == len(messages) - 1
            trace.calls.append(
                ModelCall(
                    node=index,
                    finish_reason="length" if cut_off_last and last else "stop",
                    time=TimeSpan(start=1.0, end=2.0),
                )
            )
    return trace


def _work() -> list:
    call = ToolCall(id="c1", name="bash", arguments="{}")
    return [
        UserMessage(content="build it"),
        AssistantMessage(tool_calls=[call]),
        ToolMessage(tool_call_id="c1", content="ok"),
    ]


def test_a_reply_cut_off_at_the_token_limit_is_not_continued_from() -> None:
    cut = AssistantMessage(
        content="<tool_call>bash<arg_key>command</arg_key><arg_value>mk"
    )
    trace = _trace(*_work(), cut, cut_off_last=True)
    assert trace.without_cut_off_reply(trace.conversation_at()) == _work()


def test_a_finished_reply_stays() -> None:
    done = AssistantMessage(content="ready")
    trace = _trace(*_work(), done)
    assert trace.without_cut_off_reply(trace.conversation_at())[-1] == done


def test_host_notes_no_reply_answered_give_way_to_the_next() -> None:
    stacked = [
        *_work(),
        UserMessage(content=RETRY_NOTE),
        UserMessage(content=KILLED_NOTE),
    ]
    assert without_unanswered_notes(stacked) == _work()
    answered = [
        *_work(),
        UserMessage(content=RETRY_NOTE),
        AssistantMessage(content="ok"),
    ]
    assert without_unanswered_notes(answered) == answered
