"""An ACP turn the agent reports as complete must have finished: Pi and OpenClaw report
`end_turn` when a model request fails, so the trace has to show it."""

import pytest

import verifiers.v1 as vf
from verifiers.v1.acp import _require_model_turn
from verifiers.v1.graph import MessageNode
from verifiers.v1.runtimes import ProgramResult
from verifiers.v1.types import AssistantMessage, UserMessage

TOOL_CALL = vf.ToolCall(id="call_0", name="bash", arguments='{"command": "ls"}')


def trace(*replies: AssistantMessage) -> vf.Trace:
    nodes = [MessageNode(parent=None, message=UserMessage(content="q"), sampled=False)]
    calls = []
    for reply in replies:
        nodes.append(MessageNode(parent=len(nodes) - 1, message=reply, sampled=True))
        calls.append(vf.ModelCall(node=len(nodes) - 1))
    return vf.Trace(
        agent=vf.AgentInfo(config=vf.AgentConfig()),
        task=vf.TraceTask(type="Task", data=vf.TaskData(idx=0, prompt="q")),
        nodes=nodes,
        calls=calls,
    )


def done(stdout: str = "DONE") -> ProgramResult:
    return ProgramResult(exit_code=0, stdout=stdout, stderr="")


def test_a_final_reply_completes_the_turn():
    tr = trace(
        AssistantMessage(tool_calls=[TOOL_CALL]), AssistantMessage(content="done")
    )
    _require_model_turn(tr, 0, done())


def test_a_turn_ending_on_unanswered_tool_calls_fails_with_the_agent_stderr():
    tr = trace(AssistantMessage(tool_calls=[TOOL_CALL]))
    with pytest.raises(
        RuntimeError, match="before the model saw its last tool results.*404"
    ):
        _require_model_turn(tr, 0, done(""), stderr="Provider request failed: 404")


def test_a_turn_without_a_model_turn_fails():
    with pytest.raises(RuntimeError, match="without committing a model turn"):
        _require_model_turn(trace(), 0, done())


def test_only_this_turns_calls_count():
    tr = trace(AssistantMessage(content="earlier answer"))
    tr.calls.append(vf.ModelCall())  # a failed call commits no node
    with pytest.raises(RuntimeError, match="without committing a model turn"):
        _require_model_turn(tr, 1, done())


def test_a_stop_condition_ends_the_turn_on_purpose():
    tr = trace(AssistantMessage(tool_calls=[TOOL_CALL]))
    tr.stop("max_turns")
    _require_model_turn(tr, 0, done())
