"""`Harness.resume` relaunches a chat program on the trace's conversation. No model or
sandbox: the relaunch prompt is captured."""

from types import SimpleNamespace

from verifiers.v1.configs.harness import HarnessConfig
from verifiers.v1.harness import Harness
from verifiers.v1.task import TaskData
from verifiers.v1.types import (
    AssistantMessage,
    SystemMessage,
    ToolCall,
    ToolMessage,
    UserMessage,
)

# --- resuming a conversation --------------------------------------------------------


class ResumableHarness(Harness[HarnessConfig]):
    SUPPORTS_RESUME = True

    async def launch(self, ctx, trace, runtime, endpoint, secret, mcp_urls, data, **_):
        self.prompt = data.prompt


async def resumed(branch: list, data: TaskData | None = None) -> list:
    harness = ResumableHarness(HarnessConfig(id="resumable"))
    trace = SimpleNamespace(branches=[SimpleNamespace(messages=branch)])
    new = [UserMessage(content="continue")]
    await harness.resume(None, trace, None, "", "", {}, data or TaskData(), new)
    return harness.prompt


def call(call_id: str) -> ToolCall:
    return ToolCall(id=call_id, name="bash", arguments='{"command": "sleep 99"}')


async def test_resume_drops_tool_calls_a_cut_off_segment_left_unanswered():
    opened = [UserMessage(content="fix it"), AssistantMessage(content="looking")]
    answered = [
        AssistantMessage(tool_calls=[call("a")]),
        ToolMessage(tool_call_id="a", content="done"),
    ]
    cut_off = AssistantMessage(content="next", tool_calls=[call("b"), call("c")])
    prompt = await resumed([*opened, *answered, cut_off])
    assert prompt == [*opened, *answered, UserMessage(content="continue")]


async def test_resume_keeps_a_conversation_that_ends_answered():
    branch = [
        SystemMessage(content="rules"),
        UserMessage(content="fix it"),
        AssistantMessage(content="fixed"),
    ]
    prompt = await resumed(branch)
    assert prompt == [*branch, UserMessage(content="continue")]
    # A system prompt the task re-emits is not carried twice.
    prompt = await resumed(branch, TaskData(system_prompt="rules"))
    assert prompt == [*branch[1:], UserMessage(content="continue")]
