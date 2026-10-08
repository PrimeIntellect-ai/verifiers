"""The rlm harness resumes a saved conversation: a Messages prompt with model or tool messages is rlm's seed."""

from types import SimpleNamespace

import verifiers.v1 as vf
from verifiers.v1.harnesses.rlm.harness import (
    RLM_RUNTIME_METADATA_KEY,
    RLMHarness,
    RLMHarnessConfig,
)
from verifiers.v1.types import (
    AssistantMessage,
    SystemMessage,
    ToolCall,
    ToolMessage,
    UserMessage,
)


async def test_a_saved_conversation_becomes_the_seed_and_its_last_user_messages_the_turn():
    saved = [
        SystemMessage(content="the old system prompt"),
        UserMessage(content="build it"),
        AssistantMessage(
            content="", tool_calls=[ToolCall(id="c1", name="ipython", arguments="{}")]
        ),
        ToolMessage(tool_call_id="c1", content="done"),
        UserMessage(content="your box was replaced"),
    ]
    harness = RLMHarness(RLMHarnessConfig(id="rlm"))
    config = await harness.prepare_acp(
        SimpleNamespace(model="m"),
        SimpleNamespace(id="trace"),
        SimpleNamespace(env={}),
        "http://endpoint",
        "secret",
        {},
        vf.TaskData(prompt=saved, system_prompt="be brief"),
    )
    payload = config.session_meta[RLM_RUNTIME_METADATA_KEY]
    assert [m["role"] for m in payload["seed_messages"]] == [
        "user",
        "assistant",
        "tool",
    ]
    assert payload["seed_messages"][1]["tool_calls"][0]["id"] == "c1"
    assert config.prompt == [UserMessage(content="your box was replaced")]
    assert payload["append_to_system_prompt"] == "be brief"
