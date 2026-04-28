"""GptOssRenderer parity vs ``openai-harmony``'s reference encoder.

Why this file exists
--------------------
HF's ``apply_chat_template`` for gpt-oss diverges from the model's
canonical wire format in small ways (trailing ``\\n\\n`` after developer
instructions; different layout for function-tool declarations). The
canonical format is what ``openai-harmony`` produces — vLLM uses that
encoder directly when serving gpt-oss, and it's what the model was
trained on.

So our ``GptOssRenderer`` matches harmony, not HF Jinja, and this file
checks that. ``test_render_ids`` skips gpt-oss for the same reason.

Each test mirrors the corresponding case in ``test_render_ids`` but
uses ``HarmonyEncoding.render_conversation_for_training`` as the
oracle.
"""

from __future__ import annotations

from datetime import datetime

import pytest
from openai_harmony import (
    Conversation,
    DeveloperContent,
    HarmonyEncodingName,
    Message as HarmonyMessage,
    ReasoningEffort,
    Role,
    SystemContent,
    ToolDescription,
    load_harmony_encoding,
)
from renderers.gpt_oss import GptOssRenderer
from transformers import AutoTokenizer

GPT_OSS_MODEL = "openai/gpt-oss-20b"
DATE_FOR_PARITY = datetime.now().strftime("%Y-%m-%d")


@pytest.fixture(scope="module")
def tokenizer():
    return AutoTokenizer.from_pretrained(GPT_OSS_MODEL)


@pytest.fixture(scope="module")
def renderer(tokenizer):
    # Pin the date so the rendered preamble matches the harmony oracle
    # built with the same fixed date.
    return GptOssRenderer(tokenizer, conversation_start_date=DATE_FOR_PARITY)


@pytest.fixture(scope="module")
def encoder():
    return load_harmony_encoding(HarmonyEncodingName.HARMONY_GPT_OSS)


def _system_content() -> SystemContent:
    return (
        SystemContent.new()
        .with_reasoning_effort(ReasoningEffort.MEDIUM)
        .with_conversation_start_date(DATE_FOR_PARITY)
    )


def _tool_description() -> ToolDescription:
    return ToolDescription.new(
        name="get_weather",
        description="Get the current weather for a city",
        parameters={
            "type": "object",
            "properties": {
                "city": {"type": "string", "description": "The city name"},
            },
            "required": ["city"],
        },
    )


TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get the current weather for a city",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {"type": "string", "description": "The city name"},
                },
                "required": ["city"],
            },
        },
    }
]


def test_no_system_message(renderer, encoder):
    """A user-only conversation: only the auto-injected system preamble + user."""
    msgs = [{"role": "user", "content": "Hello!"}]
    got = renderer.render_ids(msgs, add_generation_prompt=False)

    conv = Conversation.from_messages([
        HarmonyMessage.from_role_and_content(Role.SYSTEM, _system_content()),
        HarmonyMessage.from_role_and_content(Role.USER, "Hello!"),
    ])
    expected = encoder.render_conversation_for_training(conv)
    assert got == expected


def test_system_and_user(renderer, encoder):
    """The caller's first system message becomes the developer Instructions block."""
    msgs = [
        {"role": "system", "content": "You are helpful."},
        {"role": "user", "content": "Hello!"},
    ]
    got = renderer.render_ids(msgs, add_generation_prompt=False)

    conv = Conversation.from_messages([
        HarmonyMessage.from_role_and_content(Role.SYSTEM, _system_content()),
        HarmonyMessage.from_role_and_content(
            Role.DEVELOPER,
            DeveloperContent.new().with_instructions("You are helpful."),
        ),
        HarmonyMessage.from_role_and_content(Role.USER, "Hello!"),
    ])
    expected = encoder.render_conversation_for_training(conv)
    assert got == expected


def test_terminal_assistant_uses_return(renderer, encoder):
    """Terminal assistant turn (no follow-up) closes with ``<|return|>``."""
    msgs = [
        {"role": "user", "content": "Hi"},
        {"role": "assistant", "content": "Hello!"},
    ]
    got = renderer.render_ids(msgs, add_generation_prompt=False)

    conv = Conversation.from_messages([
        HarmonyMessage.from_role_and_content(Role.SYSTEM, _system_content()),
        HarmonyMessage.from_role_and_content(Role.USER, "Hi"),
        HarmonyMessage.from_role_and_content(Role.ASSISTANT, "Hello!").with_channel(
            "final"
        ),
    ])
    expected = encoder.render_conversation_for_training(conv)
    assert got == expected


def test_tools_with_system(renderer, encoder):
    """Tools attach to the developer message alongside the instructions."""
    msgs = [
        {"role": "system", "content": "You are a weather assistant."},
        {"role": "user", "content": "Weather?"},
    ]
    got = renderer.render_ids(msgs, tools=TOOLS, add_generation_prompt=False)

    dev = (
        DeveloperContent.new()
        .with_instructions("You are a weather assistant.")
        .with_function_tools([_tool_description()])
    )
    conv = Conversation.from_messages([
        HarmonyMessage.from_role_and_content(Role.SYSTEM, _system_content()),
        HarmonyMessage.from_role_and_content(Role.DEVELOPER, dev),
        HarmonyMessage.from_role_and_content(Role.USER, "Weather?"),
    ])
    expected = encoder.render_conversation_for_training(conv)
    assert got == expected


def test_tool_call_and_response(renderer, encoder):
    """Tool call → tool result → final assistant: each in its own harmony message."""
    msgs = [
        {"role": "user", "content": "Weather in Paris?"},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "id": "c1",
                    "function": {
                        "name": "get_weather",
                        "arguments": '{"city": "Paris"}',
                    },
                }
            ],
        },
        {
            "role": "tool",
            "content": '{"temp": 20}',
            "tool_call_id": "c1",
            "name": "get_weather",
        },
        {"role": "assistant", "content": "It's 20 degrees in Paris."},
    ]
    got = renderer.render_ids(msgs, tools=TOOLS, add_generation_prompt=False)

    dev = DeveloperContent.new().with_function_tools([_tool_description()])
    conv = Conversation.from_messages([
        HarmonyMessage.from_role_and_content(Role.SYSTEM, _system_content()),
        HarmonyMessage.from_role_and_content(Role.DEVELOPER, dev),
        HarmonyMessage.from_role_and_content(Role.USER, "Weather in Paris?"),
        HarmonyMessage.from_role_and_content(
            Role.ASSISTANT, '{"city": "Paris"}'
        )
        .with_channel("commentary")
        .with_recipient("functions.get_weather"),
        HarmonyMessage.from_author_and_content(
            {"role": "tool", "name": "functions.get_weather"}, '{"temp": 20}'
        )
        .with_recipient("assistant")
        .with_channel("commentary"),
        HarmonyMessage.from_role_and_content(
            Role.ASSISTANT, "It's 20 degrees in Paris."
        ).with_channel("final"),
    ])
    expected = encoder.render_conversation_for_training(conv)
    assert got == expected


def test_assistant_with_reasoning_content_strips_analysis(renderer, encoder):
    """Historical reasoning_content is stripped — harmony drops
    analysis-channel messages from rendered history. The renderer matches
    that behaviour, emitting only the final-channel text."""
    msgs = [
        {"role": "user", "content": "2+2?"},
        {
            "role": "assistant",
            "content": "Four.",
            "reasoning_content": "Two plus two is four.",
        },
    ]
    got = renderer.render_ids(msgs, add_generation_prompt=False)

    # Oracle has no analysis message — it would be stripped anyway.
    conv = Conversation.from_messages([
        HarmonyMessage.from_role_and_content(Role.SYSTEM, _system_content()),
        HarmonyMessage.from_role_and_content(Role.USER, "2+2?"),
        HarmonyMessage.from_role_and_content(Role.ASSISTANT, "Four.").with_channel(
            "final"
        ),
    ])
    expected = encoder.render_conversation_for_training(conv)
    assert got == expected


def test_generation_prompt(renderer):
    """``add_generation_prompt=True`` appends the analysis-channel start."""
    msgs = [{"role": "user", "content": "Hi"}]
    got = renderer.render_ids(msgs, add_generation_prompt=True)
    no_genprompt = renderer.render_ids(msgs, add_generation_prompt=False)
    # The trailing tokens should be: <|start|>assistant<|channel|>analysis<|message|>
    suffix = got[len(no_genprompt) :]
    decoded = renderer._tokenizer.decode(suffix)  # noqa: SLF001
    assert decoded == "<|start|>assistant<|channel|>analysis<|message|>", decoded
