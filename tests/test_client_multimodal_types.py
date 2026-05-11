import pytest
from types import SimpleNamespace
from typing import Any

from verifiers.clients.openai_chat_completions_client import OpenAIChatCompletionsClient
from verifiers.types import (
    AssistantMessage,
    ImageUrlContentPart,
    ImageUrlSource,
    InputAudioContentPart,
    InputAudioSource,
    SystemMessage,
    TextContentPart,
    ToolCall,
    ToolMessage,
    Usage,
    UserMessage,
)
from verifiers.utils.response_utils import parse_response_message


class _RecordingCreate:
    def __init__(self, response: Any) -> None:
        self.response = response
        self.calls: list[dict[str, Any]] = []

    async def create(self, **kwargs: Any) -> Any:
        self.calls.append(kwargs)
        return self.response


def _recording_openai(response: Any) -> tuple[Any, _RecordingCreate]:
    recorder = _RecordingCreate(response)
    return SimpleNamespace(chat=SimpleNamespace(completions=recorder)), recorder


def _recording_anthropic(response: Any) -> tuple[Any, _RecordingCreate]:
    recorder = _RecordingCreate(response)
    return SimpleNamespace(messages=recorder), recorder


@pytest.mark.asyncio
async def test_openai_to_native_prompt_with_typed_multimodal_content_parts():
    client = OpenAIChatCompletionsClient(object())
    messages = [
        UserMessage(
            content=[
                TextContentPart(text="describe this"),
                ImageUrlContentPart(
                    image_url=ImageUrlSource(url="data:image/png;base64,abc123")
                ),
                InputAudioContentPart(
                    input_audio=InputAudioSource(data="ZHVtbXk=", format="wav")
                ),
            ]
        )
    ]

    prompt, kwargs = await client.to_native_prompt(messages)
    assert kwargs == {}
    assert len(prompt) == 1
    assert prompt[0]["role"] == "user"
    assert prompt[0]["content"] == [
        {"type": "text", "text": "describe this"},
        {
            "type": "image_url",
            "image_url": {"url": "data:image/png;base64,abc123"},
        },
        {
            "type": "input_audio",
            "input_audio": {"data": "ZHVtbXk=", "format": "wav"},
        },
    ]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("model", "effort"),
    [
        ("anthropic/claude-opus-4.7", "xhigh"),
        ("anthropic/claude-sonnet-4.6", "max"),
    ],
)
async def test_openrouter_anthropic_reasoning_effort_maps_to_verbosity(
    model: str, effort: str
):
    recording_client, recorder = _recording_openai(SimpleNamespace())
    client = OpenAIChatCompletionsClient(recording_client)
    client._config = SimpleNamespace(api_base_url="https://openrouter.ai/api/v1")

    response = await client.get_native_response(
        prompt=[],
        model=model,
        sampling_args={
            "n": 1,
            "reasoning_effort": effort,
            "extra_body": {"reasoning": {"enabled": True}},
        },
    )

    assert response is recorder.response
    call = recorder.calls[0]
    assert "reasoning_effort" not in call
    assert call["extra_body"] == {
        "reasoning": {"enabled": True},
        "verbosity": effort,
    }


@pytest.mark.asyncio
async def test_openrouter_anthropic_reasoning_effort_enables_reasoning():
    recording_client, recorder = _recording_openai(SimpleNamespace())
    client = OpenAIChatCompletionsClient(recording_client)
    client._config = SimpleNamespace(api_base_url="https://api.pinference.ai/api/v1")

    await client.get_native_response(
        prompt=[],
        model="anthropic/claude-opus-4.7",
        sampling_args={"reasoning_effort": "high"},
    )

    call = recorder.calls[0]
    assert call["extra_body"] == {
        "reasoning": {"enabled": True},
        "verbosity": "high",
    }


@pytest.mark.asyncio
async def test_anthropic_to_native_prompt_with_typed_multimodal_content_parts():
    pytest.importorskip("anthropic")
    from verifiers.clients.anthropic_messages_client import AnthropicMessagesClient

    client = AnthropicMessagesClient(object())
    messages = [
        SystemMessage(
            content=[
                TextContentPart(text="You are a helpful assistant."),
                ImageUrlContentPart(
                    image_url=ImageUrlSource(url="data:image/png;base64,sys")
                ),
            ]
        ),
        UserMessage(
            content=[
                TextContentPart(text="what is in this?"),
                ImageUrlContentPart(
                    image_url=ImageUrlSource(url="data:image/png;base64,abc123")
                ),
                InputAudioContentPart(
                    input_audio=InputAudioSource(data="ZHVtbXk=", format="wav")
                ),
            ]
        ),
    ]

    prompt, kwargs = await client.to_native_prompt(messages)
    assert kwargs["system"] == "You are a helpful assistant. [image]"
    assert len(prompt) == 1
    assert prompt[0]["role"] == "user"
    assert prompt[0]["content"] == [
        {"type": "text", "text": "what is in this?"},
        {
            "type": "image",
            "source": {
                "type": "base64",
                "media_type": "image/png",
                "data": "abc123",
            },
        },
        {"type": "text", "text": "[audio]"},
    ]


@pytest.mark.asyncio
async def test_anthropic_assistant_tool_calls_use_text_chunks_not_model_repr():
    pytest.importorskip("anthropic")
    from verifiers.clients.anthropic_messages_client import AnthropicMessagesClient

    client = AnthropicMessagesClient(object())
    messages = [
        AssistantMessage(
            content=[TextContentPart(text="calling a tool")],
            tool_calls=[ToolCall(id="call_1", name="lookup", arguments='{"q":"x"}')],
        )
    ]

    prompt, kwargs = await client.to_native_prompt(messages)
    assert kwargs["system"] == ""
    assert len(prompt) == 1
    assert prompt[0]["role"] == "assistant"
    assert prompt[0]["content"] == [
        {"type": "text", "text": "calling a tool"},
        {"type": "tool_use", "id": "call_1", "name": "lookup", "input": {"q": "x"}},
    ]


@pytest.mark.asyncio
async def test_anthropic_merges_consecutive_tool_results_into_single_user_message():
    pytest.importorskip("anthropic")
    from verifiers.clients.anthropic_messages_client import AnthropicMessagesClient

    client = AnthropicMessagesClient(object())
    messages = [
        AssistantMessage(
            content="calling tools",
            tool_calls=[
                ToolCall(id="call_1", name="lookup_a", arguments='{"q":"a"}'),
                ToolCall(id="call_2", name="lookup_b", arguments='{"q":"b"}'),
            ],
        ),
        ToolMessage(tool_call_id="call_1", content="result a"),
        ToolMessage(tool_call_id="call_2", content="result b"),
    ]

    prompt, kwargs = await client.to_native_prompt(messages)

    assert kwargs["system"] == ""
    assert len(prompt) == 2
    assert prompt[0]["role"] == "assistant"
    assert prompt[1]["role"] == "user"
    assert prompt[1]["content"] == [
        {"type": "tool_result", "tool_use_id": "call_1", "content": "result a"},
        {"type": "tool_result", "tool_use_id": "call_2", "content": "result b"},
    ]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("model", "effort"),
    [("claude-opus-4-7", "xhigh"), ("claude-sonnet-4-6", "max")],
)
async def test_anthropic_reasoning_effort_maps_to_output_config(
    model: str, effort: str
):
    pytest.importorskip("anthropic")
    from verifiers.clients.anthropic_messages_client import AnthropicMessagesClient

    recording_client, recorder = _recording_anthropic(SimpleNamespace())
    client = AnthropicMessagesClient(recording_client)

    response = await client.get_native_response(
        prompt=[],
        model=model,
        sampling_args={"max_tokens": 128, "reasoning_effort": effort},
    )

    assert response is recorder.response
    call = recorder.calls[0]
    assert "reasoning_effort" not in call
    assert call["output_config"] == {"effort": effort}
    assert call["thinking"] == {"type": "adaptive"}


@pytest.mark.asyncio
async def test_anthropic_reasoning_effort_preserves_existing_output_config():
    pytest.importorskip("anthropic")
    from verifiers.clients.anthropic_messages_client import AnthropicMessagesClient

    recording_client, recorder = _recording_anthropic(SimpleNamespace())
    client = AnthropicMessagesClient(recording_client)

    await client.get_native_response(
        prompt=[],
        model="claude-opus-4-7",
        sampling_args={
            "max_tokens": 128,
            "reasoning_effort": "high",
            "output_config": {"format": {"type": "text"}},
        },
    )

    call = recorder.calls[0]
    assert call["output_config"] == {
        "format": {"type": "text"},
        "effort": "high",
    }


@pytest.mark.asyncio
async def test_anthropic_from_native_response_extracts_usage():
    anthropic = pytest.importorskip("anthropic")
    from anthropic.types import Message as AnthropicMessage

    from verifiers.clients.anthropic_messages_client import AnthropicMessagesClient

    client = AnthropicMessagesClient(object())

    native_response = AnthropicMessage(
        id="msg_test123",
        type="message",
        role="assistant",
        content=[{"type": "text", "text": "Hello!"}],
        model="claude-haiku-4-5",
        stop_reason="end_turn",
        stop_sequence=None,
        usage=anthropic.types.Usage(input_tokens=42, output_tokens=17),
    )

    response = await client.from_native_response(native_response)

    assert response.usage is not None
    assert isinstance(response.usage, Usage)
    assert response.usage.prompt_tokens == 42
    assert response.usage.completion_tokens == 17
    assert response.usage.total_tokens == 59
    assert response.usage.reasoning_tokens == 0


@pytest.mark.asyncio
async def test_anthropic_from_native_response_always_parses_reasoning():
    pytest.importorskip("anthropic")
    from verifiers.clients.anthropic_messages_client import AnthropicMessagesClient

    client = AnthropicMessagesClient(object())
    native_response = SimpleNamespace(
        id="msg_think",
        model="claude-haiku-4-5",
        stop_reason="end_turn",
        usage=SimpleNamespace(input_tokens=1, output_tokens=1),
        content=[
            SimpleNamespace(type="thinking", thinking="hidden chain"),
            SimpleNamespace(type="text", text="final answer"),
        ],
    )

    response = await client.from_native_response(native_response)
    assert response.message.reasoning_content == "hidden chain"
    assert response.message.content == "final answer"


@pytest.mark.asyncio
async def test_anthropic_tool_call_round_trips_thinking_blocks():
    pytest.importorskip("anthropic")
    from anthropic.types import Message as AnthropicMessage
    from anthropic.types import Usage as AnthropicUsage

    from verifiers.clients.anthropic_messages_client import AnthropicMessagesClient

    client = AnthropicMessagesClient(object())
    native_response = AnthropicMessage(
        id="msg_tool_think",
        type="message",
        role="assistant",
        content=[
            {"type": "thinking", "thinking": "hidden chain", "signature": "sig_1"},
            {"type": "tool_use", "id": "call_1", "name": "lookup", "input": {"q": "x"}},
        ],
        model="claude-haiku-4-5",
        stop_reason="tool_use",
        stop_sequence=None,
        usage=AnthropicUsage(input_tokens=1, output_tokens=1),
    )

    response = await client.from_native_response(native_response)
    completion_messages = await parse_response_message(response)
    prompt, kwargs = await client.to_native_prompt(completion_messages)

    assert kwargs["system"] == ""
    assert len(prompt) == 1
    assert prompt[0]["role"] == "assistant"
    assert prompt[0]["content"] == [
        {"type": "thinking", "thinking": "hidden chain", "signature": "sig_1"},
        {"type": "tool_use", "id": "call_1", "name": "lookup", "input": {"q": "x"}},
    ]
