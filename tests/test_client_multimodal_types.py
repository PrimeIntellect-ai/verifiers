import base64

import pytest
import numpy as np
from types import SimpleNamespace

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


class _CaptureAnthropicMessages:
    def __init__(self) -> None:
        self.last_kwargs: dict | None = None

    async def create(self, **kwargs):
        self.last_kwargs = kwargs
        return SimpleNamespace()


class _CaptureAnthropicClient:
    def __init__(self) -> None:
        self.messages = _CaptureAnthropicMessages()


@pytest.mark.asyncio
async def test_anthropic_get_native_response_forwards_router_replay_with_extra_body():
    pytest.importorskip("anthropic")
    from verifiers.clients.anthropic_messages_client import AnthropicMessagesClient

    native_client = _CaptureAnthropicClient()
    client = AnthropicMessagesClient(native_client)

    await client.get_native_response(
        prompt=[{"role": "user", "content": "hello"}],
        model="claude-test",
        sampling_args={
            "max_tokens": 32,
            "temperature": 0.2,
            "extra_body": {"seed": 7},
            "routed_experts": [[[1]]],
        },
    )

    sent = native_client.messages.last_kwargs
    assert sent is not None
    assert sent["temperature"] == 0.2
    assert sent["extra_body"] == {"seed": 7, "routed_experts": [[[1]]]}
    assert "routed_experts" not in sent


@pytest.mark.asyncio
async def test_anthropic_from_native_response_extracts_tokens_and_router_replay():
    pytest.importorskip("anthropic")
    from verifiers.clients.anthropic_messages_client import AnthropicMessagesClient

    client = AnthropicMessagesClient(object())
    routed = np.array([[[11, 12]], [[21, 22]]], dtype=np.int32)
    native_response = SimpleNamespace(
        id="msg_tokens",
        model="claude-haiku-4-5",
        stop_reason="end_turn",
        usage=SimpleNamespace(input_tokens=3, output_tokens=2),
        content=[SimpleNamespace(type="text", text="ok")],
        prompt_token_ids=[1, 2, 3],
        token_ids=[4, 5],
        logprobs={"content": [{"logprob": -0.1}, {"logprob": -0.2}]},
        routed_experts={
            "data": base64.b85encode(routed.tobytes()).decode("utf-8"),
            "shape": list(routed.shape),
        },
    )

    response = await client.from_native_response(native_response)

    assert response.message.tokens is not None
    assert response.message.tokens.prompt_ids == [1, 2, 3]
    assert response.message.tokens.completion_ids == [4, 5]
    assert response.message.tokens.completion_logprobs == [-0.1, -0.2]
    assert response.message.tokens.routed_experts == routed.tolist()


@pytest.mark.asyncio
async def test_anthropic_from_native_response_requires_logprobs_for_tokens():
    pytest.importorskip("anthropic")
    from verifiers.clients.anthropic_messages_client import AnthropicMessagesClient

    client = AnthropicMessagesClient(object())
    native_response = SimpleNamespace(
        id="msg_tokens_missing",
        model="claude-haiku-4-5",
        stop_reason="end_turn",
        usage=SimpleNamespace(input_tokens=2, output_tokens=1),
        content=[SimpleNamespace(type="text", text="ok")],
        prompt_token_ids=[1, 2],
        token_ids=[3],
        logprobs=None,
    )

    response = await client.from_native_response(native_response)
    assert response.message.tokens is None
