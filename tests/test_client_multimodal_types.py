import pytest

from verifiers.clients.openai.openai_clients import OAIChatCompletionsClient
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
    UserMessage,
)


@pytest.mark.asyncio
async def test_openai_to_native_prompt_with_typed_multimodal_content_parts():
    client = OAIChatCompletionsClient(object())
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
    from verifiers.clients.anthropic.anthropic_clients import AnthropicMessagesClient

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
    from verifiers.clients.anthropic.anthropic_clients import AnthropicMessagesClient

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
    from verifiers.clients.anthropic.anthropic_clients import AnthropicMessagesClient

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
