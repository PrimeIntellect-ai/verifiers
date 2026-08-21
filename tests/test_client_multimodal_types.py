from copy import deepcopy
from types import SimpleNamespace
from typing import Any, cast

import pytest

from verifiers.clients.openai_chat_completions_client import OpenAIChatCompletionsClient
from verifiers.legacy.clients import (
    openai_chat_completions_client as openai_chat_completions_module,
)
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


class _OpenAIMessage(SimpleNamespace):
    def model_dump(self):
        return self.__dict__


async def _capture_chat_request_body(
    monkeypatch: pytest.MonkeyPatch,
    *,
    base_url: str,
    model: str,
    prompt: Any,
    tools: Any = None,
    sampling_args: Any = None,
) -> dict[str, Any]:
    captured_body: dict[str, Any] = {}
    sentinel = cast(Any, object())

    async def fake_post_chat_completion(
        _client: object,
        path: str,
        *,
        body: dict[str, Any],
        extra_headers: Any = None,
    ) -> Any:
        assert path == "/chat/completions"
        assert extra_headers is None
        captured_body.update(body)
        return sentinel

    monkeypatch.setattr(
        openai_chat_completions_module,
        "post_chat_completion_with_routed_experts_sidecar",
        fake_post_chat_completion,
    )
    client = OpenAIChatCompletionsClient(SimpleNamespace(base_url=base_url))
    response = await client.get_native_response(
        prompt=prompt,
        model=model,
        sampling_args=sampling_args or {},
        tools=tools,
    )
    assert response is sentinel
    return captured_body


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
    "base_url",
    ["https://openrouter.ai/api/v1", "https://api.pinference.ai/api/v1"],
)
async def test_qwen3_max_gateway_marks_only_static_prompt_cache_boundaries(
    monkeypatch: pytest.MonkeyPatch,
    base_url: str,
):
    prompt = cast(
        Any,
        [
            {"role": "system", "content": "first static instruction"},
            {
                "role": "developer",
                "content": [
                    {"type": "text", "text": "end of static instructions"},
                    {
                        "type": "image_url",
                        "image_url": {"url": "https://example.com/static.png"},
                    },
                ],
            },
            {
                "role": "user",
                "content": [{"type": "text", "text": "dynamic rollout turn"}],
            },
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [{"id": "call_1", "type": "function"}],
            },
        ],
    )
    tools = cast(
        Any,
        [
            {
                "type": "function",
                "function": {"name": "first", "parameters": {}},
            },
            {
                "type": "function",
                "function": {
                    "name": "last",
                    "parameters": {"properties": {"cache_control": {"type": "string"}}},
                },
            },
        ],
    )
    original_prompt = deepcopy(prompt)
    original_tools = deepcopy(tools)

    body = await _capture_chat_request_body(
        monkeypatch,
        base_url=base_url,
        model="qwen/qwen3-max",
        prompt=prompt,
        tools=tools,
    )

    assert prompt == original_prompt
    assert tools == original_tools
    assert body["messages"][0] == original_prompt[0]
    assert body["messages"][1]["content"][0] == {
        "type": "text",
        "text": "end of static instructions",
        "cache_control": {"type": "ephemeral"},
    }
    assert body["messages"][1]["content"][1] == original_prompt[1]["content"][1]
    assert body["messages"][2:] == original_prompt[2:]
    assert body["tools"][0] == original_tools[0]
    assert body["tools"][1]["cache_control"] == {"type": "ephemeral"}
    assert set(body["messages"][1]["content"][0]["cache_control"]) == {"type"}
    assert set(body["tools"][1]["cache_control"]) == {"type"}


@pytest.mark.parametrize(
    ("prompt", "tools", "extra_body"),
    [
        ([{"role": "system", "content": "static", "cache_control": {}}], None, {}),
        (
            [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": "dynamic",
                            "prompt_cache_breakpoint": True,
                        }
                    ],
                }
            ],
            None,
            {},
        ),
        ([], [{"type": "function", "cache_control": {}}], {}),
        ([], None, {"prompt_cache_breakpoint": True}),
    ],
)
def test_qwen3_max_prompt_cache_recognizes_caller_intent(
    prompt: Any,
    tools: Any,
    extra_body: dict[str, Any],
):
    assert openai_chat_completions_module._has_caller_prompt_cache_intent(
        prompt, tools, extra_body
    )


@pytest.mark.asyncio
async def test_qwen3_max_top_level_prompt_cache_intent_suppresses_auto_markers(
    monkeypatch: pytest.MonkeyPatch,
):
    prompt = cast(Any, [{"role": "system", "content": "stable instruction"}])
    tools = cast(Any, [{"type": "function", "function": {"name": "lookup"}}])
    caller_marker = {"source": "caller"}

    body = await _capture_chat_request_body(
        monkeypatch,
        base_url="https://openrouter.ai/api/v1",
        model="qwen/qwen3-max",
        prompt=prompt,
        tools=tools,
        sampling_args={"extra_body": {"prompt_cache_breakpoint": caller_marker}},
    )

    assert body["messages"] is prompt
    assert body["tools"] is tools
    assert body["prompt_cache_breakpoint"] is caller_marker


@pytest.mark.parametrize(
    ("model", "base_url"),
    [
        ("qwen/qwen3-max", "https://api.openai.com/v1"),
        ("qwen/qwen3-max-turbo", "https://openrouter.ai/api/v1"),
        ("QWEN/QWEN3-MAX", "https://openrouter.ai/api/v1"),
        ("qwen/qwen3-max", "https://api.openrouter.ai/v1"),
        ("qwen/qwen3-max", "https://foo.pinference.ai/v1"),
        ("qwen/qwen3-max", "https://notopenrouter.ai/v1"),
    ],
)
def test_qwen3_max_prompt_cache_is_exactly_scoped(
    model: str,
    base_url: str,
):
    assert not openai_chat_completions_module._uses_qwen3_max_explicit_prompt_cache(
        model, base_url
    )


@pytest.mark.asyncio
async def test_openai_chat_accepts_refusal_with_reasoning_native_response():
    client = OpenAIChatCompletionsClient(object())
    native_response = SimpleNamespace(
        id="chatcmpl_refusal",
        created=0,
        model="gpt-5.2",
        usage=None,
        choices=[
            SimpleNamespace(
                finish_reason="stop",
                message=_OpenAIMessage(
                    content=None,
                    refusal="I cannot help with that.",
                    reasoning_content="hidden chain",
                    tool_calls=None,
                ),
            )
        ],
    )

    await client.raise_from_native_response(native_response)
    response = await client.from_native_response(native_response)

    assert response.message.content == "I cannot help with that."
    assert response.message.reasoning_content == "hidden chain"


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
async def test_anthropic_to_native_prompt_marks_unsupported_images_in_mixed_content():
    pytest.importorskip("anthropic")
    from verifiers.clients.anthropic_messages_client import AnthropicMessagesClient

    client = AnthropicMessagesClient(object())
    messages = [
        UserMessage(
            content=[
                TextContentPart(text="describe this"),
                ImageUrlContentPart(
                    image_url=ImageUrlSource(url="https://example.com/image.png")
                ),
            ]
        )
    ]

    prompt, kwargs = await client.to_native_prompt(messages)
    assert kwargs["system"] == ""
    assert prompt[0]["content"] == [
        {"type": "text", "text": "describe this"},
        {"type": "text", "text": "[image]"},
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
