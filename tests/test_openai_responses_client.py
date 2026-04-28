from types import SimpleNamespace
from typing import Any

import pytest

from verifiers.clients import resolve_client
from verifiers.clients.openai_responses_client import (
    OPENAI_RESPONSES_OUTPUT_FIELD,
    OpenAIResponsesClient,
)
from verifiers.types import (
    AssistantMessage,
    ClientConfig,
    Response,
    ResponseMessage,
    TextContentPart,
    Tool,
    ToolCall,
    ToolMessage,
    Usage,
    UserMessage,
)
from verifiers.utils.response_utils import parse_response_message


class _RecordingResponses:
    def __init__(self, response: Any) -> None:
        self.response = response
        self.calls: list[dict[str, Any]] = []

    async def create(self, **kwargs: Any) -> Any:
        self.calls.append(kwargs)
        return self.response


class _RecordingOpenAI:
    def __init__(self, response: Any) -> None:
        self.responses = _RecordingResponses(response)

    async def close(self) -> None:
        pass


@pytest.mark.asyncio
async def test_to_native_prompt_converts_messages_and_tool_outputs():
    client = OpenAIResponsesClient(object())
    messages = [
        UserMessage(
            content=[
                TextContentPart(text="describe this"),
                {
                    "type": "image_url",
                    "image_url": {"url": "data:image/png;base64,abc123"},
                },
            ]
        ),
        AssistantMessage(
            content=None,
            tool_calls=[ToolCall(id="call_1", name="lookup", arguments='{"q":"x"}')],
        ),
        ToolMessage(tool_call_id="call_1", content="result x"),
    ]

    prompt, kwargs = await client.to_native_prompt(messages)

    assert kwargs == {}
    assert prompt == [
        {
            "type": "message",
            "role": "user",
            "content": [
                {"type": "input_text", "text": "describe this"},
                {
                    "type": "input_image",
                    "image_url": "data:image/png;base64,abc123",
                    "detail": "auto",
                },
            ],
        },
        {
            "type": "function_call",
            "call_id": "call_1",
            "name": "lookup",
            "arguments": '{"q":"x"}',
        },
        {
            "type": "function_call_output",
            "call_id": "call_1",
            "output": "result x",
        },
    ]


@pytest.mark.asyncio
async def test_get_native_response_normalizes_sampling_args_and_tools():
    native_response = SimpleNamespace(output=[{"type": "message", "content": []}])
    recording_client = _RecordingOpenAI(native_response)
    client = OpenAIResponsesClient(recording_client)

    response = await client.get_native_response(
        prompt=[{"type": "message", "role": "user", "content": "hi"}],
        model="gpt-5.2",
        sampling_args={"n": 1, "max_tokens": 12, "extra_body": {"foo": "bar"}},
        tools=[
            await client.to_native_tool(
                Tool(
                    name="lookup",
                    description="Lookup a thing",
                    parameters={"type": "object"},
                    strict=True,
                )
            )
        ],
        extra_headers={"X-Test": "1"},
    )

    assert response is native_response
    assert len(recording_client.responses.calls) == 1
    call = recording_client.responses.calls[0]
    assert call["model"] == "gpt-5.2"
    assert call["max_output_tokens"] == 12
    assert "max_tokens" not in call
    assert call["extra_body"] == {"foo": "bar"}
    assert call["extra_headers"] == {"X-Test": "1"}
    assert call["tools"] == [
        {
            "type": "function",
            "name": "lookup",
            "description": "Lookup a thing",
            "parameters": {"type": "object"},
            "strict": True,
        }
    ]


@pytest.mark.asyncio
async def test_to_native_tool_omits_strict_when_unset():
    client = OpenAIResponsesClient(object())

    native_tool = await client.to_native_tool(
        Tool(
            name="lookup",
            description="Lookup a thing",
            parameters={"type": "object"},
        )
    )

    assert native_tool == {
        "type": "function",
        "name": "lookup",
        "description": "Lookup a thing",
        "parameters": {"type": "object"},
    }


@pytest.mark.asyncio
async def test_get_native_response_drops_max_token_aliases_when_native_arg_is_set():
    recording_client = _RecordingOpenAI(SimpleNamespace(output=[]))
    client = OpenAIResponsesClient(recording_client)

    await client.get_native_response(
        prompt=[{"type": "message", "role": "user", "content": "hi"}],
        model="gpt-5.2",
        sampling_args={
            "max_output_tokens": 8,
            "max_tokens": 12,
            "max_completion_tokens": 16,
        },
    )

    call = recording_client.responses.calls[0]
    assert call["max_output_tokens"] == 8
    assert "max_tokens" not in call
    assert "max_completion_tokens" not in call


@pytest.mark.asyncio
async def test_from_native_response_parses_text_tool_usage_and_raw_output():
    native_response = SimpleNamespace(
        id="resp_1",
        created_at=123.0,
        model="gpt-5.2",
        status="completed",
        incomplete_details=None,
        usage={
            "input_tokens": 10,
            "output_tokens": 7,
            "total_tokens": 17,
            "output_tokens_details": {"reasoning_tokens": 3},
        },
        output=[
            {
                "type": "reasoning",
                "id": "rs_1",
                "summary": [{"type": "summary_text", "text": "thinking"}],
                "status": "completed",
            },
            {
                "type": "message",
                "id": "msg_1",
                "role": "assistant",
                "status": "completed",
                "content": [{"type": "output_text", "text": "hello"}],
            },
            {
                "type": "function_call",
                "call_id": "call_1",
                "name": "lookup",
                "arguments": '{"q":"x"}',
                "status": "completed",
            },
        ],
    )
    client = OpenAIResponsesClient(object())

    response = await client.from_native_response(native_response)

    assert response.id == "resp_1"
    assert response.created == 123
    assert response.model == "gpt-5.2"
    assert response.usage == Usage(
        prompt_tokens=10,
        reasoning_tokens=3,
        completion_tokens=7,
        total_tokens=17,
    )
    assert response.message.content == "hello"
    assert response.message.reasoning_content == "thinking"
    assert response.message.finish_reason == "tool_calls"
    assert response.message.tool_calls == [
        ToolCall(id="call_1", name="lookup", arguments='{"q":"x"}')
    ]
    assert (
        getattr(response.message, OPENAI_RESPONSES_OUTPUT_FIELD)
        == native_response.output
    )


@pytest.mark.asyncio
async def test_from_native_response_uses_none_content_for_tool_call_only_response():
    native_response = SimpleNamespace(
        id="resp_1",
        created_at=123.0,
        model="gpt-5.2",
        status="completed",
        incomplete_details=None,
        usage=None,
        output=[
            {
                "type": "function_call",
                "call_id": "call_1",
                "name": "lookup",
                "arguments": '{"q":"x"}',
                "status": "completed",
            },
        ],
    )
    client = OpenAIResponsesClient(object())

    await client.raise_from_native_response(native_response)
    response = await client.from_native_response(native_response)

    assert response.message.content is None
    assert response.message.finish_reason == "tool_calls"
    assert response.message.tool_calls == [
        ToolCall(id="call_1", name="lookup", arguments='{"q":"x"}')
    ]


@pytest.mark.asyncio
async def test_response_message_extras_round_trip_into_next_prompt():
    response = Response(
        id="resp_1",
        created=123,
        model="gpt-5.2",
        usage=None,
        message=ResponseMessage(
            content=None,
            reasoning_content=None,
            finish_reason="tool_calls",
            is_truncated=False,
            tokens=None,
            tool_calls=[ToolCall(id="call_1", name="lookup", arguments='{"q":"x"}')],
            **{
                OPENAI_RESPONSES_OUTPUT_FIELD: [
                    {
                        "type": "reasoning",
                        "id": "rs_1",
                        "summary": [],
                        "status": "completed",
                    },
                    {
                        "type": "function_call",
                        "call_id": "call_1",
                        "name": "lookup",
                        "arguments": '{"q":"x"}',
                        "status": "completed",
                    },
                ]
            },
        ),
    )
    completion_messages = await parse_response_message(response)

    client = OpenAIResponsesClient(object())
    prompt, _ = await client.to_native_prompt(
        [
            *completion_messages,
            ToolMessage(tool_call_id="call_1", content="result x"),
        ]
    )

    assert prompt == [
        {
            "type": "reasoning",
            "id": "rs_1",
            "summary": [],
            "status": "completed",
        },
        {
            "type": "function_call",
            "call_id": "call_1",
            "name": "lookup",
            "arguments": '{"q":"x"}',
            "status": "completed",
        },
        {
            "type": "function_call_output",
            "call_id": "call_1",
            "output": "result x",
        },
    ]


def test_resolve_client_supports_openai_responses():
    client = resolve_client(ClientConfig(client_type="openai_responses"))

    assert isinstance(client, OpenAIResponsesClient)
