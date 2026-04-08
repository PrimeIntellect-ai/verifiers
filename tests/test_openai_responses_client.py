from typing import Any

import httpx
import pytest
from openai import AuthenticationError, BadRequestError
from openai.types.responses import (
    FunctionTool,
    Response as OAIResponse,
    ResponseFunctionToolCall,
    ResponseOutputMessage,
    ResponseReasoningItem,
)
from openai.types.responses.response_error import ResponseError
from openai.types.responses.response_output_text import ResponseOutputText
from openai.types.responses.response_reasoning_item import Content as ReasoningContent
from openai.types.responses.response_reasoning_item import Summary
from openai.types.responses.response_usage import (
    InputTokensDetails,
    OutputTokensDetails,
    ResponseUsage,
)

from verifiers.clients.openai_responses_client import OpenAIResponsesClient
from verifiers.errors import (
    EmptyModelResponseError,
    InvalidModelResponseError,
    ModelError,
    OverlongPromptError,
)
from verifiers.types import (
    AssistantMessage,
    ImageUrlContentPart,
    ImageUrlSource,
    SystemMessage,
    TextContentPart,
    TextMessage,
    Tool,
    ToolCall,
    ToolMessage,
    Usage,
    UserMessage,
)


# ---------------------------------------------------------------------------
# Helpers for constructing native Responses API objects
# ---------------------------------------------------------------------------


def _make_usage(
    input_tokens: int = 10,
    output_tokens: int = 20,
    reasoning_tokens: int = 0,
) -> ResponseUsage:
    return ResponseUsage(
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        total_tokens=input_tokens + output_tokens,
        input_tokens_details=InputTokensDetails(cached_tokens=0),
        output_tokens_details=OutputTokensDetails(reasoning_tokens=reasoning_tokens),
    )


def _make_text_output(text: str, msg_id: str = "msg_1") -> ResponseOutputMessage:
    return ResponseOutputMessage(
        id=msg_id,
        type="message",
        role="assistant",
        status="completed",
        content=[ResponseOutputText(type="output_text", text=text, annotations=[])],
    )


def _make_response(
    output: list | None = None,
    status: str = "completed",
    usage: ResponseUsage | None = None,
    error: ResponseError | None = None,
    model: str = "gpt-4.1-nano",
) -> OAIResponse:
    return OAIResponse(
        id="resp_test",
        created_at=1700000000.0,
        model=model,
        object="response",
        status=status,
        output=output or [],
        tools=[],
        usage=usage or _make_usage(),
        parallel_tool_calls=True,
        temperature=0.0,
        tool_choice="auto",
        top_p=None,
        text=None,
        truncation=None,
        error=error,
    )


# ---------------------------------------------------------------------------
# to_native_prompt
# ---------------------------------------------------------------------------


class TestToNativePrompt:
    @pytest.mark.asyncio
    async def test_system_message_extracted_to_instructions(self):
        client = OpenAIResponsesClient(object())
        messages = [
            SystemMessage(content="You are helpful."),
            UserMessage(content="Hi"),
        ]

        items, kwargs = await client.to_native_prompt(messages)

        assert kwargs["instructions"] == "You are helpful."
        assert len(items) == 1
        assert items[0] == {"type": "message", "role": "user", "content": "Hi"}

    @pytest.mark.asyncio
    async def test_multiple_system_messages_joined(self):
        client = OpenAIResponsesClient(object())
        messages = [
            SystemMessage(content="Be concise."),
            SystemMessage(content="Use tools."),
            UserMessage(content="Hello"),
        ]

        items, kwargs = await client.to_native_prompt(messages)

        assert kwargs["instructions"] == "Be concise.\n\nUse tools."
        assert len(items) == 1

    @pytest.mark.asyncio
    async def test_no_system_message_omits_instructions(self):
        client = OpenAIResponsesClient(object())
        messages = [UserMessage(content="Hello")]

        items, kwargs = await client.to_native_prompt(messages)

        assert "instructions" not in kwargs
        assert len(items) == 1

    @pytest.mark.asyncio
    async def test_user_message(self):
        client = OpenAIResponsesClient(object())
        messages = [UserMessage(content="What is 2+2?")]

        items, _ = await client.to_native_prompt(messages)

        assert items[0] == {
            "type": "message",
            "role": "user",
            "content": "What is 2+2?",
        }

    @pytest.mark.asyncio
    async def test_text_message_treated_as_user(self):
        client = OpenAIResponsesClient(object())
        messages = [TextMessage(content="legacy input")]

        items, _ = await client.to_native_prompt(messages)

        assert items[0]["role"] == "user"
        assert items[0]["content"] == "legacy input"

    @pytest.mark.asyncio
    async def test_assistant_message_text_only(self):
        client = OpenAIResponsesClient(object())
        messages = [AssistantMessage(content="I can help with that.")]

        items, _ = await client.to_native_prompt(messages)

        assert len(items) == 1
        assert items[0] == {
            "type": "message",
            "role": "assistant",
            "content": "I can help with that.",
        }

    @pytest.mark.asyncio
    async def test_assistant_message_with_tool_calls(self):
        client = OpenAIResponsesClient(object())
        messages = [
            AssistantMessage(
                content=None,
                tool_calls=[
                    ToolCall(
                        id="call_1",
                        name="get_weather",
                        arguments='{"city": "Paris"}',
                    ),
                ],
            ),
        ]

        items, _ = await client.to_native_prompt(messages)

        # No content message (content is None), just the function_call
        assert len(items) == 1
        assert items[0] == {
            "type": "function_call",
            "call_id": "call_1",
            "name": "get_weather",
            "arguments": '{"city": "Paris"}',
        }

    @pytest.mark.asyncio
    async def test_assistant_message_with_content_and_tool_calls(self):
        client = OpenAIResponsesClient(object())
        messages = [
            AssistantMessage(
                content="Let me check that.",
                tool_calls=[
                    ToolCall(id="call_1", name="lookup", arguments='{"q": "x"}'),
                ],
            ),
        ]

        items, _ = await client.to_native_prompt(messages)

        assert len(items) == 2
        assert items[0]["type"] == "message"
        assert items[0]["role"] == "assistant"
        assert items[1]["type"] == "function_call"
        assert items[1]["call_id"] == "call_1"

    @pytest.mark.asyncio
    async def test_tool_message(self):
        client = OpenAIResponsesClient(object())
        messages = [ToolMessage(tool_call_id="call_1", content="Sunny, 22C")]

        items, _ = await client.to_native_prompt(messages)

        assert items[0] == {
            "type": "function_call_output",
            "call_id": "call_1",
            "output": "Sunny, 22C",
        }

    @pytest.mark.asyncio
    async def test_full_multi_turn_conversation(self):
        client = OpenAIResponsesClient(object())
        messages = [
            SystemMessage(content="You are helpful."),
            UserMessage(content="Weather in Paris?"),
            AssistantMessage(
                content=None,
                tool_calls=[
                    ToolCall(
                        id="call_1",
                        name="get_weather",
                        arguments='{"city": "Paris"}',
                    ),
                ],
            ),
            ToolMessage(tool_call_id="call_1", content="Sunny, 22C"),
        ]

        items, kwargs = await client.to_native_prompt(messages)

        assert kwargs["instructions"] == "You are helpful."
        assert len(items) == 3
        assert items[0]["type"] == "message"
        assert items[0]["role"] == "user"
        assert items[1]["type"] == "function_call"
        assert items[2]["type"] == "function_call_output"

    @pytest.mark.asyncio
    async def test_multimodal_content_parts(self):
        client = OpenAIResponsesClient(object())
        messages = [
            UserMessage(
                content=[
                    TextContentPart(text="describe this"),
                    ImageUrlContentPart(
                        image_url=ImageUrlSource(url="data:image/png;base64,abc123")
                    ),
                ]
            ),
        ]

        items, _ = await client.to_native_prompt(messages)

        assert items[0]["content"] == [
            {"type": "input_text", "text": "describe this"},
            {
                "type": "input_image",
                "image_url": "data:image/png;base64,abc123",
                "detail": "auto",
            },
        ]


# ---------------------------------------------------------------------------
# to_native_tool
# ---------------------------------------------------------------------------


class TestToNativeTool:
    @pytest.mark.asyncio
    async def test_basic_tool(self):
        client = OpenAIResponsesClient(object())
        tool = Tool(
            name="get_weather",
            description="Get weather for a city",
            parameters={
                "type": "object",
                "properties": {"city": {"type": "string"}},
                "required": ["city"],
            },
        )

        native = await client.to_native_tool(tool)

        assert isinstance(native, FunctionTool)
        assert native.type == "function"
        assert native.name == "get_weather"
        assert native.description == "Get weather for a city"
        assert native.parameters == tool.parameters
        assert native.strict is None

    @pytest.mark.asyncio
    async def test_strict_tool(self):
        client = OpenAIResponsesClient(object())
        tool = Tool(
            name="lookup",
            description="Look up data",
            parameters={"type": "object", "properties": {}},
            strict=True,
        )

        native = await client.to_native_tool(tool)

        assert native.strict is True


# ---------------------------------------------------------------------------
# from_native_response
# ---------------------------------------------------------------------------


class TestFromNativeResponse:
    @pytest.mark.asyncio
    async def test_text_only_response(self):
        client = OpenAIResponsesClient(object())
        native = _make_response(
            output=[_make_text_output("Hello!")],
            usage=_make_usage(input_tokens=10, output_tokens=5),
        )

        response = await client.from_native_response(native)

        assert response.message.content == "Hello!"
        assert response.message.tool_calls is None
        assert response.message.reasoning_content is None
        assert response.message.finish_reason == "stop"
        assert response.message.is_truncated is False
        assert response.id == "resp_test"
        assert response.model == "gpt-4.1-nano"

    @pytest.mark.asyncio
    async def test_tool_call_response(self):
        client = OpenAIResponsesClient(object())
        native = _make_response(
            output=[
                ResponseFunctionToolCall(
                    type="function_call",
                    call_id="call_abc",
                    name="get_weather",
                    arguments='{"city": "Paris"}',
                ),
            ],
        )

        response = await client.from_native_response(native)

        assert response.message.content is None
        assert response.message.finish_reason == "tool_calls"
        assert len(response.message.tool_calls) == 1
        tc = response.message.tool_calls[0]
        assert tc.id == "call_abc"
        assert tc.name == "get_weather"
        assert tc.arguments == '{"city": "Paris"}'

    @pytest.mark.asyncio
    async def test_multiple_tool_calls(self):
        client = OpenAIResponsesClient(object())
        native = _make_response(
            output=[
                ResponseFunctionToolCall(
                    type="function_call",
                    call_id="call_1",
                    name="tool_a",
                    arguments="{}",
                ),
                ResponseFunctionToolCall(
                    type="function_call",
                    call_id="call_2",
                    name="tool_b",
                    arguments="{}",
                ),
            ],
        )

        response = await client.from_native_response(native)

        assert len(response.message.tool_calls) == 2
        assert response.message.tool_calls[0].name == "tool_a"
        assert response.message.tool_calls[1].name == "tool_b"

    @pytest.mark.asyncio
    async def test_reasoning_from_summary(self):
        client = OpenAIResponsesClient(object())
        native = _make_response(
            output=[
                ResponseReasoningItem(
                    id="rs_1",
                    type="reasoning",
                    summary=[
                        Summary(type="summary_text", text="Thinking step by step")
                    ],
                ),
                _make_text_output("42"),
            ],
        )

        response = await client.from_native_response(native)

        assert response.message.content == "42"
        assert response.message.reasoning_content == "Thinking step by step"
        assert response.message.finish_reason == "stop"

    @pytest.mark.asyncio
    async def test_reasoning_prefers_content_over_summary(self):
        client = OpenAIResponsesClient(object())
        native = _make_response(
            output=[
                ResponseReasoningItem(
                    id="rs_1",
                    type="reasoning",
                    summary=[Summary(type="summary_text", text="brief summary")],
                    content=[
                        ReasoningContent(
                            type="reasoning_text", text="full reasoning chain"
                        )
                    ],
                ),
                _make_text_output("answer"),
            ],
        )

        response = await client.from_native_response(native)

        assert response.message.reasoning_content == "full reasoning chain"

    @pytest.mark.asyncio
    async def test_usage_mapping(self):
        client = OpenAIResponsesClient(object())
        native = _make_response(
            output=[_make_text_output("ok")],
            usage=_make_usage(input_tokens=42, output_tokens=17, reasoning_tokens=5),
        )

        response = await client.from_native_response(native)

        assert isinstance(response.usage, Usage)
        assert response.usage.prompt_tokens == 42
        assert response.usage.completion_tokens == 17
        assert response.usage.reasoning_tokens == 5
        assert response.usage.total_tokens == 59

    @pytest.mark.asyncio
    async def test_incomplete_status_maps_to_length(self):
        client = OpenAIResponsesClient(object())
        native = _make_response(
            output=[_make_text_output("partial...")],
            status="incomplete",
        )

        response = await client.from_native_response(native)

        assert response.message.finish_reason == "length"
        assert response.message.is_truncated is True

    @pytest.mark.asyncio
    async def test_created_at_converted_to_int(self):
        client = OpenAIResponsesClient(object())
        native = _make_response(output=[_make_text_output("ok")])

        response = await client.from_native_response(native)

        assert response.created == 1700000000
        assert isinstance(response.created, int)


# ---------------------------------------------------------------------------
# raise_from_native_response
# ---------------------------------------------------------------------------


class TestRaiseFromNativeResponse:
    @pytest.mark.asyncio
    async def test_error_response_raises(self):
        client = OpenAIResponsesClient(object())
        native = _make_response(
            error=ResponseError(code="server_error", message="something broke"),
        )

        with pytest.raises(InvalidModelResponseError, match="something broke"):
            await client.raise_from_native_response(native)

    @pytest.mark.asyncio
    async def test_failed_status_raises(self):
        client = OpenAIResponsesClient(object())
        native = _make_response(status="failed")

        with pytest.raises(EmptyModelResponseError, match="failed"):
            await client.raise_from_native_response(native)

    @pytest.mark.asyncio
    async def test_empty_output_raises(self):
        client = OpenAIResponsesClient(object())
        native = _make_response(output=[], status="completed")

        with pytest.raises(EmptyModelResponseError, match="no output items"):
            await client.raise_from_native_response(native)

    @pytest.mark.asyncio
    async def test_output_with_empty_message_raises(self):
        """Output has a message but no text content."""
        client = OpenAIResponsesClient(object())
        native = _make_response(
            output=[
                ResponseOutputMessage(
                    id="msg_1",
                    type="message",
                    role="assistant",
                    status="completed",
                    content=[
                        ResponseOutputText(type="output_text", text="", annotations=[])
                    ],
                ),
            ],
        )

        with pytest.raises(EmptyModelResponseError):
            await client.raise_from_native_response(native)

    @pytest.mark.asyncio
    async def test_valid_text_response_passes(self):
        client = OpenAIResponsesClient(object())
        native = _make_response(output=[_make_text_output("hello")])

        await client.raise_from_native_response(native)  # should not raise

    @pytest.mark.asyncio
    async def test_valid_tool_call_response_passes(self):
        client = OpenAIResponsesClient(object())
        native = _make_response(
            output=[
                ResponseFunctionToolCall(
                    type="function_call",
                    call_id="call_1",
                    name="f",
                    arguments="{}",
                ),
            ],
        )

        await client.raise_from_native_response(native)  # should not raise

    @pytest.mark.asyncio
    async def test_valid_reasoning_only_response_passes(self):
        client = OpenAIResponsesClient(object())
        native = _make_response(
            output=[
                ResponseReasoningItem(
                    id="rs_1",
                    type="reasoning",
                    summary=[Summary(type="summary_text", text="thinking...")],
                ),
            ],
        )

        await client.raise_from_native_response(native)  # should not raise


# ---------------------------------------------------------------------------
# get_native_response — sampling args normalization
# ---------------------------------------------------------------------------


class _RecordingResponses:
    """Fake responses endpoint that records create() calls."""

    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    async def create(self, **kwargs: Any) -> OAIResponse:
        self.calls.append(kwargs)
        return _make_response(output=[_make_text_output("ok")])


class _FakeClientWithResponses:
    def __init__(self) -> None:
        self.responses = _RecordingResponses()


class TestGetNativeResponse:
    @pytest.mark.asyncio
    async def test_temperature_and_top_p_passed_through(self):
        fake = _FakeClientWithResponses()
        client = OpenAIResponsesClient(fake)

        await client.get_native_response(
            prompt=[],
            model="gpt-4.1-nano",
            sampling_args={"temperature": 0.5, "top_p": 0.9},
        )

        call = fake.responses.calls[0]
        assert call["temperature"] == 0.5
        assert call["top_p"] == 0.9

    @pytest.mark.asyncio
    async def test_max_tokens_mapped_to_max_output_tokens(self):
        fake = _FakeClientWithResponses()
        client = OpenAIResponsesClient(fake)

        await client.get_native_response(
            prompt=[],
            model="gpt-4.1-nano",
            sampling_args={"max_tokens": 256},
        )

        assert fake.responses.calls[0]["max_output_tokens"] == 256

    @pytest.mark.asyncio
    async def test_max_completion_tokens_mapped_to_max_output_tokens(self):
        fake = _FakeClientWithResponses()
        client = OpenAIResponsesClient(fake)

        await client.get_native_response(
            prompt=[],
            model="gpt-4.1-nano",
            sampling_args={"max_completion_tokens": 512},
        )

        assert fake.responses.calls[0]["max_output_tokens"] == 512

    @pytest.mark.asyncio
    async def test_reasoning_effort_mapped_to_reasoning(self):
        fake = _FakeClientWithResponses()
        client = OpenAIResponsesClient(fake)

        await client.get_native_response(
            prompt=[],
            model="gpt-5.4",
            sampling_args={"reasoning_effort": "medium"},
        )

        assert fake.responses.calls[0]["reasoning"] == {"effort": "medium"}

    @pytest.mark.asyncio
    async def test_reasoning_summary_mapped_to_reasoning(self):
        fake = _FakeClientWithResponses()
        client = OpenAIResponsesClient(fake)

        await client.get_native_response(
            prompt=[],
            model="gpt-5.4",
            sampling_args={"reasoning_effort": "high", "reasoning_summary": "auto"},
        )

        assert fake.responses.calls[0]["reasoning"] == {
            "effort": "high",
            "summary": "auto",
        }

    @pytest.mark.asyncio
    async def test_n_stripped(self):
        fake = _FakeClientWithResponses()
        client = OpenAIResponsesClient(fake)

        await client.get_native_response(
            prompt=[],
            model="gpt-4.1-nano",
            sampling_args={"n": 3, "temperature": 0.0},
        )

        assert "n" not in fake.responses.calls[0]

    @pytest.mark.asyncio
    async def test_instructions_forwarded_from_kwargs(self):
        fake = _FakeClientWithResponses()
        client = OpenAIResponsesClient(fake)

        await client.get_native_response(
            prompt=[],
            model="gpt-4.1-nano",
            sampling_args={},
            instructions="Be helpful.",
        )

        assert fake.responses.calls[0]["instructions"] == "Be helpful."

    @pytest.mark.asyncio
    async def test_extra_headers_forwarded(self):
        fake = _FakeClientWithResponses()
        client = OpenAIResponsesClient(fake)

        await client.get_native_response(
            prompt=[],
            model="gpt-4.1-nano",
            sampling_args={},
            extra_headers={"X-Custom": "value"},
        )

        assert fake.responses.calls[0]["extra_headers"] == {"X-Custom": "value"}


# ---------------------------------------------------------------------------
# Error handling
# ---------------------------------------------------------------------------


def _make_bad_request_error(message: str) -> BadRequestError:
    response = httpx.Response(
        status_code=400,
        request=httpx.Request("POST", "https://api.openai.com/v1/responses"),
        text=message,
    )
    return BadRequestError(message, response=response, body=None)


class _FailingResponses:
    def __init__(self, error: Exception) -> None:
        self._error = error

    async def create(self, **kwargs: Any) -> None:
        raise self._error


class _FakeClientWithFailingResponses:
    def __init__(self, error: Exception) -> None:
        self.responses = _FailingResponses(error)


class TestErrorHandling:
    @pytest.mark.asyncio
    async def test_overlong_prompt_error(self):
        error = _make_bad_request_error("This model's maximum context length is 128000")
        client = OpenAIResponsesClient(_FakeClientWithFailingResponses(error))

        with pytest.raises(OverlongPromptError):
            await client.get_response(
                prompt=[UserMessage(content="test")],
                model="gpt-test",
                sampling_args={},
            )

    @pytest.mark.asyncio
    async def test_auth_error_not_wrapped(self):
        response = httpx.Response(
            status_code=401,
            request=httpx.Request("POST", "https://api.openai.com/v1/responses"),
            text="invalid api key",
        )
        error = AuthenticationError("auth failed", response=response, body=None)
        client = OpenAIResponsesClient(_FakeClientWithFailingResponses(error))

        with pytest.raises(AuthenticationError):
            await client.get_response(
                prompt=[UserMessage(content="test")],
                model="gpt-test",
                sampling_args={},
            )

    @pytest.mark.asyncio
    async def test_non_overlong_bad_request_becomes_model_error(self):
        error = _make_bad_request_error("unsupported parameter: reasoning.effort")
        client = OpenAIResponsesClient(_FakeClientWithFailingResponses(error))

        with pytest.raises(ModelError):
            await client.get_response(
                prompt=[UserMessage(content="test")],
                model="gpt-test",
                sampling_args={},
            )
