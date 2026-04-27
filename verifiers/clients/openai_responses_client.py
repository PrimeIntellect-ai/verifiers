"""
OpenAI Responses API client for verifiers.

This mirrors the interface of OpenAIChatCompletionsClient but targets
the OpenAI Responses API (client.responses.create) instead of the
Chat Completions API (client.chat.completions.create).

Key differences from Chat Completions:
  - System messages become the `instructions` parameter
  - Input is a list of typed items (messages, function_call_outputs, etc.)
  - Output is a list of typed items (ResponseOutputMessage, ResponseFunctionToolCall,
    ResponseReasoningItem) instead of choices[0].message
  - Tool definitions use FunctionTool (flat) instead of ChatCompletionToolParam (nested)
  - Tool calls are separate output items with call_id/name/arguments
  - Tool results are FunctionCallOutput input items (type="function_call_output")
  - Reasoning is exposed as ResponseReasoningItem output items
  - reasoning_effort is passed via the `reasoning` parameter
  - Usage fields are input_tokens/output_tokens (not prompt_tokens/completion_tokens)
"""

from __future__ import annotations

import functools
from collections.abc import Mapping
from typing import Any, TypeAlias, cast

from openai import (
    AsyncOpenAI,
    AuthenticationError,
    BadRequestError,
    NotGiven,
    PermissionDeniedError,
)
from openai.types.responses import (
    FunctionTool,
    Response as OAIResponsesResponse,
    ResponseFunctionToolCall,
    ResponseOutputMessage,
    ResponseReasoningItem,
)
from openai.types.responses.response_output_text import ResponseOutputText

from verifiers.clients.client import Client
from verifiers.errors import (
    EmptyModelResponseError,
    InvalidModelResponseError,
    OverlongPromptError,
)
from verifiers.types import (
    AssistantMessage,
    ClientConfig,
    FinishReason,
    Messages,
    Response,
    ResponseMessage,
    SamplingArgs,
    SystemMessage,
    TextMessage,
    Tool,
    ToolCall,
    ToolMessage,
    Usage,
    UserMessage,
)
from verifiers.utils.client_utils import setup_openai_client

NOT_GIVEN = NotGiven()


def handle_openai_responses_errors(func):
    """Decorator to handle overlong prompt errors from the Responses API."""

    @functools.wraps(func)
    async def wrapper(*args, **kwargs):
        try:
            return await func(*args, **kwargs)
        except (AuthenticationError, PermissionDeniedError):
            raise
        except BadRequestError as e:
            error_text = e.response.text.lower()
            context_length_phrases = [
                "this model's maximum context length is",
                "is longer than the model's context length",
                "exceeds the model's context length",
                "exceed the configured limit",
                "exceeds the configured limit",
                "exceeded model",
                "prompt_too_long",
                "context length",
            ]
            if any(phrase in error_text for phrase in context_length_phrases):
                raise OverlongPromptError from e
            raise

    return wrapper


# -- Native types for the Responses API --
ResponsesInputItem: TypeAlias = dict[str, Any]
ResponsesInput: TypeAlias = list[ResponsesInputItem]
ResponsesResponse: TypeAlias = OAIResponsesResponse
ResponsesTool: TypeAlias = FunctionTool


def _content_to_str(content: Any) -> str:
    """Extract plain text from message content (str or list of content parts)."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        chunks: list[str] = []
        for part in content:
            if isinstance(part, Mapping):
                if part.get("type") == "text":
                    text = part.get("text")
                    if isinstance(text, str):
                        chunks.append(text)
            elif hasattr(part, "text"):
                text = getattr(part, "text", None)
                if isinstance(text, str):
                    chunks.append(text)
        return " ".join(chunks).strip()
    return ""


def _convert_content_parts(content: Any) -> str | list[dict[str, Any]]:
    """Convert vf content (str or list of ContentPart) to Responses API input content.

    String content is passed through directly.
    List content parts are converted:
      {"type": "text", "text": ...}  ->  {"type": "input_text", "text": ...}
      {"type": "image_url", "image_url": {"url": ...}}  ->  {"type": "input_image", "image_url": url, "detail": "auto"}
    """
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        return str(content) if content else ""

    parts: list[dict[str, Any]] = []
    for part in content:
        if isinstance(part, Mapping):
            part_dict = dict(part)
        elif hasattr(part, "model_dump"):
            part_dict = part.model_dump()
        else:
            continue

        part_type = part_dict.get("type", "")
        if part_type == "text":
            parts.append({"type": "input_text", "text": part_dict.get("text", "")})
        elif part_type == "image_url":
            image_url_obj = part_dict.get("image_url", {})
            url = (
                image_url_obj.get("url", "") if isinstance(image_url_obj, dict) else ""
            )
            parts.append({"type": "input_image", "image_url": url, "detail": "auto"})
        # Skip unsupported content part types

    return parts if parts else ""


class OpenAIResponsesClient(
    Client[
        AsyncOpenAI,
        ResponsesInput,
        ResponsesResponse,
        ResponsesTool,
    ]
):
    """Wrapper for the OpenAI Responses API via AsyncOpenAI client."""

    def setup_client(self, config: ClientConfig) -> AsyncOpenAI:
        return setup_openai_client(config)

    async def close(self) -> None:
        await self.client.close()

    # ---- Prompt conversion ----

    async def to_native_prompt(self, messages: Messages) -> tuple[ResponsesInput, dict]:
        """Convert vf.Messages to Responses API input items.

        System messages are extracted into an `instructions` extra kwarg.
        All other messages become typed input items.
        """
        input_items: ResponsesInput = []
        instructions_parts: list[str] = []

        for message in messages:
            if isinstance(message, SystemMessage):
                instructions_parts.append(_content_to_str(message.content))

            elif isinstance(message, (UserMessage, TextMessage)):
                content = message.content
                input_items.append(
                    {
                        "type": "message",
                        "role": "user",
                        "content": _convert_content_parts(content),
                    }
                )

            elif isinstance(message, AssistantMessage):
                # If the assistant message has text content, emit a message item
                if message.content:
                    input_items.append(
                        {
                            "type": "message",
                            "role": "assistant",
                            "content": _convert_content_parts(message.content),
                        }
                    )

                # Each tool call becomes a separate function_call input item
                if message.tool_calls:
                    for tc in message.tool_calls:
                        input_items.append(
                            {
                                "type": "function_call",
                                "call_id": tc.id,
                                "name": tc.name,
                                "arguments": tc.arguments,
                            }
                        )

            elif isinstance(message, ToolMessage):
                input_items.append(
                    {
                        "type": "function_call_output",
                        "call_id": message.tool_call_id,
                        "output": _content_to_str(message.content),
                    }
                )

        extra_kwargs: dict[str, Any] = {}
        if instructions_parts:
            extra_kwargs["instructions"] = "\n\n".join(instructions_parts)

        return input_items, extra_kwargs

    # ---- Tool conversion ----

    async def to_native_tool(self, tool: Tool) -> ResponsesTool:
        """Convert a vf.Tool to a Responses API FunctionTool."""
        return FunctionTool(
            type="function",
            name=tool.name,
            description=tool.description,
            parameters=tool.parameters,
            strict=tool.strict,
        )

    # ---- API call ----

    @handle_openai_responses_errors
    async def get_native_response(
        self,
        prompt: ResponsesInput,
        model: str,
        sampling_args: SamplingArgs,
        tools: list[ResponsesTool] | None = None,
        **kwargs,
    ) -> ResponsesResponse:
        """Call client.responses.create() with the converted inputs."""
        sampling_args = dict(sampling_args)

        # Extract instructions (set by to_native_prompt via extra_kwargs)
        instructions = kwargs.pop("instructions", None)

        # Extra headers (set by Client._build_state_headers)
        extra_headers = kwargs.pop("extra_headers", None)

        # -- Map sampling args --

        # temperature, top_p pass through directly
        temperature: float | None = None
        if "temperature" in sampling_args:
            temperature = sampling_args.pop("temperature")

        top_p: float | None = None
        if "top_p" in sampling_args:
            top_p = sampling_args.pop("top_p")

        # max_tokens / max_completion_tokens -> max_output_tokens
        max_output_tokens: int | None = sampling_args.pop(
            "max_tokens", None
        ) or sampling_args.pop("max_completion_tokens", None)

        # reasoning_effort / reasoning_summary -> reasoning={"effort": ..., "summary": ...}
        reasoning_effort = sampling_args.pop("reasoning_effort", None)
        reasoning_summary = sampling_args.pop("reasoning_summary", None)
        reasoning_param: dict[str, Any] | None = None
        if reasoning_effort is not None or reasoning_summary is not None:
            reasoning_param = {}
            if reasoning_effort is not None:
                reasoning_param["effort"] = reasoning_effort
            if reasoning_summary is not None:
                reasoning_param["summary"] = reasoning_summary

        # Strip params not supported by Responses API
        sampling_args.pop("n", None)
        sampling_args.pop("extra_body", None)

        response = await self.client.responses.create(
            model=model,
            input=cast(Any, prompt),
            instructions=instructions,
            temperature=temperature,
            top_p=top_p,
            max_output_tokens=max_output_tokens,
            reasoning=cast(Any, reasoning_param),
            tools=cast(Any, tools or NOT_GIVEN),
            extra_headers=extra_headers,
        )
        return response

    # ---- Response validation ----

    async def raise_from_native_response(self, response: ResponsesResponse) -> None:
        """Validate the Responses API response, raising on errors."""
        if response.error is not None:
            raise InvalidModelResponseError(
                f"Responses API error: {response.error.code} - {response.error.message}"
            )

        if response.status not in ("completed",):
            raise EmptyModelResponseError(
                f"Response status is '{response.status}', expected 'completed'"
            )

        if not response.output:
            raise EmptyModelResponseError("Response has no output items")

        has_content = False
        has_tool_calls = False
        has_reasoning = False

        for item in response.output:
            if isinstance(item, ResponseOutputMessage):
                for content_part in item.content:
                    if (
                        isinstance(content_part, ResponseOutputText)
                        and content_part.text
                    ):
                        has_content = True
            elif isinstance(item, ResponseFunctionToolCall):
                has_tool_calls = True
            elif isinstance(item, ResponseReasoningItem):
                has_reasoning = True

        if not (has_content or has_tool_calls or has_reasoning):
            raise EmptyModelResponseError(
                "Response has no content, reasoning, and no tool calls"
            )

    # ---- Response conversion ----

    async def from_native_response(self, response: ResponsesResponse) -> Response:
        """Convert an OpenAI Responses API response to a vf.Response."""

        # -- Collect content, tool calls, and reasoning from output items --
        content_texts: list[str] = []
        tool_calls: list[ToolCall] = []
        reasoning_texts: list[str] = []

        for item in response.output:
            if isinstance(item, ResponseOutputMessage):
                for content_part in item.content:
                    if (
                        isinstance(content_part, ResponseOutputText)
                        and content_part.text
                    ):
                        content_texts.append(content_part.text)

            elif isinstance(item, ResponseFunctionToolCall):
                tool_calls.append(
                    ToolCall(
                        id=item.call_id,
                        name=item.name,
                        arguments=item.arguments,
                    )
                )

            elif isinstance(item, ResponseReasoningItem):
                # Prefer full content if available, otherwise use summary
                if item.content:
                    for block in item.content:
                        text = getattr(block, "text", None)
                        if text:
                            reasoning_texts.append(text)
                elif item.summary:
                    for summary in item.summary:
                        text = getattr(summary, "text", None)
                        if text:
                            reasoning_texts.append(text)

        content_text = "\n\n".join(content_texts) if content_texts else None
        reasoning_text = "\n\n".join(reasoning_texts) if reasoning_texts else None

        # -- Finish reason --
        finish_reason: FinishReason
        if tool_calls:
            finish_reason = "tool_calls"
        elif response.status == "incomplete":
            finish_reason = "length"
        else:
            finish_reason = "stop"

        # -- Usage --
        usage: Usage | None = None
        if response.usage is not None:
            reasoning_tokens = 0
            if response.usage.output_tokens_details is not None:
                reasoning_tokens = response.usage.output_tokens_details.reasoning_tokens
            usage = Usage(
                prompt_tokens=response.usage.input_tokens,
                completion_tokens=response.usage.output_tokens,
                reasoning_tokens=reasoning_tokens,
                total_tokens=response.usage.total_tokens,
            )

        # -- Model name (may be a string or enum value) --
        model = str(response.model) if response.model else ""

        return Response(
            id=response.id or "",
            created=int(response.created_at) if response.created_at else 0,
            model=model,
            usage=usage,
            message=ResponseMessage(
                content=content_text,
                reasoning_content=reasoning_text,
                finish_reason=finish_reason,
                is_truncated=(response.status == "incomplete"),
                tokens=None,
                tool_calls=tool_calls or None,
            ),
        )
