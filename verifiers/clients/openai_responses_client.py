from __future__ import annotations

import time
from typing import Any, cast

from openai import AsyncOpenAI

from verifiers.clients.client import Client
from verifiers.clients.openai_chat_completions_client import (
    content_to_text,
    get_usage_field,
    handle_openai_overlong_prompt,
)
from verifiers.errors import EmptyModelResponseError, InvalidModelResponseError
from verifiers.types import (
    AssistantMessage,
    ClientConfig,
    FinishReason,
    Message,
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


OpenAIResponsesInput = list[dict[str, Any]]
OpenAIResponsesTool = dict[str, Any]
OpenAIResponsesResponse = Any


class OpenAIResponsesClient(
    Client[
        AsyncOpenAI,
        OpenAIResponsesInput,
        OpenAIResponsesResponse,
        OpenAIResponsesTool,
    ]
):
    """Wrapper for OpenAI-compatible Responses API via AsyncOpenAI client."""

    def setup_client(self, config: ClientConfig) -> AsyncOpenAI:
        return setup_openai_client(config)

    async def close(self) -> None:
        await self.client.close()

    async def to_native_prompt(
        self, messages: Messages
    ) -> tuple[OpenAIResponsesInput, dict]:
        def message_content(content: object) -> object:
            if isinstance(content, list):
                parts: list[object] = []
                for part in content:
                    model_dump = getattr(part, "model_dump", None)
                    if callable(model_dump):
                        parts.append(cast(dict[str, Any], model_dump()))
                    else:
                        parts.append(part)
                return parts
            return content

        def from_message(message: Message) -> list[dict[str, Any]]:
            if isinstance(message, SystemMessage):
                return [
                    {
                        "type": "message",
                        "role": "system",
                        "content": message_content(message.content),
                    }
                ]
            if isinstance(message, UserMessage):
                return [
                    {
                        "type": "message",
                        "role": "user",
                        "content": message_content(message.content),
                    }
                ]
            if isinstance(message, TextMessage):
                return [
                    {
                        "type": "message",
                        "role": "user",
                        "content": message.content,
                    }
                ]
            if isinstance(message, ToolMessage):
                return [
                    {
                        "type": "function_call_output",
                        "call_id": message.tool_call_id,
                        "output": content_to_text(message.content),
                    }
                ]
            if isinstance(message, AssistantMessage):
                items: list[dict[str, Any]] = []
                if message.content:
                    items.append(
                        {
                            "type": "message",
                            "role": "assistant",
                            "content": message_content(message.content),
                        }
                    )
                for tool_call in message.tool_calls or []:
                    items.append(
                        {
                            "type": "function_call",
                            "call_id": tool_call.id,
                            "id": tool_call.id,
                            "name": tool_call.name,
                            "arguments": tool_call.arguments,
                            "status": "completed",
                        }
                    )
                return items
            raise ValueError(f"Invalid responses message: {message}")

        prompt: OpenAIResponsesInput = []
        for message in messages:
            prompt.extend(from_message(message))
        return prompt, {}

    async def to_native_tool(self, tool: Tool) -> OpenAIResponsesTool:
        return {
            "type": "function",
            "name": tool.name,
            "description": tool.description,
            "parameters": tool.parameters,
            "strict": tool.strict,
        }

    @handle_openai_overlong_prompt
    async def get_native_response(
        self,
        prompt: OpenAIResponsesInput,
        model: str,
        sampling_args: SamplingArgs,
        tools: list[OpenAIResponsesTool] | None = None,
        **kwargs: Any,
    ) -> OpenAIResponsesResponse:
        sampling = dict(sampling_args)
        if "max_tokens" in sampling:
            sampling["max_output_tokens"] = sampling.pop("max_tokens")
        sampling.pop("n", None)
        sampling = {key: value for key, value in sampling.items() if value is not None}

        extra_headers = kwargs.pop("extra_headers", None)
        kwargs.pop("state", None)
        responses = cast(Any, self.client.responses)
        return await responses.create(
            model=model,
            input=prompt,
            tools=tools,
            extra_headers=extra_headers,
            **sampling,
            **kwargs,
        )

    async def raise_from_native_response(
        self, response: OpenAIResponsesResponse
    ) -> None:
        if response is None:
            raise EmptyModelResponseError("Model returned no response")
        if getattr(response, "status", None) == "failed":
            error = getattr(response, "error", None)
            raise InvalidModelResponseError(f"Responses API request failed: {error}")
        has_output = bool(getattr(response, "output", None))
        if not has_output:
            raise EmptyModelResponseError("Model returned no response output")

    async def from_native_response(self, response: OpenAIResponsesResponse) -> Response:
        content = ""
        tool_calls: list[ToolCall] = []

        for item in getattr(response, "output", []) or []:
            item_type = getattr(item, "type", None)
            if item_type == "message":
                for part in getattr(item, "content", []) or []:
                    text = getattr(part, "text", None)
                    if isinstance(text, str):
                        content += text
                continue
            if item_type == "function_call":
                call_id = getattr(item, "call_id", None) or getattr(item, "id", None)
                name = getattr(item, "name", None)
                arguments = getattr(item, "arguments", None)
                if (
                    isinstance(call_id, str)
                    and isinstance(name, str)
                    and isinstance(arguments, str)
                ):
                    tool_calls.append(
                        ToolCall(id=call_id, name=name, arguments=arguments)
                    )

        created = getattr(response, "created_at", None)
        if isinstance(created, float):
            created_int = int(created)
        elif isinstance(created, int):
            created_int = created
        else:
            created_int = int(time.time())

        response_id = getattr(response, "id", "")
        model = getattr(response, "model", "")
        return Response(
            id=response_id if isinstance(response_id, str) else "",
            created=created_int,
            model=model if isinstance(model, str) else "",
            usage=parse_usage(response),
            message=ResponseMessage(
                content=content or None,
                finish_reason=parse_finish_reason(response, tool_calls),
                is_truncated=getattr(response, "status", None) == "incomplete",
                tokens=None,
                tool_calls=tool_calls or None,
            ),
        )


def parse_usage(response: object) -> Usage | None:
    usage = getattr(response, "usage", None)
    if usage is None:
        return None
    prompt_tokens = get_usage_field(usage, "input_tokens")
    completion_tokens = get_usage_field(usage, "output_tokens")
    total_tokens = get_usage_field(usage, "total_tokens")
    if not isinstance(prompt_tokens, int) or not isinstance(completion_tokens, int):
        return None
    if not isinstance(total_tokens, int):
        total_tokens = prompt_tokens + completion_tokens
    return Usage(
        prompt_tokens=prompt_tokens,
        reasoning_tokens=0,
        completion_tokens=completion_tokens,
        total_tokens=total_tokens,
    )


def parse_finish_reason(response: object, tool_calls: list[ToolCall]) -> FinishReason:
    if tool_calls:
        return "tool_calls"
    if getattr(response, "status", None) == "incomplete":
        return "length"
    return "stop"
