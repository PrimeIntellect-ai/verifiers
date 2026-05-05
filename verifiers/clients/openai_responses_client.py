from __future__ import annotations

import time
from collections.abc import Iterable, Mapping
from typing import Any, TypeAlias

from openai import AsyncOpenAI
from openai.types.responses import Response as OpenAIResponsesNativeResponse

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

OpenAIResponsesInput: TypeAlias = list[dict[str, Any]]
OpenAIResponsesTool: TypeAlias = dict[str, Any]

OPENAI_RESPONSES_OUTPUT_FIELD = "openai_responses_output"


def _as_mapping(value: Any) -> Mapping[str, Any] | None:
    if isinstance(value, Mapping):
        return value
    if hasattr(value, "model_dump"):
        dumped = value.model_dump(exclude_none=True)
        return dumped if isinstance(dumped, Mapping) else None
    return None


def _get_field(value: Any, key: str, default: Any = None) -> Any:
    if isinstance(value, Mapping):
        return value.get(key, default)
    return getattr(value, key, default)


def _model_dump(value: Any) -> dict[str, Any]:
    if isinstance(value, Mapping):
        return dict(value)
    if hasattr(value, "model_dump"):
        dumped = value.model_dump(exclude_none=True)
        if isinstance(dumped, dict):
            return dumped
    return {}


class OpenAIResponsesClient(
    Client[
        AsyncOpenAI,
        OpenAIResponsesInput,
        OpenAIResponsesNativeResponse,
        OpenAIResponsesTool,
    ]
):
    """Wrapper for OpenAI Responses API via AsyncOpenAI client."""

    def setup_client(self, config: ClientConfig) -> AsyncOpenAI:
        return setup_openai_client(config)

    async def close(self) -> None:
        await self.client.close()

    async def to_native_prompt(
        self, messages: Messages
    ) -> tuple[OpenAIResponsesInput, dict]:
        def normalize_content_part(part: Any) -> dict[str, Any]:
            part_map = _as_mapping(part)
            if part_map is None:
                raise ValueError(f"Invalid content part type: {type(part)}")

            part_type = part_map.get("type")
            if part_type == "text":
                text = part_map.get("text")
                if not isinstance(text, str):
                    raise ValueError("Text content part is missing string text")
                return {"type": "input_text", "text": text}
            if part_type == "input_text":
                return dict(part_map)
            if part_type == "image_url":
                image_url = part_map.get("image_url", {})
                url = image_url.get("url") if isinstance(image_url, Mapping) else None
                if not isinstance(url, str):
                    raise ValueError("Image content part is missing image_url.url")
                return {"type": "input_image", "image_url": url, "detail": "auto"}
            if part_type == "input_image":
                image_part = dict(part_map)
                image_part.setdefault("detail", "auto")
                return image_part
            if part_type == "input_audio":
                raise ValueError(
                    "Responses API client does not support input_audio content parts"
                )
            return dict(part_map)

        def normalize_message_content(content: Any) -> str | list[dict[str, Any]]:
            if isinstance(content, list):
                return [normalize_content_part(part) for part in content]
            if isinstance(content, str):
                return content
            if content is None:
                return ""
            return str(content)

        def tool_output_content(content: Any) -> str:
            if isinstance(content, str):
                return content
            text = content_to_text(content)
            return text if text else str(content)

        def raw_output_items(message: AssistantMessage) -> list[dict[str, Any]]:
            raw = getattr(message, OPENAI_RESPONSES_OUTPUT_FIELD, None)
            if raw is None:
                return []
            if not isinstance(raw, list):
                raise ValueError(
                    f"{OPENAI_RESPONSES_OUTPUT_FIELD} must be a list when present"
                )
            items: list[dict[str, Any]] = []
            for item in raw:
                dumped = _model_dump(item)
                if not dumped:
                    raise ValueError(
                        f"Invalid {OPENAI_RESPONSES_OUTPUT_FIELD} item: {type(item)}"
                    )
                items.append(dumped)
            return items

        def from_message(message: Message) -> list[dict[str, Any]]:
            if isinstance(message, SystemMessage):
                return [
                    {
                        "type": "message",
                        "role": "system",
                        "content": normalize_message_content(message.content),
                    }
                ]
            if isinstance(message, UserMessage):
                return [
                    {
                        "type": "message",
                        "role": "user",
                        "content": normalize_message_content(message.content),
                    }
                ]
            if isinstance(message, TextMessage):
                return [{"type": "message", "role": "user", "content": message.content}]
            if isinstance(message, ToolMessage):
                return [
                    {
                        "type": "function_call_output",
                        "call_id": message.tool_call_id,
                        "output": tool_output_content(message.content),
                    }
                ]
            if isinstance(message, AssistantMessage):
                raw_items = raw_output_items(message)
                if raw_items:
                    return raw_items

                items: list[dict[str, Any]] = []
                if content_to_text(message.content):
                    items.append(
                        {
                            "type": "message",
                            "role": "assistant",
                            "content": normalize_message_content(message.content),
                        }
                    )
                for tool_call in message.tool_calls or []:
                    items.append(
                        {
                            "type": "function_call",
                            "call_id": tool_call.id,
                            "name": tool_call.name,
                            "arguments": tool_call.arguments,
                        }
                    )
                return items
            raise ValueError(f"Invalid chat message: {message}")

        prompt: OpenAIResponsesInput = []
        for message in messages:
            prompt.extend(from_message(message))
        return prompt, {}

    async def to_native_tool(self, tool: Tool) -> OpenAIResponsesTool:
        native_tool: OpenAIResponsesTool = {
            "type": "function",
            "name": tool.name,
            "description": tool.description,
            "parameters": tool.parameters,
        }
        if tool.strict is not None:
            native_tool["strict"] = tool.strict
        return native_tool

    @handle_openai_overlong_prompt
    async def get_native_response(
        self,
        prompt: OpenAIResponsesInput,
        model: str,
        sampling_args: SamplingArgs,
        tools: list[OpenAIResponsesTool] | None = None,
        **kwargs,
    ) -> OpenAIResponsesNativeResponse:
        def normalize_sampling_args(
            sampling_args: SamplingArgs,
        ) -> tuple[dict[str, Any], dict[str, Any] | None]:
            sampling_args = dict(sampling_args)
            extra_body = sampling_args.pop("extra_body", None)
            n = sampling_args.pop("n", None)
            if n not in (None, 1):
                raise ValueError("Responses API client only supports n=1")
            max_tokens = sampling_args.pop("max_tokens", None)
            max_completion_tokens = sampling_args.pop("max_completion_tokens", None)
            if "max_output_tokens" not in sampling_args:
                sampling_args["max_output_tokens"] = (
                    max_tokens if max_tokens is not None else max_completion_tokens
                )
            if sampling_args.get("stop") is not None:
                raise ValueError("Responses API client does not support stop sequences")
            sampling_args.pop("stop", None)
            sampling_args.pop("modalities", None)
            normalized = {k: v for k, v in sampling_args.items() if v is not None}
            return normalized, extra_body if isinstance(extra_body, dict) else None

        extra_headers = kwargs.pop("extra_headers", None)
        kwargs.pop("state", None)
        normalized_args, extra_body = normalize_sampling_args(sampling_args)
        if extra_body is not None:
            existing_extra_body = kwargs.pop("extra_body", None)
            if isinstance(existing_extra_body, Mapping):
                extra_body = {**existing_extra_body, **extra_body}
            kwargs["extra_body"] = extra_body

        request_args: dict[str, Any] = {
            "model": model,
            "input": prompt,
            "extra_headers": extra_headers,
            **normalized_args,
            **kwargs,
        }
        if tools:
            request_args["tools"] = tools
        return await self.client.responses.create(**request_args)

    async def raise_from_native_response(
        self, response: OpenAIResponsesNativeResponse
    ) -> None:
        if response is None:
            raise EmptyModelResponseError("Model returned no response")
        error = getattr(response, "error", None)
        if error is not None:
            message = _get_field(error, "message", "Model response failed")
            raise InvalidModelResponseError(str(message))

        output = getattr(response, "output", None)
        if output is None:
            raise EmptyModelResponseError("Model returned no output")
        if not isinstance(output, Iterable):
            raise InvalidModelResponseError("Model returned invalid output")

        has_text = False
        has_tool_call = False
        has_reasoning = False
        for item in output:
            item_type = _get_field(item, "type")
            if item_type == "function_call":
                has_tool_call = True
            elif item_type == "reasoning":
                has_reasoning = True
            elif item_type == "message":
                for part in _get_field(item, "content", []) or []:
                    if _get_field(part, "type") == "output_text" and _get_field(
                        part, "text"
                    ):
                        has_text = True
                    if _get_field(part, "type") == "refusal" and _get_field(
                        part, "refusal"
                    ):
                        has_text = True

        if not (has_text or has_tool_call or has_reasoning):
            raise EmptyModelResponseError(
                "Model returned no content, reasoning, and did not call any tools"
            )

    async def from_native_response(
        self, response: OpenAIResponsesNativeResponse
    ) -> Response:
        def raw_output_items(
            response: OpenAIResponsesNativeResponse,
        ) -> list[dict[str, Any]]:
            items = []
            for item in getattr(response, "output", []) or []:
                dumped = _model_dump(item)
                if dumped:
                    items.append(dumped)
            return items

        def parse_content(response: OpenAIResponsesNativeResponse) -> str | None:
            chunks: list[str] = []
            for item in getattr(response, "output", []) or []:
                if _get_field(item, "type") != "message":
                    continue
                for part in _get_field(item, "content", []) or []:
                    part_type = _get_field(part, "type")
                    if part_type == "output_text":
                        text = _get_field(part, "text")
                        if isinstance(text, str):
                            chunks.append(text)
                    elif part_type == "refusal":
                        refusal = _get_field(part, "refusal")
                        if isinstance(refusal, str):
                            chunks.append(refusal)
            return "".join(chunks) or None

        def parse_tool_calls(response: OpenAIResponsesNativeResponse) -> list[ToolCall]:
            tool_calls: list[ToolCall] = []
            for item in getattr(response, "output", []) or []:
                if _get_field(item, "type") != "function_call":
                    continue
                call_id = _get_field(item, "call_id")
                name = _get_field(item, "name")
                arguments = _get_field(item, "arguments")
                if (
                    isinstance(call_id, str)
                    and isinstance(name, str)
                    and isinstance(arguments, str)
                ):
                    tool_calls.append(
                        ToolCall(id=call_id, name=name, arguments=arguments)
                    )
            return tool_calls

        def parse_reasoning_content(
            response: OpenAIResponsesNativeResponse,
        ) -> str | None:
            chunks: list[str] = []
            for item in getattr(response, "output", []) or []:
                if _get_field(item, "type") != "reasoning":
                    continue
                for summary in _get_field(item, "summary", []) or []:
                    text = _get_field(summary, "text")
                    if isinstance(text, str):
                        chunks.append(text)
                for content in _get_field(item, "content", []) or []:
                    text = _get_field(content, "text")
                    if isinstance(text, str):
                        chunks.append(text)
            return "\n".join(chunks) or None

        def parse_usage(response: OpenAIResponsesNativeResponse) -> Usage | None:
            usage = getattr(response, "usage", None)
            if usage is None:
                return None
            prompt_tokens = get_usage_field(usage, "input_tokens")
            completion_tokens = get_usage_field(usage, "output_tokens")
            total_tokens = get_usage_field(usage, "total_tokens")
            output_details = get_usage_field(usage, "output_tokens_details")
            reasoning_tokens = (
                get_usage_field(output_details, "reasoning_tokens")
                if output_details is not None
                else 0
            )
            if not isinstance(prompt_tokens, int) or not isinstance(
                completion_tokens, int
            ):
                return None
            if not isinstance(total_tokens, int):
                total_tokens = prompt_tokens + completion_tokens
            if not isinstance(reasoning_tokens, int):
                reasoning_tokens = 0
            return Usage(
                prompt_tokens=prompt_tokens,
                reasoning_tokens=reasoning_tokens,
                completion_tokens=completion_tokens,
                total_tokens=total_tokens,
            )

        def parse_is_truncated(response: OpenAIResponsesNativeResponse) -> bool:
            incomplete_details = getattr(response, "incomplete_details", None)
            return getattr(response, "status", None) == "incomplete" or bool(
                incomplete_details
            )

        def parse_finish_reason(
            response: OpenAIResponsesNativeResponse,
        ) -> FinishReason:
            if parse_tool_calls(response):
                return "tool_calls"
            if parse_is_truncated(response):
                return "length"
            if getattr(response, "status", None) == "completed":
                return "stop"
            return None

        response_id = getattr(response, "id", "")
        if not isinstance(response_id, str):
            response_id = ""
        created_at = getattr(response, "created_at", time.time())
        created = int(created_at) if isinstance(created_at, (int, float)) else 0
        model = getattr(response, "model", "")
        if not isinstance(model, str):
            model = ""

        message_data: dict[str, Any] = {
            "content": parse_content(response),
            "reasoning_content": parse_reasoning_content(response),
            "finish_reason": parse_finish_reason(response),
            "is_truncated": parse_is_truncated(response),
            "tokens": None,
            "tool_calls": parse_tool_calls(response) or None,
            OPENAI_RESPONSES_OUTPUT_FIELD: raw_output_items(response),
        }

        return Response(
            id=response_id,
            created=created,
            model=model,
            usage=parse_usage(response),
            message=ResponseMessage.model_validate(message_data),
        )
