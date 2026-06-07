from __future__ import annotations

import json

from aiohttp import web
from pydantic import TypeAdapter

from verifiers.types import (
    AssistantMessage,
    Message,
    Messages,
    Response,
    SystemMessage,
    TextMessage,
    Tool,
    ToolCall,
    ToolMessage,
    UserMessage,
)

from .interception import EndpointProtocol, InterceptedRequest, ProtocolRoute
from .types import JsonData, JsonValue
from .utils.json_utils import json_data, json_value

_MESSAGES_ADAPTER = TypeAdapter(Messages)


class OpenAIProtocol:
    def env(self, *, base_url: str, api_key: str, model: str) -> dict[str, str]:
        return {
            "OPENAI_BASE_URL": f"{base_url.rstrip('/')}/v1",
            "OPENAI_API_KEY": api_key,
            "OPENAI_MODEL": model,
        }

    def usage(self, response: Response) -> JsonData | None:
        if response.usage is None:
            return None
        return {
            "prompt_tokens": response.usage.prompt_tokens,
            "completion_tokens": response.usage.completion_tokens,
            "total_tokens": response.usage.total_tokens,
        }

    def tool_calls(self, response: Response) -> list[JsonValue] | None:
        calls = response.message.tool_calls
        if not calls:
            return None
        return [
            {
                "id": call.id,
                "type": "function",
                "function": {"name": call.name, "arguments": call.arguments},
            }
            for call in calls
        ]


class OpenAIChatCompletionsProtocol(OpenAIProtocol, EndpointProtocol):
    name = "openai_chat_completions"
    routes = (ProtocolRoute("POST", "/v1/chat/completions"),)

    async def parse(self, request: web.Request, body: JsonData) -> InterceptedRequest:
        _ = request
        return InterceptedRequest(
            protocol=self.name,
            prompt=parse_openai_messages(body.get("messages")),
            model=string_value(body.get("model")),
            sampling_args=openai_sampling_args(body),
            tools=parse_openai_tools(body.get("tools")),
            body=body,
        )

    def serialize(self, response: Response, request: InterceptedRequest) -> JsonData:
        message: JsonData = {
            "role": "assistant",
            "content": message_content(response.message.content),
        }
        tool_calls = self.tool_calls(response)
        if tool_calls:
            message["tool_calls"] = tool_calls
        return {
            "id": response.id or "vf-v1-intercept",
            "object": "chat.completion",
            "created": response.created,
            "model": response.model or request.model or "",
            "choices": [
                {
                    "index": 0,
                    "message": message,
                    "finish_reason": response.message.finish_reason or "stop",
                }
            ],
            "usage": self.usage(response),
        }


class OpenAICompletionsProtocol(OpenAIProtocol, EndpointProtocol):
    name = "openai_completions"
    routes = (ProtocolRoute("POST", "/v1/completions"),)

    async def parse(self, request: web.Request, body: JsonData) -> InterceptedRequest:
        _ = request
        raw_prompt = body.get("prompt")
        return InterceptedRequest(
            protocol=self.name,
            prompt=(
                [TextMessage(content=raw_prompt)]
                if isinstance(raw_prompt, str)
                else _MESSAGES_ADAPTER.validate_python(raw_prompt or [])
            ),
            model=string_value(body.get("model")),
            sampling_args=openai_sampling_args(body),
            body=body,
        )

    def serialize(self, response: Response, request: InterceptedRequest) -> JsonData:
        return {
            "id": response.id or "vf-v1-intercept",
            "object": "text_completion",
            "created": response.created,
            "model": response.model or request.model or "",
            "choices": [
                {
                    "index": 0,
                    "text": content_text(response.message.content),
                    "finish_reason": response.message.finish_reason or "stop",
                }
            ],
            "usage": self.usage(response),
        }


class OpenAIResponsesProtocol(OpenAIProtocol, EndpointProtocol):
    name = "openai_responses"
    routes = (ProtocolRoute("POST", "/v1/responses"),)

    async def parse(self, request: web.Request, body: JsonData) -> InterceptedRequest:
        _ = request
        return InterceptedRequest(
            protocol=self.name,
            prompt=parse_openai_responses_input(body.get("input")),
            model=string_value(body.get("model")),
            sampling_args=openai_sampling_args(body),
            tools=parse_responses_tools(body.get("tools")),
            body=body,
        )

    def serialize(self, response: Response, request: InterceptedRequest) -> JsonData:
        output: list[JsonValue] = []
        tool_calls = response.message.tool_calls or []
        for call in tool_calls:
            output.append(
                {
                    "type": "function_call",
                    "id": call.id,
                    "call_id": call.id,
                    "name": call.name,
                    "arguments": call.arguments,
                }
            )
        content = content_text(response.message.content)
        if content:
            output.append(
                {
                    "type": "message",
                    "role": "assistant",
                    "content": [{"type": "output_text", "text": content}],
                }
            )
        return json_data(
            {
                "id": response.id or "vf-v1-intercept",
                "object": "response",
                "created_at": response.created,
                "model": response.model or request.model or "",
                "output": output,
                "output_text": content,
                "usage": self.usage(response),
            },
            context="OpenAI responses serialization",
        )


class AnthropicMessagesProtocol(EndpointProtocol):
    name = "anthropic_messages"
    routes = (ProtocolRoute("POST", "/v1/messages"),)

    def env(self, *, base_url: str, api_key: str, model: str) -> dict[str, str]:
        return {
            "ANTHROPIC_BASE_URL": base_url.rstrip("/"),
            "ANTHROPIC_API_KEY": api_key,
            "ANTHROPIC_MODEL": model,
        }

    async def parse(self, request: web.Request, body: JsonData) -> InterceptedRequest:
        _ = request
        return InterceptedRequest(
            protocol=self.name,
            prompt=parse_anthropic_messages(body),
            model=string_value(body.get("model")),
            sampling_args=anthropic_sampling_args(body),
            tools=parse_anthropic_tools(body.get("tools")),
            body=body,
        )

    def serialize(self, response: Response, request: InterceptedRequest) -> JsonData:
        content: list[JsonValue] = []
        text = content_text(response.message.content)
        if text:
            content.append({"type": "text", "text": text})
        for call in response.message.tool_calls or []:
            content.append(
                {
                    "type": "tool_use",
                    "id": call.id,
                    "name": call.name,
                    "input": json.loads(call.arguments or "{}"),
                }
            )
        usage = None
        if response.usage is not None:
            usage = {
                "input_tokens": response.usage.prompt_tokens,
                "output_tokens": response.usage.completion_tokens,
            }
        return json_data(
            {
                "id": response.id or "vf-v1-intercept",
                "type": "message",
                "role": "assistant",
                "model": response.model or request.model or "",
                "content": content,
                "stop_reason": response.message.finish_reason or "end_turn",
                "usage": usage,
            },
            context="Anthropic messages serialization",
        )


def default_protocols() -> list[EndpointProtocol]:
    return [
        OpenAIChatCompletionsProtocol(),
        OpenAICompletionsProtocol(),
        OpenAIResponsesProtocol(),
        AnthropicMessagesProtocol(),
    ]


def parse_openai_messages(raw: JsonValue | None) -> Messages:
    if not isinstance(raw, list):
        raise TypeError("OpenAI messages must be a list.")
    messages: Messages = []
    for item in raw:
        if not isinstance(item, dict):
            raise TypeError("OpenAI message entries must be objects.")
        messages.append(parse_openai_message(json_data(item)))
    return messages


def parse_openai_message(raw: JsonData) -> Message:
    role = raw.get("role")
    content = raw.get("content")
    if role == "system":
        return SystemMessage(content=content_text(content))
    if role == "tool":
        return ToolMessage(
            tool_call_id=str(raw.get("tool_call_id") or ""),
            content=content_text(content),
        )
    if role == "assistant":
        return AssistantMessage(
            content=content_text(content) if content is not None else None,
            tool_calls=parse_openai_tool_calls(raw.get("tool_calls")),
        )
    return UserMessage(content=content_text(content))


def parse_openai_tool_calls(raw: JsonValue | None) -> list[ToolCall] | None:
    if raw is None:
        return None
    if not isinstance(raw, list):
        raise TypeError("OpenAI tool_calls must be a list.")
    calls: list[ToolCall] = []
    for item in raw:
        if not isinstance(item, dict):
            continue
        data = json_data(item)
        function = data.get("function")
        if not isinstance(function, dict):
            continue
        function_data = json_data(function)
        calls.append(
            ToolCall(
                id=str(data.get("id") or ""),
                name=str(function_data.get("name") or ""),
                arguments=str(function_data.get("arguments") or "{}"),
            )
        )
    return calls or None


def parse_openai_tools(raw: JsonValue | None) -> list[Tool] | None:
    if raw is None:
        return None
    if not isinstance(raw, list):
        raise TypeError("OpenAI tools must be a list.")
    tools: list[Tool] = []
    for item in raw:
        if not isinstance(item, dict):
            continue
        data = json_data(item)
        function = data.get("function")
        if data.get("type") != "function" or not isinstance(function, dict):
            continue
        function_data = json_data(function)
        tools.append(
            Tool(
                name=str(function_data.get("name") or ""),
                description=str(function_data.get("description") or ""),
                parameters=tool_parameters(function_data.get("parameters")),
                strict=bool_value(function_data.get("strict")),
            )
        )
    return tools or None


def parse_responses_tools(raw: JsonValue | None) -> list[Tool] | None:
    if raw is None:
        return None
    if not isinstance(raw, list):
        raise TypeError("Responses tools must be a list.")
    tools: list[Tool] = []
    for item in raw:
        if not isinstance(item, dict):
            continue
        data = json_data(item)
        tools.append(
            Tool(
                name=str(data.get("name") or ""),
                description=str(data.get("description") or ""),
                parameters=tool_parameters(data.get("parameters")),
                strict=bool_value(data.get("strict")),
            )
        )
    return tools or None


def parse_anthropic_tools(raw: JsonValue | None) -> list[Tool] | None:
    if raw is None:
        return None
    if not isinstance(raw, list):
        raise TypeError("Anthropic tools must be a list.")
    tools: list[Tool] = []
    for item in raw:
        if not isinstance(item, dict):
            continue
        data = json_data(item)
        tools.append(
            Tool(
                name=str(data.get("name") or ""),
                description=str(data.get("description") or ""),
                parameters=tool_parameters(data.get("input_schema")),
            )
        )
    return tools or None


def parse_openai_responses_input(raw: JsonValue | None) -> Messages:
    if isinstance(raw, str):
        return [UserMessage(content=raw)]
    if not isinstance(raw, list):
        raise TypeError("Responses input must be a string or list.")
    messages: Messages = []
    for item in raw:
        if not isinstance(item, dict):
            raise TypeError("Responses input entries must be objects.")
        data = json_data(item)
        item_type = data.get("type")
        if item_type == "function_call":
            call_id = data.get("call_id") or data.get("id")
            name = data.get("name")
            arguments = data.get("arguments")
            if isinstance(call_id, str) and isinstance(name, str):
                messages.append(
                    AssistantMessage(
                        tool_calls=[
                            ToolCall(
                                id=call_id,
                                name=name,
                                arguments=str(arguments or "{}"),
                            )
                        ]
                    )
                )
            continue
        if item_type == "function_call_output":
            call_id = data.get("call_id")
            if isinstance(call_id, str):
                messages.append(
                    ToolMessage(
                        tool_call_id=call_id,
                        content=content_text(data.get("output")),
                    )
                )
            continue
        role = data.get("role")
        content = content_text(data.get("content"))
        if role in {"system", "developer"}:
            messages.append(SystemMessage(content=content))
        elif role == "assistant":
            messages.append(AssistantMessage(content=content))
        else:
            messages.append(UserMessage(content=content))
    return messages


def parse_anthropic_messages(body: JsonData) -> Messages:
    messages: Messages = []
    system = body.get("system")
    if isinstance(system, str) and system:
        messages.append(SystemMessage(content=system))
    raw_messages = body.get("messages")
    if not isinstance(raw_messages, list):
        raise TypeError("Anthropic messages must be a list.")
    for item in raw_messages:
        if not isinstance(item, dict):
            raise TypeError("Anthropic message entries must be objects.")
        data = json_data(item)
        role = data.get("role")
        if role == "assistant":
            messages.append(parse_anthropic_assistant_message(data.get("content")))
        elif role == "user":
            messages.extend(parse_anthropic_user_messages(data.get("content")))
        else:
            raise ValueError(f"Unsupported Anthropic role: {role!r}.")
    return messages


def parse_anthropic_assistant_message(content: JsonValue | None) -> AssistantMessage:
    if isinstance(content, str):
        return AssistantMessage(content=content)
    if not isinstance(content, list):
        return AssistantMessage(content=content_text(content))
    text_parts: list[str] = []
    tool_calls: list[ToolCall] = []
    for block in content:
        if not isinstance(block, dict):
            continue
        data = json_data(block)
        if data.get("type") == "text":
            text_parts.append(content_text(data.get("text")))
        elif data.get("type") == "tool_use":
            tool_id = data.get("id")
            name = data.get("name")
            if isinstance(tool_id, str) and isinstance(name, str):
                tool_calls.append(
                    ToolCall(
                        id=tool_id,
                        name=name,
                        arguments=json.dumps(data.get("input") or {}),
                    )
                )
    return AssistantMessage(
        content="\n".join(text_parts) if text_parts else None,
        tool_calls=tool_calls or None,
    )


def parse_anthropic_user_messages(content: JsonValue | None) -> Messages:
    if isinstance(content, str):
        return [UserMessage(content=content)]
    if not isinstance(content, list):
        return [UserMessage(content=content_text(content))]
    messages: Messages = []
    text_parts: list[str] = []
    for block in content:
        if not isinstance(block, dict):
            continue
        data = json_data(block)
        if data.get("type") == "text":
            text_parts.append(content_text(data.get("text")))
        elif data.get("type") == "tool_result":
            tool_use_id = data.get("tool_use_id")
            if isinstance(tool_use_id, str):
                messages.append(
                    ToolMessage(
                        tool_call_id=tool_use_id,
                        content=content_text(data.get("content")),
                    )
                )
    if text_parts:
        messages.insert(0, UserMessage(content="\n".join(text_parts)))
    return messages


def openai_sampling_args(body: JsonData) -> dict[str, JsonValue]:
    keys = (
        "temperature",
        "top_p",
        "max_tokens",
        "frequency_penalty",
        "presence_penalty",
        "seed",
        "stop",
    )
    return {key: body[key] for key in keys if key in body}


def anthropic_sampling_args(body: JsonData) -> dict[str, JsonValue]:
    keys = ("temperature", "top_p", "top_k", "max_tokens", "stop_sequences")
    return {key: body[key] for key in keys if key in body}


def message_content(content: object) -> JsonValue | None:
    if content is None:
        return None
    return json_value(content, context="message content")


def content_text(content: object) -> str:
    value = message_content(content)
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        text_parts: list[str] = []
        for item in value:
            if isinstance(item, dict):
                data = json_data(item)
                text = data.get("text")
                if isinstance(text, str):
                    text_parts.append(text)
            else:
                text_parts.append(str(item))
        return "\n".join(text_parts)
    return str(content)


def tool_parameters(value: JsonValue | None) -> dict[str, object]:
    if value is None:
        return {"type": "object", "properties": {}}
    if not isinstance(value, dict):
        raise TypeError("Tool parameters must be an object.")
    return {str(key): item for key, item in value.items()}


def string_value(value: JsonValue | None) -> str | None:
    return value if isinstance(value, str) and value else None


def bool_value(value: JsonValue | None) -> bool | None:
    return value if isinstance(value, bool) else None
