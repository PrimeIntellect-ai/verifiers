"""The OpenAI chat-completions dialect.

Translates the OpenAI chat-completions wire format into vf types: requests (`parse_request`)
and responses (`parse_response`). Reasoning extraction mirrors the v0 chat client's
`parse_reasoning_content` — providers expose the model's reasoning under different keys, so
read them in the same precedence (`reasoning` / `reasoning_content` / `reasoning_details`).
"""

import json
from collections.abc import Mapping
from functools import partial
from typing import Any

from openai.types import CompletionUsage
from openai.types.chat import ChatCompletionChunk

from verifiers.v1.dialects.base import Dialect, RawRequest, Setter, patch_content
from verifiers.v1.types import (
    AssistantMessage,
    FinishReason,
    Message,
    Messages,
    NativeContentPart,
    Request,
    Response,
    SystemMessage,
    Tool,
    ToolCall,
    ToolMessage,
    Usage,
    UserMessage,
    content_to_parts,
)
from verifiers.v1.utils.chat import read_chat_completion

FINISH_REASONS = frozenset({"stop", "length", "tool_calls"})

# Providers name the model's reasoning differently; read them in the v0 client's precedence.
# `reasoning` (vLLM / Together / OpenRouter), `reasoning_content` (DeepSeek / Qwen / SGLang /
# Fireworks / Kimi), `reasoning_details` (OpenRouter / MiniMax).
REASONING_FIELDS = ("reasoning", "reasoning_content", "reasoning_details")


def reasoning_text(data: Mapping[str, Any]) -> str | None:
    """The model's reasoning string, from whichever field the provider used."""
    for field in REASONING_FIELDS:
        value = data.get(field)
        if isinstance(value, str) and value:
            return value
    details = data.get("reasoning_details")
    if isinstance(details, list):
        parts = []
        for detail in details:
            if not isinstance(detail, Mapping):
                continue
            value = detail.get("text") or detail.get("summary")
            if isinstance(value, str) and value:
                parts.append(value)
        return "\n".join(parts) or None
    return None


def parse_assistant(raw: dict) -> AssistantMessage:
    """An OpenAI assistant message dict -> a typed assistant message; its body flattens to
    text."""
    content = raw.get("content")
    details = raw.get("reasoning_details")
    text = (
        "".join(part.get("text", "") for part in content if isinstance(part, dict))
        if isinstance(content, list)
        else content or ""
    )
    calls = []
    for call in raw.get("tool_calls") or []:
        kind = call.get("type") or (
            "custom" if call.get("custom") is not None else "function"
        )
        native = call[kind]
        calls.append(
            ToolCall(
                id=call["id"],
                type=kind,
                name=native["name"],
                namespace=native.get("namespace"),
                arguments=native["input" if kind == "custom" else "arguments"],
            )
        )
    return AssistantMessage(
        content=text or None,
        reasoning_content=reasoning_text(raw),
        tool_calls=calls or None,
        provider_state=details if isinstance(details, list) and details else None,
    )


def parse_message(raw: dict) -> Message:
    """An OpenAI chat request message dict -> a typed Message. User/system bodies keep their
    content parts (multimodal ingress); assistant bodies flatten to text."""
    role = raw.get("role")
    content = raw.get("content")
    if role == "system":
        return SystemMessage(content=content_to_parts(content))
    if role == "tool":
        return ToolMessage(
            tool_call_id=raw.get("tool_call_id", ""),
            content=content_to_parts(content),
            name=raw.get("name"),
        )
    if role == "assistant":
        return parse_assistant(raw)
    return UserMessage(content=content_to_parts(content))


def parse_tools(raw: list[dict] | None) -> list[Tool] | None:
    tools = []
    for declaration in raw or []:
        kind = declaration.get("type", "function")
        tool = declaration.get(kind, declaration)
        if kind == "mcp":
            tool = {
                key: value
                for key, value in tool.items()
                if key.lower() not in ("authorization", "headers")
            }
        tools.append(
            Tool.model_validate(
                tool
                | {
                    "type": kind,
                    "name": tool.get("name") or tool.get("server_label") or kind,
                }
            )
        )
    return tools or None


# --- vf -> chat wire ----------------------------------------------------------
# `message_to_wire` (chat-only): used by the bash harness (a Messages prompt) and the train
# client (its generate request). The proxy preserves its parsed native JSON independently and
# does not use this serializer.


def _content_to_wire(content):
    """Plain text passes through; a content-part list becomes OpenAI wire dicts (so the
    provider / renderer sees the native `image_url` shape)."""
    if isinstance(content, str):
        return content
    return [
        part.native if isinstance(part, NativeContentPart) else part.model_dump()
        for part in content
    ]


def message_to_wire(message: Message) -> dict:
    if message.role == "assistant":
        # Strict providers reject `content: null` without tool calls.
        content = message.content
        if content is None and not message.tool_calls:
            content = ""
        wire: dict = {"role": "assistant", "content": content}
        if message.provider_state:
            wire["reasoning_details"] = message.provider_state
        elif message.reasoning_content is not None:
            wire["reasoning_content"] = message.reasoning_content
        if message.tool_calls:
            wire["tool_calls"] = [
                {
                    "id": call.id,
                    "type": call.type,
                    call.type: {
                        "name": call.name,
                        **({"namespace": call.namespace} if call.namespace else {}),
                        "input"
                        if call.type == "custom"
                        else "arguments": call.arguments,
                    },
                }
                for call in message.tool_calls
            ]
        return wire
    if message.role == "tool":
        wire = {
            "role": "tool",
            "tool_call_id": message.tool_call_id,
            "content": _content_to_wire(message.content),
        }
        if message.name:
            wire["name"] = message.name
        return wire
    return {"role": message.role, "content": _content_to_wire(message.content)}


def _write_message(native: dict, message: Message) -> None:
    native["content"] = patch_content(
        message.content, _content_to_wire(message.content)
    )
    if isinstance(message, ToolMessage):
        if message.name is None:
            native.pop("name", None)
        else:
            native["name"] = message.name


def response_from_wire(completion: dict) -> Response:
    """An OpenAI chat.completion -> a vf `Response` (the one place raw provider objects cross
    into our typed `Response`). No token ids: training tokens come from the renderer client."""
    choice = completion["choices"][0]
    finish_reason = choice.get("finish_reason")
    finish: FinishReason = finish_reason if finish_reason in FINISH_REASONS else None
    usage = completion.get("usage")
    return Response(
        id=completion.get("id") or "",
        created=completion.get("created") or 0,
        model=completion.get("model") or "",
        message=parse_assistant(choice["message"]),
        finish_reason=finish,
        # The SDK's lenient model carries provider extensions such as `cost`.
        usage=Usage.from_openai(CompletionUsage.construct(**usage)) if usage else None,
    )


class ChatDialect(Dialect):
    sampling_fields = frozenset(
        {
            "temperature",
            "top_p",
            "top_k",
            "min_p",
            "max_tokens",
            "max_completion_tokens",
            "reasoning_effort",
            "seed",
            "stop",
            "n",
            "logprobs",
            "top_logprobs",
            "logit_bias",
            "frequency_penalty",
            "presence_penalty",
            "repetition_penalty",
            "response_format",
            "service_tier",
            "tool_choice",
            "parallel_tool_calls",
            "extra_body",
        }
    )
    max_tokens_keys = ("max_tokens", "max_completion_tokens")
    effort_path = ("reasoning_effort",)
    routes = ("/v1/chat/completions",)
    upstream_path = "/chat/completions"
    event_type = ChatCompletionChunk

    def parse_request(self, body: RawRequest) -> tuple[Request, list[Setter | None]]:
        if body.get("n", 1) != 1:
            raise ValueError("chat completions require n=1")
        messages: Messages = []
        setters: list[Setter | None] = []
        tool_names: dict[str, str] = {}
        for raw in body.get("messages", []):
            message = parse_message(raw)
            if isinstance(message, ToolMessage) and message.name is None:
                name = tool_names.get(message.tool_call_id)
                if name is not None:
                    message = message.model_copy(update={"name": name})
            messages.append(message)
            setters.append(
                partial(_write_message, raw)
                if isinstance(message, (UserMessage, ToolMessage))
                else None
            )
            if isinstance(message, AssistantMessage):
                for call in message.tool_calls or []:
                    tool_names[call.id] = call.name
        return Request(messages=messages, tools=parse_tools(body.get("tools"))), setters

    def parse_response(self, response: dict) -> Response:
        return response_from_wire(response)

    def rewrite_response(self, raw: dict, text: str) -> None:
        for choice in raw.get("choices") or []:
            if isinstance(choice.get("message"), dict):
                choice["message"] = {"role": "assistant", "content": text}
                choice["finish_reason"] = "stop"
                choice.pop("logprobs", None)

    def stream_events(self, raw: dict) -> list[bytes]:
        choice = (raw.get("choices") or [{}])[0]
        message = dict(choice.get("message") or {"role": "assistant", "content": ""})
        if message.get("tool_calls"):
            message["tool_calls"] = [
                {**call, "index": index}
                for index, call in enumerate(message["tool_calls"])
            ]
        chunk = {
            **{key: value for key, value in raw.items() if key != "choices"},
            "object": "chat.completion.chunk",
            "choices": [
                {
                    "index": 0,
                    "delta": message,
                    "finish_reason": choice.get("finish_reason"),
                }
            ],
        }
        return [f"data: {json.dumps(chunk)}\n\n".encode(), b"data: [DONE]\n\n"]

    async def read_stream(self, stream) -> dict:
        return (await read_chat_completion(stream)).to_dict()
