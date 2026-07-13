"""The OpenAI chat-completions dialect.

Translates the OpenAI chat-completions wire format into vf types: requests (`parse_request`)
and responses (`parse_response`). Reasoning extraction mirrors the v0 chat client's
`parse_reasoning_content` — providers expose the model's reasoning under different keys, so
read them in the same precedence (`reasoning` / `reasoning_content` / `reasoning_details`).
"""

import time
from collections.abc import Mapping
from dataclasses import dataclass, field as dataclass_field
from typing import Any

from openai.types.chat import ChatCompletion as OpenAIChatCompletion

from verifiers.v1.dialects.base import Dialect, StreamParser, parse_sse_event
from verifiers.v1.types import (
    AssistantMessage,
    FinishReason,
    Message,
    Messages,
    Response,
    SamplingConfig,
    SystemMessage,
    Tool,
    ToolCall,
    ToolMessage,
    Usage,
    UserMessage,
    content_to_parts,
)


class ChatCompletion(OpenAIChatCompletion):
    """The OpenAI SDK closes `service_tier` to a fixed `Literal`, but providers return tiers
    outside it (e.g. Prime's `provisioned`), which makes `model_validate` reject an otherwise
    valid completion. Widen the field to a plain string — we don't consume it — so parsing stays
    lenient about the label instead of dropping it."""

    service_tier: str | None = None


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


def _content_text(content) -> str:
    if isinstance(content, list):
        return "".join(p.get("text", "") for p in content if isinstance(p, dict))
    return content or ""


def parse_message(raw: dict) -> Message:
    """An OpenAI chat request message dict -> a typed Message. User/system bodies keep their
    image parts (multimodal ingress); assistant bodies flatten to text."""
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
        details = raw.get("reasoning_details")
        calls = [
            ToolCall(
                id=c["id"],
                name=c["function"]["name"],
                arguments=c["function"]["arguments"],
            )
            for c in (raw.get("tool_calls") or [])
        ] or None
        return AssistantMessage(
            content=_content_text(content) or None,
            reasoning_content=reasoning_text(raw),
            tool_calls=calls,
            provider_state=details if isinstance(details, list) and details else None,
        )
    return UserMessage(content=content_to_parts(content))


def parse_tools(raw: list[dict] | None) -> list[Tool] | None:
    # `or None` so a tools array with no function entries (e.g. only `custom`/built-in
    # tools) parses to None, not [] — the same contract as the anthropic/responses
    # dialects, and what keeps an empty parse from clearing `Trace.tools`.
    if not raw:
        return None
    return [
        Tool(
            name=t["function"]["name"],
            description=t["function"].get("description", ""),
            parameters=t["function"].get("parameters", {}),
            strict=t["function"].get("strict"),
        )
        for t in raw
        if t.get("type", "function") == "function"
    ] or None


# --- vf -> chat wire ----------------------------------------------------------
# `message_to_wire` (chat-only): used by `extend` (user-sim turn injection), the default harness
# (a Messages prompt), and the train client (its generate request). The proxy preserves its parsed
# native JSON independently and does not use this serializer.


def _content_to_wire(content):
    """Plain text passes through; a content-part list becomes OpenAI wire dicts (so the
    provider / renderer sees the native `image_url` shape)."""
    if isinstance(content, str):
        return content
    return [part.model_dump() for part in content]


def message_to_wire(message: Message) -> dict:
    if message.role == "assistant":
        wire: dict = {"role": "assistant", "content": message.content}
        if message.provider_state:
            wire["reasoning_details"] = message.provider_state
        elif message.reasoning_content is not None:
            wire["reasoning_content"] = message.reasoning_content
        if message.tool_calls:
            wire["tool_calls"] = [
                {
                    "id": call.id,
                    "type": "function",
                    "function": {"name": call.name, "arguments": call.arguments},
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


def response_from_wire(completion: ChatCompletion) -> Response:
    """An OpenAI chat.completion -> a vf `Response` (the one place raw provider objects cross
    into our typed `Response`). No token ids: training tokens come from the renderer client."""
    choice = completion.choices[0]
    message = choice.message
    data = message.model_dump()
    details = data.get("reasoning_details")
    tool_calls = [
        ToolCall(id=tc.id, name=tc.function.name, arguments=tc.function.arguments)
        for tc in (message.tool_calls or [])
    ] or None
    finish: FinishReason = (
        choice.finish_reason if choice.finish_reason in FINISH_REASONS else None
    )
    usage = Usage.from_openai(completion.usage)
    return Response(
        id=completion.id,
        created=completion.created,
        model=completion.model,
        message=AssistantMessage(
            content=message.content,
            reasoning_content=reasoning_text(data),
            tool_calls=tool_calls,
            provider_state=details if isinstance(details, list) and details else None,
        ),
        finish_reason=finish,
        usage=usage,
    )


@dataclass
class ChatStreamParser(StreamParser):
    """Incrementally assemble Chat Completions deltas without retaining SSE bytes."""

    message: dict = dataclass_field(
        default_factory=lambda: {"role": "assistant", "content": None}
    )
    message_parts: dict[str, list[str]] = dataclass_field(
        default_factory=lambda: {key: [] for key in REASONING_FIELDS[:2] + ("content",)}
    )
    tool_calls: dict[int, dict] = dataclass_field(default_factory=dict)
    tool_arguments: dict[int, list[str]] = dataclass_field(default_factory=dict)
    reasoning_details: list[dict] = dataclass_field(default_factory=list)
    reasoning_detail_parts: dict[int, list[str]] = dataclass_field(default_factory=dict)
    finish_reason: str | None = None
    usage: dict | None = None
    head: dict | None = None

    def feed(self, raw: bytes) -> None:
        chunk = parse_sse_event(raw)
        if chunk is None:
            return
        if self.head is None:
            self.head = chunk
        self.usage = chunk.get("usage") or self.usage
        for choice in chunk.get("choices") or []:
            if choice.get("index", 0) != 0:
                continue
            self.finish_reason = choice.get("finish_reason") or self.finish_reason
            delta = choice.get("delta") or {}
            for key in ("content", "reasoning_content", "reasoning"):
                if delta.get(key) is not None:
                    self.message_parts[key].append(delta[key])
            for detail in delta.get("reasoning_details") or []:
                previous = self.reasoning_details[-1] if self.reasoning_details else {}
                if detail.get("type") == previous.get("type") == "reasoning.text":
                    self.reasoning_detail_parts.setdefault(
                        len(self.reasoning_details) - 1,
                        [previous.get("text") or ""],
                    ).append(detail.get("text") or "")
                    for field_name in ("signature", "format"):
                        previous[field_name] = previous.get(field_name) or detail.get(
                            field_name
                        )
                else:
                    self.reasoning_details.append(detail)
            for tool_call in delta.get("tool_calls") or []:
                index = tool_call.get("index", 0)
                slot = self.tool_calls.setdefault(
                    index,
                    {"type": "function", "function": {"name": "", "arguments": ""}},
                )
                slot["id"] = tool_call.get("id") or slot.get("id", "")
                function = tool_call.get("function") or {}
                if function.get("name"):
                    slot["function"]["name"] = function["name"]
                self.tool_arguments.setdefault(index, []).append(
                    function.get("arguments") or ""
                )

    def finish(self) -> Response:
        for key, parts in self.message_parts.items():
            if parts:
                self.message[key] = "".join(parts)
        for index, parts in self.reasoning_detail_parts.items():
            self.reasoning_details[index]["text"] = "".join(parts)
        for index, parts in self.tool_arguments.items():
            self.tool_calls[index]["function"]["arguments"] = "".join(parts)
        if self.tool_calls:
            self.message["tool_calls"] = [
                self.tool_calls[index] for index in sorted(self.tool_calls)
            ]
        if self.reasoning_details:
            self.message["reasoning_details"] = self.reasoning_details
        head = self.head or {}
        return response_from_wire(
            ChatCompletion.model_validate(
                {
                    "id": head.get("id", "vf-intercept"),
                    "object": "chat.completion",
                    "created": head.get("created", int(time.time())),
                    "model": head.get("model", ""),
                    "choices": [
                        {
                            "index": 0,
                            "message": self.message,
                            "finish_reason": self.finish_reason or "stop",
                        }
                    ],
                    "usage": self.usage,
                }
            )
        )


class ChatDialect(Dialect[dict, ChatCompletion]):
    routes = ("/v1/chat/completions",)
    upstream_path = "/chat/completions"
    response_type = ChatCompletion

    def parse_request(self, body: dict) -> tuple[Messages, list[Tool] | None]:
        messages: Messages = []
        tool_names: dict[str, str] = {}
        for raw in body.get("messages", []):
            message = parse_message(raw)
            if isinstance(message, ToolMessage) and message.name is None:
                name = tool_names.get(message.tool_call_id)
                if name is not None:
                    message = message.model_copy(update={"name": name})
            messages.append(message)
            if isinstance(message, AssistantMessage):
                for call in message.tool_calls or []:
                    tool_names[call.id] = call.name
        return messages, parse_tools(body.get("tools"))

    def parse_response(self, response: ChatCompletion) -> Response:
        return response_from_wire(response)

    def stream_parser(self) -> StreamParser:
        return ChatStreamParser()

    def apply_overrides(self, body: dict, model: str, sampling: SamplingConfig) -> dict:
        # Preserve the program's native fields, overlaying only what the eval owns: the model and
        # the sampling knobs it set (later keys win, so the eval's override the program's).
        return {**body, "model": model, **sampling.model_dump(exclude_none=True)}

    def extend(
        self, body: dict, completion: dict | None, user_messages: Messages
    ) -> dict:
        # Append the model's turn (the verbatim assistant message, so its reasoning survives for
        # the next turn's passback) and the simulator's injected user turn(s) to the wire history.
        # A None completion seeds the opening turn (no model message yet) — only the user turn(s).
        messages = [*body.get("messages", [])]
        if completion is not None:
            messages.append(completion["choices"][0]["message"])
        messages.extend(message_to_wire(m) for m in user_messages)
        return {**body, "messages": messages}
