"""The OpenAI chat-completions dialect.

Translates the OpenAI chat-completions wire format into vf types: requests (`parse_request`)
and responses (`parse_response`). Reasoning extraction mirrors the v0 chat client's
`parse_reasoning_content` — providers expose the model's reasoning under different keys, so
read them in the same precedence (`reasoning` / `reasoning_content` / `reasoning_details`).
"""

import time
from collections.abc import Mapping
from typing import Any

from openai.types.chat import ChatCompletion

from verifiers.v1.dialects.base import Dialect, iter_sse
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

FINISH_REASONS = frozenset({"stop", "length", "tool_calls"})

# Providers name the model's reasoning differently; read them in the v0 client's precedence.
# `reasoning` (vLLM / Together / OpenRouter), `reasoning_content` (DeepSeek / Qwen / SGLang /
# Fireworks / Kimi), `reasoning_details` (OpenRouter / MiniMax — usually structured, kept here
# for the rare provider that returns it as a plain string).
REASONING_FIELDS = ("reasoning", "reasoning_content", "reasoning_details")


def reasoning_text(data: Mapping[str, Any]) -> str | None:
    """The model's reasoning string, from whichever field the provider used."""
    for field in REASONING_FIELDS:
        value = data.get(field)
        if isinstance(value, str) and value:
            return value
    return None


def _content_text(content) -> str:
    """Flatten content to text — for roles that never carry images (assistant, tool)."""
    if isinstance(content, list):
        return "".join(p.get("text", "") for p in content if isinstance(p, dict))
    return content or ""


def parse_message(raw: dict) -> Message:
    """An OpenAI chat request message dict -> a typed Message. User/system bodies keep their
    image parts (multimodal ingress); assistant/tool bodies flatten to text."""
    role = raw.get("role")
    content = raw.get("content")
    if role == "system":
        return SystemMessage(content=content_to_parts(content))
    if role == "tool":
        return ToolMessage(
            tool_call_id=raw.get("tool_call_id", ""), content=_content_text(content)
        )
    if role == "assistant":
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
        )
    return UserMessage(content=content_to_parts(content))


def parse_tools(raw: list[dict] | None) -> list[Tool] | None:
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
    ]


# --- vf -> chat wire ----------------------------------------------------------
# `message_to_wire` (chat-only): used by `extend` (user-sim turn injection), the default harness
# (a Messages instruction), and the train client (its generate request). The proxy never
# serializes — it relays the provider's raw bytes.


def _content_to_wire(content):
    """Plain text passes through; a content-part list becomes OpenAI wire dicts (so the
    provider / renderer sees the native `image_url` shape)."""
    if isinstance(content, str):
        return content
    return [part.model_dump() for part in content]


def message_to_wire(message: Message) -> dict:
    """A vf message -> the OpenAI chat wire dict."""
    if message.role == "assistant":
        wire: dict = {"role": "assistant", "content": message.content}
        # Reasoning models (DeepSeek V4, Kimi K2 Thinking, ...) require the prior turns'
        # `reasoning_content` sent back as a message-level field; carry it when present.
        if message.reasoning_content is not None:
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
        return {
            "role": "tool",
            "tool_call_id": message.tool_call_id,
            "content": message.content,
        }
    return {"role": message.role, "content": _content_to_wire(message.content)}


def response_from_wire(completion: ChatCompletion) -> Response:
    """An OpenAI chat.completion -> a vf `Response` (the one place raw provider objects cross
    into our typed `Response`). No token ids: training tokens come from the renderer client."""
    choice = completion.choices[0]
    message = choice.message
    tool_calls = [
        ToolCall(id=tc.id, name=tc.function.name, arguments=tc.function.arguments)
        for tc in (message.tool_calls or [])
    ] or None
    finish: FinishReason = (
        choice.finish_reason if choice.finish_reason in FINISH_REASONS else None
    )
    usage = (
        Usage(
            prompt_tokens=completion.usage.prompt_tokens,
            completion_tokens=completion.usage.completion_tokens,
        )
        if completion.usage
        else None
    )
    return Response(
        id=completion.id,
        created=completion.created,
        model=completion.model,
        message=AssistantMessage(
            content=message.content,
            reasoning_content=reasoning_text(message.model_dump()),
            tool_calls=tool_calls,
        ),
        finish_reason=finish,
        usage=usage,
    )


class ChatDialect(Dialect[dict, ChatCompletion]):
    """The OpenAI chat-completions wire format."""

    routes = ("/v1/chat/completions",)
    upstream_path = "/chat/completions"
    response_type = ChatCompletion

    def parse_request(self, body: dict) -> tuple[Messages, list[Tool] | None]:
        return [parse_message(m) for m in body.get("messages", [])], parse_tools(
            body.get("tools")
        )

    def parse_response(self, response: ChatCompletion) -> Response:
        return response_from_wire(response)

    def parse_stream(self, raw: bytes) -> Response:
        """Assemble `chat.completion.chunk` deltas into one completion, then parse it."""
        chunks = iter_sse(raw)
        message: dict = {"role": "assistant", "content": None}
        tool_calls: dict[int, dict] = {}
        finish_reason = None
        usage = None
        for chunk in chunks:
            usage = chunk.get("usage") or usage
            for choice in chunk.get("choices") or []:
                if choice.get("index", 0) != 0:
                    continue
                finish_reason = choice.get("finish_reason") or finish_reason
                delta = choice.get("delta") or {}
                for key in ("content", "reasoning_content", "reasoning"):
                    if delta.get(key) is not None:
                        message[key] = (message.get(key) or "") + delta[key]
                for tc in delta.get("tool_calls") or []:
                    slot = tool_calls.setdefault(
                        tc.get("index", 0),
                        {"type": "function", "function": {"name": "", "arguments": ""}},
                    )
                    slot["id"] = tc.get("id") or slot.get("id", "")
                    fn = tc.get("function") or {}
                    if fn.get("name"):
                        slot["function"]["name"] = fn["name"]
                    slot["function"]["arguments"] += fn.get("arguments") or ""
        if tool_calls:
            message["tool_calls"] = [tool_calls[i] for i in sorted(tool_calls)]
        head = chunks[0] if chunks else {}
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
                            "message": message,
                            "finish_reason": finish_reason or "stop",
                        }
                    ],
                    "usage": usage,
                }
            )
        )

    def apply_overrides(self, body: dict, model: str, sampling: SamplingConfig) -> dict:
        # Forward the program's body verbatim, overlaying only what the eval owns: the model and
        # the sampling knobs it set (later keys win, so the eval's override the program's).
        return {**body, "model": model, **sampling.model_dump(exclude_none=True)}

    def extend(self, body: dict, completion: dict, user_messages: Messages) -> dict:
        # Append the model's turn (the verbatim assistant message, so its reasoning survives for
        # the next turn's passback) and the simulator's injected user turn(s) to the wire history.
        messages = [
            *body.get("messages", []),
            completion["choices"][0]["message"],
            *(message_to_wire(m) for m in user_messages),
        ]
        return {**body, "messages": messages}
