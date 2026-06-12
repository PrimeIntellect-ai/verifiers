"""The OpenAI chat-completions dialect.

Translates the OpenAI chat-completions wire format into vf types: requests (`parse_request`)
and responses (`parse_response`). Reasoning extraction mirrors the v0 chat client's
`parse_reasoning_content` — providers expose the model's reasoning under different keys, so
read them in the same precedence (`reasoning` / `reasoning_content` / `reasoning_details`).
"""

from collections.abc import Mapping
from typing import Any

from openai.types.chat import ChatCompletion

from verifiers.v1.dialects.base import Dialect
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

# Sampling knobs the eval owns: stripped from the program's request before the eval's are
# applied, so its sampling is authoritative (see `apply_overrides`). The chat wire names,
# including the `max_tokens` alias the OpenAI SDK also accepts.
_SAMPLING_KEYS = frozenset(
    {"temperature", "top_p", "max_tokens", "max_completion_tokens"}
)

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
# The proxy relays the provider's raw response, so it never serializes; these are for the
# renderer (which builds its generate request and has no raw to relay) and user-sim turn
# injection. Public so the (chat-only) renderer can reuse them.


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


def tool_to_wire(tool: Tool) -> dict:
    function: dict = {
        "name": tool.name,
        "description": tool.description,
        "parameters": tool.parameters,
    }
    if tool.strict is not None:
        function["strict"] = tool.strict
    return {"type": "function", "function": function}


def serialize_completion(response: Response, model: str) -> dict:
    """A vf `Response` -> an OpenAI chat.completion dict the program's SDK expects. The renderer
    sets this on `Response.raw` (it generates, so has no provider response to relay)."""
    message: dict = {"role": "assistant", "content": response.message.content}
    if response.message.reasoning_content is not None:
        message["reasoning_content"] = response.message.reasoning_content
    if response.message.tool_calls:
        message["tool_calls"] = [
            {
                "id": c.id,
                "type": "function",
                "function": {"name": c.name, "arguments": c.arguments},
            }
            for c in response.message.tool_calls
        ]
    usage = (
        {
            "prompt_tokens": response.usage.prompt_tokens,
            "completion_tokens": response.usage.completion_tokens,
            "total_tokens": response.usage.total_tokens,
        }
        if response.usage
        else None
    )
    return {
        "id": response.id or "vf-intercept",
        "object": "chat.completion",
        "created": response.created,
        "model": response.model or model,
        "choices": [
            {
                "index": 0,
                "message": message,
                "finish_reason": response.finish_reason or "stop",
            }
        ],
        "usage": usage,
    }


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


class ChatCompletionsDialect(Dialect[dict, ChatCompletion]):
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

    def apply_overrides(self, body: dict, model: str, sampling: SamplingConfig) -> dict:
        # Forward the program's body verbatim except what the eval owns: model (overlay) and
        # sampling (authoritative — drop the program's sampling keys, then apply the eval's).
        overrides = sampling.model_dump(exclude_none=True)
        steered = {
            k: v
            for k, v in body.items()
            if k not in _SAMPLING_KEYS and k not in overrides
        }
        return {**steered, "model": model, **overrides}

    def extend(self, body: dict, completion: dict, user_messages: Messages) -> dict:
        # Append the model's turn (the verbatim assistant message, so its reasoning survives for
        # the next turn's passback) and the simulator's injected user turn(s) to the wire history.
        messages = [
            *body.get("messages", []),
            completion["choices"][0]["message"],
            *(message_to_wire(m) for m in user_messages),
        ]
        return {**body, "messages": messages}
