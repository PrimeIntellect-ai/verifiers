"""The OpenAI chat-completions dialect (the only dialect today).

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
    SystemMessage,
    Tool,
    ToolCall,
    ToolMessage,
    TurnTokens,
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


def _tokens_from_wire(completion: ChatCompletion, choice) -> TurnTokens | None:
    """Parse vLLM's token ids + sampling logprobs into `TurnTokens` (best-effort): vLLM
    surfaces the completion ids on the choice (`return_token_ids`), the prompt ids on the
    completion, and sampled logprobs as one `logprobs.content` entry per generated token."""
    completion_ids = getattr(choice, "token_ids", None)
    if not completion_ids:
        return None
    content = choice.logprobs.content if choice.logprobs else None
    return TurnTokens(
        prompt_ids=list(getattr(completion, "prompt_token_ids", None) or []),
        completion_ids=list(completion_ids),
        completion_logprobs=[lp.logprob for lp in content] if content else [],
    )


def response_from_wire(completion: ChatCompletion) -> Response:
    """An OpenAI chat.completion -> a vf `Response` (the one place raw provider objects cross
    into our typed `Response`)."""
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
        tokens=_tokens_from_wire(completion, choice),
    )


class ChatCompletionsDialect(Dialect[dict, ChatCompletion]):
    """The OpenAI chat-completions wire format (the only dialect today)."""

    routes = ("/v1/chat/completions",)
    response_type = ChatCompletion

    def parse_request(self, body: dict) -> tuple[Messages, list[Tool] | None]:
        return [parse_message(m) for m in body.get("messages", [])], parse_tools(
            body.get("tools")
        )

    def parse_response(self, response: ChatCompletion) -> Response:
        return response_from_wire(response)
