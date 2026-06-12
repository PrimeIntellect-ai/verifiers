"""Wire dialects: translate one native API's request/response into vf types.

A `Dialect[ReqT, RespT]` is the per-format translator the interception server uses to build
the trace from the program's native request + the provider's native response. It is one-way
(wire -> vf): the proxy relays the provider's raw response to the harness verbatim, so there
is no vf -> wire. Generic over the native request (`ReqT`) and response (`RespT`) types so
each dialect is self-typed.

A harness declares which dialect it speaks (`Harness.DIALECT`) — there is no auto-detection
(a follow-up, for harnesses that support several native clients). `ChatCompletionsDialect` is
the only dialect today; OpenAI Responses / Anthropic Messages become new `Dialect`s.
"""

from abc import ABC, abstractmethod
from typing import Generic, TypeVar

from openai.types.chat import ChatCompletion
from pydantic import BaseModel

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

ReqT = TypeVar("ReqT")
RespT = TypeVar("RespT", bound=BaseModel)


class Dialect(ABC, Generic[ReqT, RespT]):
    """Translate ONE native API's wire format into vf, fully typed over its native request
    (`ReqT`) and response (`RespT`). One-way (wire -> vf): the proxy relays the raw response to
    the harness verbatim, so there is no vf -> wire."""

    response_type: type[RespT]
    """The native response model — used to validate the provider's raw JSON before parsing."""

    @abstractmethod
    def parse_request(self, body: ReqT) -> tuple[Messages, list[Tool] | None]:
        """The native request -> vf prompt + tools (for the trace)."""

    @abstractmethod
    def parse_response(self, response: RespT) -> Response:
        """The native response -> the vf `Response` we consume."""


# --- chat completions ---------------------------------------------------------


def _content_text(content) -> str:
    """Flatten content to text — for roles that never carry images (assistant, tool)."""
    if isinstance(content, list):
        return "".join(p.get("text", "") for p in content if isinstance(p, dict))
    return content or ""


def parse_message(raw: dict) -> Message:
    """An OpenAI chat request message dict -> a typed Message. User/system bodies keep their
    image parts (multimodal ingress); assistant/tool bodies flatten to text. Providers name
    reasoning differently — `reasoning_content` (DeepSeek/vLLM) or `reasoning` (OpenRouter-
    style); accept either."""
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
            reasoning_content=raw.get("reasoning_content") or raw.get("reasoning"),
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
    into our typed `Response`). Providers name reasoning `reasoning_content` (DeepSeek/vLLM) or
    `reasoning` (OpenRouter-style); accept either."""
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
            reasoning_content=getattr(message, "reasoning_content", None)
            or getattr(message, "reasoning", None),
            tool_calls=tool_calls,
        ),
        finish_reason=finish,
        usage=usage,
        tokens=_tokens_from_wire(completion, choice),
    )


class ChatCompletionsDialect(Dialect[dict, ChatCompletion]):
    """The OpenAI chat-completions wire format (the only dialect today)."""

    response_type = ChatCompletion

    def parse_request(self, body: dict) -> tuple[Messages, list[Tool] | None]:
        return [parse_message(m) for m in body.get("messages", [])], parse_tools(
            body.get("tools")
        )

    def parse_response(self, response: ChatCompletion) -> Response:
        return response_from_wire(response)
