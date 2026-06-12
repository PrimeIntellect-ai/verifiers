"""Wire dialects: per-native-format translators (wire -> vf) for the interception trace.

`Dialect` is the abstraction; `chat_completions` is the OpenAI chat-completions dialect and
`responses` the OpenAI Responses dialect (Anthropic Messages becomes another module). The
interception server serves each registered dialect's `routes`, so the wire format is resolved
from the endpoint the program's SDK posts to — no per-harness declaration.
"""

from verifiers.v1.dialects.base import Dialect
from verifiers.v1.dialects.chat_completions import (
    FINISH_REASONS,
    ChatCompletionsDialect,
    parse_message,
    parse_tools,
    response_from_wire,
)
from verifiers.v1.dialects.responses import ResponsesDialect

DIALECTS: tuple[Dialect, ...] = (ChatCompletionsDialect(), ResponsesDialect())
"""The registered dialects. The interception server serves each one's `routes` and resolves the
wire format from the route the request arrived on (`/v1/chat/completions` -> chat completions,
`/v1/responses` -> responses)."""

__all__ = [
    "Dialect",
    "DIALECTS",
    "FINISH_REASONS",
    "ChatCompletionsDialect",
    "ResponsesDialect",
    "parse_message",
    "parse_tools",
    "response_from_wire",
]
