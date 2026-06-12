"""Wire dialects: per-native-format translators (wire -> vf) for the interception trace.

`Dialect` is the abstraction; `oai_chat_completions` is the only dialect today (OpenAI Responses
/ Anthropic Messages become new modules). A harness declares its dialect via `Harness.DIALECT`.
"""

from verifiers.v1.clients.dialects.base import Dialect
from verifiers.v1.clients.dialects.oai_chat_completions import (
    FINISH_REASONS,
    ChatCompletionsDialect,
    parse_message,
    parse_tools,
    response_from_wire,
)

__all__ = [
    "Dialect",
    "FINISH_REASONS",
    "ChatCompletionsDialect",
    "parse_message",
    "parse_tools",
    "response_from_wire",
]
