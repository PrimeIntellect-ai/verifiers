from verifiers.v1.dialects.anthropic import AnthropicDialect
from verifiers.v1.dialects.base import Dialect
from verifiers.v1.dialects.chat import (
    FINISH_REASONS,
    ChatDialect,
    parse_message,
    parse_tools,
)
from verifiers.v1.dialects.responses import ResponsesDialect

DIALECTS: tuple[Dialect, ...] = (ChatDialect(), ResponsesDialect(), AnthropicDialect())
"""The registered dialects, all served simultaneously by the interception server, which resolves
the wire format from the route a request arrived on."""

__all__ = [
    "DIALECTS",
    "FINISH_REASONS",
    "AnthropicDialect",
    "ChatDialect",
    "Dialect",
    "ResponsesDialect",
    "parse_message",
    "parse_tools",
]
