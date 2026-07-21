from verifiers.v1.dialects.anthropic import AnthropicDialect
from verifiers.v1.dialects.base import Dialect, StreamParser
from verifiers.v1.dialects.chat import (
    FINISH_REASONS,
    ChatDialect,
    parse_message,
    parse_tools,
    response_from_wire,
)
from verifiers.v1.dialects.responses import ResponsesDialect

DIALECTS: tuple[Dialect, ...] = (ChatDialect(), ResponsesDialect(), AnthropicDialect())
"""The registered dialects, all served simultaneously by the interception server, which resolves
the wire format from the route a request arrived on."""

__all__ = [
    "Dialect",
    "DIALECTS",
    "FINISH_REASONS",
    "AnthropicDialect",
    "ChatDialect",
    "ResponsesDialect",
    "StreamParser",
    "parse_message",
    "parse_tools",
    "response_from_wire",
]
