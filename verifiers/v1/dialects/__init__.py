"""Wire dialects: the native API formats the interception server understands.

One `Dialect` per native format, each owning its route, auth carrier, and wire <-> vf
codec. The interception server serves every registered dialect's routes, so a request's
format is resolved from the endpoint the program's SDK posts to — the harness declares
nothing, and a program may mix formats within one rollout.
"""

from verifiers.v1.dialects.anthropic import AnthropicDialect
from verifiers.v1.dialects.base import Dialect, iter_sse, sse
from verifiers.v1.dialects.chat import (
    ChatDialect,
    parse_message,
    parse_tools,
    serialize_completion,
)
from verifiers.v1.dialects.responses import ResponsesDialect

DIALECTS: tuple[Dialect, ...] = (ChatDialect(), AnthropicDialect(), ResponsesDialect())
"""The registered dialects, all served simultaneously by the interception server."""

__all__ = [
    "Dialect",
    "DIALECTS",
    "ChatDialect",
    "AnthropicDialect",
    "ResponsesDialect",
    "iter_sse",
    "sse",
    "parse_message",
    "parse_tools",
    "serialize_completion",
]
