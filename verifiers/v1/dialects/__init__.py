"""Wire dialects: per-native-format translators (wire -> vf) for the interception trace.

`Dialect` is the abstraction; one module per native format (`chat`, `responses`). The
interception server serves each registered dialect's `routes`, so the wire format is resolved
from the endpoint the program's SDK posts to — no per-harness declaration.
"""

from verifiers.v1.dialects.base import Dialect, iter_sse
from verifiers.v1.dialects.chat import (
    FINISH_REASONS,
    ChatDialect,
    parse_message,
    parse_tools,
    response_from_wire,
)
from verifiers.v1.dialects.responses import ResponsesDialect

DIALECTS: tuple[Dialect, ...] = (ChatDialect(), ResponsesDialect())
"""The registered dialects, all served simultaneously by the interception server, which resolves
the wire format from the route a request arrived on."""

__all__ = [
    "Dialect",
    "DIALECTS",
    "FINISH_REASONS",
    "ChatDialect",
    "ResponsesDialect",
    "iter_sse",
    "parse_message",
    "parse_tools",
    "response_from_wire",
]
