"""The `Dialect` abstraction: translate one native API's wire format into vf types.

A `Dialect[ReqT, RespT]` is the per-format translator the interception server uses to build
the trace from the program's native request + the provider's native response. It is one-way
(wire -> vf): the proxy relays the provider's raw response to the harness verbatim, so there
is no vf -> wire. Generic over the native request (`ReqT`) and response (`RespT`) types so
each dialect is self-typed.

A harness declares which dialect it speaks (`Harness.DIALECT`) — there is no auto-detection
(a follow-up, for harnesses that support several native clients). Today the only dialect is
`oai_chat_completions`; OpenAI Responses / Anthropic Messages become new modules here.
"""

from abc import ABC, abstractmethod
from typing import Generic, TypeVar

from pydantic import BaseModel

from verifiers.v1.types import Messages, Response, Tool

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
