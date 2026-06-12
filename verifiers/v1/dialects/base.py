"""The `Dialect` abstraction: translate one native API's wire format into vf types.

A `Dialect[ReqT, RespT]` is the per-format translator the interception server uses to build
the trace from the program's native request + the provider's native response. It is one-way
(wire -> vf): the proxy relays the provider's raw response to the harness verbatim, so there
is no vf -> wire. Generic over the native request (`ReqT`) and response (`RespT`) types so
each dialect is self-typed.

Each dialect declares the `routes` its native client posts to; the interception server serves
those routes (see `dialects.DIALECTS`), so the wire format is resolved from the endpoint a
request arrives on — the harness declares nothing. `chat_completions` is the OpenAI
chat-completions dialect; OpenAI Responses / Anthropic Messages become new modules here.
"""

from abc import ABC, abstractmethod
from typing import ClassVar, Generic, TypeVar

from pydantic import BaseModel

from verifiers.v1.types import Messages, Response, Tool

ReqT = TypeVar("ReqT")
RespT = TypeVar("RespT", bound=BaseModel)


class Dialect(ABC, Generic[ReqT, RespT]):
    """Translate ONE native API's wire format into vf, fully typed over its native request
    (`ReqT`) and response (`RespT`). One-way (wire -> vf): the proxy relays the raw response to
    the harness verbatim, so there is no vf -> wire."""

    routes: ClassVar[tuple[str, ...]]
    """The endpoint path(s) a program's SDK posts to for this format. The interception server
    serves one handler per route, so the wire format is resolved from the route the SDK chose
    (it commits to one when the client is picked) rather than declared by the harness."""

    response_type: type[RespT]
    """The native response model — used to validate the provider's raw JSON before parsing."""

    @abstractmethod
    def parse_request(self, body: ReqT) -> tuple[Messages, list[Tool] | None]:
        """The native request -> vf prompt + tools (for the trace)."""

    @abstractmethod
    def parse_response(self, response: RespT) -> Response:
        """The native response -> the vf `Response` we consume."""
