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
from collections.abc import Iterable
from typing import ClassVar, Generic, TypeVar

from pydantic import BaseModel

from verifiers.v1.types import Messages, Response, SamplingConfig, Tool

ReqT = TypeVar("ReqT")
RespT = TypeVar("RespT", bound=BaseModel)


class Dialect(ABC, Generic[ReqT, RespT]):
    """One native API's wire format, fully typed over its request (`ReqT`) and response
    (`RespT`). The single place a protocol lives: implement a `Dialect` + register it in
    `dialects.DIALECTS` and a harness speaking that format works end-to-end (the proxy and
    interception server are generic over this interface). Mostly one-way (wire -> vf, to build
    the trace) — the only vf -> wire is `serialize_response`/`extend`, needed where there's no
    raw provider response to relay (the renderer, and user-sim turn injection)."""

    routes: ClassVar[tuple[str, ...]]
    """The endpoint path(s) a program's SDK posts to for this format. The interception server
    serves one handler per route, so the wire format is resolved from the route the SDK chose
    (it commits to one when the client is picked) rather than declared by the harness."""

    upstream_path: ClassVar[str]
    """The provider endpoint the proxy forwards to for this format (e.g. `/chat/completions`)."""

    streams: ClassVar[bool] = False
    """Whether this format's clients require `stream: true` and so must be fake-streamed: the
    proxy still fetches the whole completion unary (the dialect forces streaming off upstream in
    `apply_overrides`), and the interception server replays it as the SSE events `stream_events`
    yields. Off by default (chat completions returns a JSON body); the Responses dialect sets it
    (codex only speaks streaming Responses)."""

    response_type: type[RespT]
    """The native response model — used to validate the provider's raw JSON before parsing."""

    def auth_headers(self, api_key: str) -> dict[str, str]:
        """The provider auth headers for this format. Defaults to OAuth2 Bearer (every
        OpenAI-compatible provider); override for a different scheme (e.g. Anthropic's
        `x-api-key` + `anthropic-version`)."""
        return {"Authorization": f"Bearer {api_key}"}

    @abstractmethod
    def parse_request(self, body: ReqT) -> tuple[Messages, list[Tool] | None]:
        """The native request -> vf prompt + tools (for the trace)."""

    @abstractmethod
    def parse_response(self, response: RespT) -> Response:
        """The native response -> the vf `Response` we consume."""

    @abstractmethod
    def apply_overrides(self, body: ReqT, model: str, sampling: SamplingConfig) -> ReqT:
        """Return `body` with the eval's `model` + `sampling` imposed in this protocol's shape —
        the only mutation the proxy makes to an otherwise byte-exact forward. Model overlays;
        sampling is authoritative (the program's sampling keys are dropped, the eval's applied)."""

    @abstractmethod
    def serialize_response(self, response: Response, model: str) -> dict:
        """A vf `Response` -> a native wire response dict for the program — used only when there
        is no raw provider response to relay 1:1 (the renderer generates one)."""

    @abstractmethod
    def extend(self, body: ReqT, completion: dict, user_messages: Messages) -> ReqT:
        """For user-sim multi-turn: return `body` with the model's turn (`completion`, native
        wire) and the simulator's `user_messages` appended to the conversation, in this
        protocol's shape — so a multi-turn exchange plays out within one program request."""

    def stream_events(self, completion: dict) -> Iterable[dict]:
        """Fake-streaming (only for dialects with `streams = True`): given the full buffered wire
        response `completion`, yield the ordered SSE event objects (each a dict with a `type`)
        that replay it for a client that asked for `stream: true`. The interception server
        serializes each as one `event:`/`data:` SSE frame. We hold the whole response already, so
        this is a straight replay — no partial deltas to reassemble."""
        raise NotImplementedError(f"{type(self).__name__} does not support streaming")
