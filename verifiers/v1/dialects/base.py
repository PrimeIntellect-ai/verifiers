"""The `Dialect` abstraction: one native wire format, translated to vf for the trace.

A `Dialect[ReqT, RespT]` is the per-format translator the interception server uses to build the
trace from the program's native request + the provider's native response. The server serves
every registered dialect's `routes` (see `dialects.DIALECTS`), so a request's format is resolved
from the endpoint the program's SDK posts to — the harness declares nothing.

The eval client relays a request's bytes verbatim to a matching endpoint, so the dialect only
parses a *copy* for the trace (incl. assembling a relayed SSE stream via `parse_stream`); the
renderer is chat-only. A dialect is therefore mostly wire -> vf
(`parse_request`/`parse_response`/`parse_stream`); the exceptions are `apply_overrides` (impose
the eval's model + sampling in this format's shape) and `extend` (chat-only user-sim injection).
"""

import json
from abc import ABC, abstractmethod
from collections.abc import Mapping
from typing import ClassVar, Generic, TypeVar

from pydantic import BaseModel

from verifiers.v1.types import Messages, Response, SamplingConfig, Tool

ReqT = TypeVar("ReqT")
RespT = TypeVar("RespT", bound=BaseModel)


def iter_sse(raw: bytes) -> list[dict]:
    """Parse a complete SSE byte stream into its JSON data payloads, in order (skipping
    non-JSON sentinels like OpenAI's `data: [DONE]`). Shared by the dialects' `parse_stream`."""
    events: list[dict] = []
    for block in raw.decode("utf-8", errors="replace").split("\n\n"):
        data = "\n".join(
            line.removeprefix("data:").strip()
            for line in block.splitlines()
            if line.startswith("data:")
        )
        if not data or data == "[DONE]":
            continue
        events.append(json.loads(data))
    return events


class Dialect(ABC, Generic[ReqT, RespT]):
    """One native API's wire format, fully typed over its request (`ReqT`) and response
    (`RespT`). The single place a protocol lives: implement a `Dialect` + register it in
    `dialects.DIALECTS` and a harness speaking that format works end-to-end (the eval client and
    interception server are generic over this interface)."""

    routes: ClassVar[tuple[str, ...]]
    """The endpoint path(s) a program's SDK posts model turns to. The interception server serves
    one handler per route, so the wire format is resolved from the route the SDK chose (it
    commits to one when the client is picked) rather than declared by the harness."""

    aux_routes: ClassVar[tuple[str, ...]] = ()
    """Side endpoints the SDK may call that aren't model turns (e.g. Anthropic's
    `count_tokens`): relayed verbatim by the eval client, never recorded on the trace."""

    upstream_path: ClassVar[str]
    """The provider endpoint the proxy forwards to for this format (e.g. `/chat/completions`)."""

    response_type: type[RespT]
    """The native response model — used to validate the provider's raw JSON before parsing."""

    def auth_headers(self, api_key: str) -> dict[str, str]:
        """The provider auth headers for this format. Defaults to OAuth2 Bearer (every
        OpenAI-compatible provider); override for a different scheme (e.g. Anthropic's
        `x-api-key` + `anthropic-version`)."""
        return {"Authorization": f"Bearer {api_key}"}

    def secret(self, headers: Mapping[str, str]) -> str:
        """The per-rollout secret from the request, read from this format's auth carrier
        (default: an `Authorization: Bearer` token; Anthropic uses `x-api-key`)."""
        return headers.get("Authorization", "").removeprefix("Bearer ")

    def streaming(self, body: ReqT) -> bool:
        """Whether the request asks for a streamed (SSE) response."""
        return bool(body.get("stream"))

    def error_body(self, message: str) -> dict:
        """An error payload in this format's error shape (OpenAI by default)."""
        return {"error": {"message": message, "type": "invalid_request_error"}}

    @abstractmethod
    def parse_request(self, body: ReqT) -> tuple[Messages, list[Tool] | None]:
        """The native request -> vf prompt + tools (for the trace)."""

    @abstractmethod
    def parse_response(self, response: RespT) -> Response:
        """A native (non-streamed) response -> the vf `Response` we consume."""

    def validate_response(self, raw: dict) -> RespT:
        """Validate a native response, normalizing provider-compatible extensions if needed."""
        return self.response_type.model_validate(raw)

    @abstractmethod
    def parse_stream(self, raw: bytes) -> Response:
        """A complete native SSE byte stream -> the vf `Response` (assembling the final message
        from the events), to record a relayed stream on the trace."""

    @abstractmethod
    def apply_overrides(self, body: ReqT, model: str, sampling: SamplingConfig) -> ReqT:
        """Return `body` with the eval's `model` + `sampling` imposed in this protocol's shape —
        the only mutation the proxy makes to an otherwise byte-exact forward. Model overlays;
        sampling is authoritative (the program's sampling keys are dropped, the eval's applied)."""

    def extend(self, body: ReqT, completion: dict, user_messages: Messages) -> ReqT:
        """For user-sim multi-turn: return `body` with the model's turn (`completion`, native
        wire) + the simulator's `user_messages` appended, in this protocol's shape. Only the
        chat dialect implements it — the one harness contract with a user simulator speaks chat."""
        raise NotImplementedError(
            f"user simulator is not supported over the {type(self).__name__} dialect"
        )
