"""The `Dialect` abstraction: one native wire format, translated both ways.

A `Dialect` is one native API's wire format (OpenAI chat completions, Anthropic
Messages, OpenAI Responses). The interception server serves every registered
dialect's `route` (see `dialects.DIALECTS`), so a request's format is resolved from
the endpoint the program's SDK posts to — the harness declares nothing.

Each dialect owns the full codec for its format:

- wire -> vf (`parse_request` / `parse_response` / `parse_stream`): always used, to
  build the trace and drive limits/stops/user-sim.
- vf -> wire (`serialize_response` / `serialize_stream`): used only on the translate
  path, when the rollout's client speaks a *different* protocol (e.g. training via
  the renderer) and the typed `Response` must be handed back to the program in its
  native format.

When the client natively speaks the dialect (`Client.dialect == Dialect.name`) the
server relays the request bytes verbatim instead — no vf -> wire reconstruction
touches the load-bearing bytes (see `interception.server`).
"""

import json
from abc import ABC, abstractmethod
from collections.abc import Mapping
from typing import ClassVar

from verifiers.v1.types import Messages, Response, Tool


def iter_sse(raw: bytes) -> list[dict]:
    """Parse a complete SSE byte stream into its JSON data payloads, in order.
    Skips non-JSON sentinels (OpenAI's `data: [DONE]`). Shared by the dialects'
    `parse_stream` implementations."""
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


def sse(payload: dict, event: str | None = None) -> bytes:
    """One SSE frame for `payload` (with an `event:` name when the format uses one)."""
    head = f"event: {event}\n" if event else ""
    return f"{head}data: {json.dumps(payload)}\n\n".encode()


class Dialect(ABC):
    """One native API's wire format: routes, auth carrier, and the wire <-> vf codec."""

    name: ClassVar[str]
    """The protocol id, matched against `Client.dialect` to pick relay vs translate."""
    route: ClassVar[str]
    """The path a program's SDK posts model turns to (e.g. `/v1/chat/completions`)."""
    aux_routes: ClassVar[tuple[str, ...]] = ()
    """Side endpoints the SDK may call that are not model turns (e.g. Anthropic's
    `count_tokens`): relayed verbatim when the client speaks this dialect, answered
    with `handle_aux` otherwise. Never recorded on the trace."""

    def secret(self, headers: Mapping[str, str]) -> str:
        """The per-rollout secret from the request, read from this format's auth carrier
        (default: an `Authorization: Bearer` token)."""
        return headers.get("Authorization", "").removeprefix("Bearer ")

    def streaming(self, body: dict) -> bool:
        """Whether the request asks for a streamed (SSE) response."""
        return bool(body.get("stream"))

    def error_body(self, message: str) -> dict:
        """An error payload in this format's error shape."""
        return {"error": {"message": message, "type": "invalid_request_error"}}

    @abstractmethod
    def parse_request(self, body: dict) -> tuple[Messages, list[Tool] | None]:
        """The native request -> the typed prompt + tools (for the trace and the
        translate path)."""

    @abstractmethod
    def parse_response(self, raw: dict) -> Response:
        """A native (non-streamed) response -> the typed `Response`."""

    @abstractmethod
    def parse_stream(self, raw: bytes) -> Response:
        """A complete native SSE byte stream -> the typed `Response` (assembles the
        final message from the events; used to record a relayed stream on the trace)."""

    @abstractmethod
    def serialize_response(self, response: Response, model: str) -> dict:
        """The typed `Response` -> this format's response payload (translate path)."""

    @abstractmethod
    def serialize_stream(self, response: Response, model: str) -> bytes:
        """The typed `Response` -> a minimal valid SSE byte stream in this format
        (translate path, for a program that requested streaming)."""

    def handle_aux(self, path: str, body: dict) -> dict:
        """Answer an aux route locally (translate path only — when the client speaks the
        dialect, aux requests are relayed verbatim instead). Default: no aux routes."""
        raise NotImplementedError

    def extend_request(
        self, body: dict, completion: dict, user_messages: Messages
    ) -> dict:
        """Extend a relayed request body with the last completion's assistant message and
        a user simulator's injected turn(s). Only the chat dialect implements this — the
        only harness contract with a user simulator speaks chat; on other dialects a
        simulator works via the translate path."""
        raise NotImplementedError(
            f"user simulator over the relay path is not supported for {self.name}"
        )
