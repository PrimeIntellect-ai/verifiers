"""The `Dialect` abstraction: one native wire format, translated to vf for the trace.

A `Dialect[RespT]` is the per-format translator the interception server uses to build the
trace from the program's native request + the provider's native response. The server serves
every registered dialect's `routes` (see `dialects.DIALECTS`), so a request's format is resolved
from the endpoint the program's SDK posts to — the harness declares nothing.

The eval client preserves a request's native JSON fields except for eval-owned overrides, while a
dialect-owned `StreamParser` incrementally assembles a response copy for the trace; the renderer is chat-only.
Request bindings project native JSON into task messages and apply supported hook edits at
their original locations. Network policy is checked separately by the gateway.
"""

import json
import logging
from abc import ABC, abstractmethod
from collections.abc import Callable, Mapping
from typing import Any, ClassVar, Generic, TypeVar

from pydantic import BaseModel
from pydantic_core import from_json

from verifiers.v1.dialects.request import NativeRequest
from verifiers.v1.types import (
    AssistantMessage,
    Request,
    Response,
    Sampling,
    SamplingConfig,
)

RespT = TypeVar("RespT", bound=BaseModel)
RawRequest = dict[str, Any]

logger = logging.getLogger(__name__)


def is_sse_done_event(raw: bytes) -> bool:
    """Whether one complete SSE event carries the DONE sentinel."""
    # Ordinary OpenAI events carry JSON objects; reject their hot path before splitting lines.
    if raw.startswith((b"data: {", b"data:{")):
        return False
    data = b"\n".join(
        line.removeprefix(b"data:").strip()
        for line in raw.splitlines()
        if line.startswith(b"data:")
    )
    return data == b"[DONE]"


def parse_sse_event(raw: bytes) -> dict | None:
    """Parse one complete SSE event's JSON data payload, ignoring comments and sentinels."""
    data = b"\n".join(
        line.removeprefix(b"data:").strip()
        for line in raw.splitlines()
        if line.startswith(b"data:")
    )
    if not data or data == b"[DONE]":
        return None
    try:
        return from_json(data)
    except ValueError:
        logger.warning(
            "SSE JSON fast-path failed; falling back to stdlib with invalid UTF-8 replacement"
        )
        return json.loads(data.decode("utf-8", errors="replace"))


class StreamParser(ABC):
    """Incrementally assemble one native SSE stream into a vf response."""

    feed: Callable[[bytes], None]
    """Consume one complete SSE event without retaining its raw bytes."""

    on_done: Callable[[], None] | None = None
    """Preserve terminal state before events following the DONE sentinel."""

    @abstractmethod
    def finish(self) -> Response:
        """Finalize and return the assembled response after the stream ends."""


class Dialect(ABC, Generic[RespT]):
    """One native API's wire format, typed over its validated response (`RespT`). Requests stay
    as mutable native JSON because the gateway preserves provider extensions while applying
    supported edits. Implement a `Dialect` + register it in `dialects.DIALECTS` and a harness
    speaking that format works end-to-end."""

    sampling_fields: ClassVar[frozenset[str]] = frozenset()
    """Request keys that are call settings — what shapes generation given the same
    conversation: decoding knobs, budgets/stops, reasoning effort, output contract.
    A whitelist, so payload, conversation state, and tracking fields can never leak
    into the per-call record by omission; an unlisted knob is simply not recorded."""

    routes: ClassVar[tuple[str, ...]]
    """The endpoint path(s) a program's SDK posts model turns to. The interception server serves
    one handler per route, so the wire format is resolved from the route the SDK chose (it
    commits to one when the client is picked) rather than declared by the harness."""

    aux_routes: ClassVar[tuple[str, ...]] = ()
    """Side endpoints the SDK may call that aren't model turns (e.g. Anthropic's
    `count_tokens`): relayed as native JSON by the eval client, never recorded on the trace."""

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

    def streaming(self, body: RawRequest) -> bool:
        """Whether the request asks for a streamed (SSE) response."""
        return bool(body.get("stream"))

    def is_terminal_event(self, chunk: bytes) -> bool:
        """Whether this complete SSE event ends the model's turn for the client. The
        interception server withholds the terminal event (and anything after it) until the
        turn is recorded, so a client that ends its turn on it can't race ahead to scoring
        with the turn still uncommitted. Defaults to the `[DONE]` sentinel; a dialect whose
        client ends on an earlier event (e.g. Responses' `response.completed`) overrides this."""
        return is_sse_done_event(chunk)

    def error_body(self, message: str) -> dict:
        """An error payload in this format's error shape (OpenAI by default)."""
        return {"error": {"message": message, "type": "invalid_request_error"}}

    def stream_keepalive(self, first: bool) -> bytes:
        """A keepalive for a committed SSE stream whose turn is still being produced (`first`
        on the stream's first one). A comment line by default: these clients count any bytes
        as activity. A dialect whose clients only count events sends a no-op event instead."""
        # Don't terminate an empty event; some SSE clients try to JSON-decode it.
        return b": keepalive\n"

    def stream_error(self, error: dict) -> bytes:
        """An `error_body` as an SSE event, for a failure after the stream is committed.
        OpenAI SDKs raise on any event carrying `error`."""
        return b"data: " + json.dumps(error).encode() + b"\n\n"

    def parse_request(self, body: RawRequest) -> Request:
        """The task-facing projection; gateway callers retain its native bindings."""
        return self.bind_request(body).view

    @abstractmethod
    def bind_request(self, body: RawRequest) -> NativeRequest:
        """Project native content and retain the locations of supported edits."""

    def validate_training(self, body: RawRequest) -> None:
        raise ValueError(f"Training does not support the {self.upstream_path} protocol")

    def parse_sampling(self, body: RawRequest) -> Sampling:
        """The native request's call settings -> the canonical `Sampling` (for the
        trace's per-call records): the `sampling_fields` whitelist, with this format's
        aliases mapped onto the typed knobs; dialect-specific keys ride as extras."""
        return Sampling.model_validate(
            {k: v for k, v in body.items() if k in self.sampling_fields}
        )

    @abstractmethod
    def parse_response(self, response: RespT) -> Response:
        """A native (non-streamed) response -> the vf `Response` we consume."""

    def validate_response(self, raw: dict) -> RespT:
        """Validate a native response, normalizing provider-compatible extensions if needed."""
        return self.response_type.model_validate(raw)

    def replace_response(self, response: Response, text: str) -> Response:
        """Apply a text replacement and regenerate the task view in one operation."""
        if response.raw is None:
            raise ValueError("response replacement requires a native response")
        self.rewrite_response(response.raw, text)
        rewritten = self.parse_response(self.validate_response(response.raw))
        rewritten.raw = response.raw
        return rewritten

    @abstractmethod
    def rewrite_response(self, raw: dict, text: str) -> None:
        """Replace the native assistant response with inert text."""

    @abstractmethod
    def stream_events(self, raw: dict) -> list[bytes]:
        """Serialize a rewritten response as a minimal native SSE stream."""

    @abstractmethod
    def stream_parser(self) -> StreamParser:
        """Create the per-request incremental parser for a native SSE response."""

    @abstractmethod
    def apply_overrides(
        self, body: RawRequest, model: str, sampling: SamplingConfig
    ) -> RawRequest:
        """Return `body` with the eval's `model` + `sampling` imposed in this protocol's shape —
        model overlays; sampling is authoritative (the program's sampling keys are dropped, the
        eval's applied)."""


_PROVIDER_STATE_FIELDS = frozenset({"encrypted_content", "signature", "data", "phase"})


def with_provider_identity(message: AssistantMessage) -> AssistantMessage:
    """Project native continuation identity before a message enters the shared graph."""
    identity = []
    for item in message.provider_state or []:
        kind = item.get("type") or (
            "message" if item.get("role") == "assistant" else ""
        )
        hashed_state = {
            key: item[key]
            for key in _PROVIDER_STATE_FIELDS
            if item.get(key) is not None
        }
        if kind == "message" and isinstance(item.get("content"), list):
            # Keep content parts the typed message does not expose, such as refusals.
            unparsed_content = [
                part
                for part in item.get("content") or []
                if part.get("type") not in ("input_text", "output_text")
            ]
            if unparsed_content:
                hashed_state["content"] = unparsed_content
        represented = kind in ("message", "reasoning") or (
            kind in ("function_call", "custom_tool_call")
            and any(call.id == item.get("call_id") for call in message.tool_calls or [])
        )
        if represented and not hashed_state:
            continue
        # Unknown provider items still distinguish built-in calls and actions.
        state = hashed_state if represented else item
        identity.append((kind, state))
    message.provider_identity = identity if message.provider_state else None
    return message
