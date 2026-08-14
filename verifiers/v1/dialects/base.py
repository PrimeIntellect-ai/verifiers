"""The `Dialect` abstraction: one native wire format, translated to vf for the trace.

A `Dialect[ReqT, RespT]` is the per-format translator the interception server uses to build the
trace from the program's native request + the provider's native response. The server serves
every registered dialect's `routes` (see `dialects.DIALECTS`), so a request's format is resolved
from the endpoint the program's SDK posts to — the harness declares nothing.

The eval client preserves a request's native JSON fields except for eval-owned overrides, while a
dialect-owned `StreamParser` incrementally assembles a response copy for the trace. Training clients
render the canonical request and ask the dialect to serialize the completed canonical response.
A dialect therefore owns both wire -> vf (`parse_request`/`parse_response`/`stream_parser`) and the
small vf -> wire surface needed for renderer-backed inference (`serialize_response`/stream events).
"""

import json
import logging
from abc import ABC, abstractmethod
from collections.abc import Callable, Iterator, Mapping
from typing import ClassVar, Generic, TypeVar

from pydantic import BaseModel
from pydantic_core import from_json

from verifiers.v1.types import Request, Response, Sampling, SamplingConfig

ReqT = TypeVar("ReqT")
RespT = TypeVar("RespT", bound=BaseModel)

logger = logging.getLogger(__name__)

PROVIDER_CAPABILITY_POLICY_CODE = "provider_capability_unavailable"
CAPABILITY_NOTICE = (
    "Network protocol blocked fetching a resource. Continue without those capabilities; "
    "use local tools or inline data already present in the conversation, and do not retry "
    "the blocked provider-side operation."
)


def blocked_url(value: str) -> bool:
    """Whether a provider-resolved resource is not inline data."""
    return not value.lower().startswith("data:")


def append_user_notice(
    messages: list,
    *,
    text_type: str = "text",
    message_type: str | None = None,
) -> None:
    """Add stable restricted-network context to the earliest user input."""
    part = {"type": text_type, "text": CAPABILITY_NOTICE}
    for message in messages:
        if not isinstance(message, dict) or message.get("role") != "user":
            continue
        content = message.get("content")
        if isinstance(content, list):
            message["content"] = [*content, part]
        elif isinstance(content, str):
            message["content"] = (
                f"{content}\n\n{CAPABILITY_NOTICE}" if content else CAPABILITY_NOTICE
            )
        else:
            message["content"] = [part]
        return
    message = {"role": "user", "content": [part]}
    if message_type is not None:
        message["type"] = message_type
    messages.append(message)


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


def iter_sse_reverse(raw: bytes) -> Iterator[dict]:
    """Yield JSON SSE payloads from the end without decoding earlier events."""
    decoded = raw.decode("utf-8", errors="replace")
    first_newline = decoded.find("\n")
    separator = (
        "\r\n\r\n"
        if first_newline > 0 and decoded[first_newline - 1] == "\r"
        else "\n\n"
    )
    for block in reversed(decoded.split(separator)):
        data = "\n".join(
            line.removeprefix("data:").strip()
            for line in block.splitlines()
            if line.startswith("data:")
        )
        if not data or data == "[DONE]":
            continue
        yield json.loads(data)


class StreamParser(ABC):
    """Incrementally assemble one native SSE stream into a vf response."""

    feed: Callable[[bytes], None]
    """Consume one complete SSE event without retaining its raw bytes."""

    on_done: Callable[[], None] | None = None
    """Preserve terminal state before events following the DONE sentinel."""

    @abstractmethod
    def finish(self) -> Response:
        """Finalize and return the assembled response after the stream ends."""


class Dialect(ABC, Generic[ReqT, RespT]):
    """One native API's wire format, fully typed over its request (`ReqT`) and response
    (`RespT`). The single place a protocol lives: implement a `Dialect` + register it in
    `dialects.DIALECTS` and a harness speaking that format works end-to-end (the eval client and
    interception server are generic over this interface)."""

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

    def streaming(self, body: ReqT) -> bool:
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

    @abstractmethod
    def mediate_external_capabilities(self, body: ReqT) -> tuple[ReqT, list[str]]:
        """Remove provider-side capabilities during restricted execution. Implementations add
        the same policy context on every call because the agent does not retain injected request
        content. Returned paths never contain request values."""

    @abstractmethod
    def parse_request(self, body: ReqT) -> Request:
        """The native request -> the typed model request."""

    def parse_sampling(self, body: ReqT) -> Sampling:
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

    @abstractmethod
    def serialize_response(self, response: Response, model: str) -> dict:
        """Serialize a renderer-backed canonical response in this dialect's native shape."""

    @abstractmethod
    def rewrite_request(self, body: ReqT, before: Request, after: Request) -> None:
        """Patch rewritten user/tool messages into the native conversation."""

    @abstractmethod
    def rewrite_response(self, raw: dict, text: str) -> None:
        """Replace the native assistant response with inert text."""

    @abstractmethod
    def stream_events(
        self,
        raw: dict,
        *,
        include_start: bool = True,
        sequence_number: int = 0,
    ) -> list[bytes]:
        """Serialize a completed or rewritten response as a minimal native SSE stream."""

    @abstractmethod
    def stream_start(self, body: ReqT, *, sequence_number: int = 0) -> list[bytes]:
        """Start a buffered native stream before generation has completed."""

    @abstractmethod
    def stream_heartbeat(self, body: ReqT, *, sequence_number: int = 0) -> bytes:
        """One valid, non-content protocol event while buffered generation is in flight.

        Unlike an SSE comment, this reaches clients whose liveness watchdog resets only after
        decoding an API event. It must not change the assistant content assembled by the client.
        """

    def stream_error(self, message: str, *, sequence_number: int = 0) -> bytes:
        """A streamed error in the default OpenAI-compatible envelope."""
        return f"data: {json.dumps(self.error_body(message))}\n\n".encode()

    @abstractmethod
    def stream_parser(self) -> StreamParser:
        """Create the per-request incremental parser for a native SSE response."""

    @abstractmethod
    def apply_overrides(self, body: ReqT, model: str, sampling: SamplingConfig) -> ReqT:
        """Return `body` with the eval's `model` + `sampling` imposed in this protocol's shape —
        model overlays; sampling is authoritative (the program's sampling keys are dropped, the
        eval's applied). Capability mediation may subsequently remove restricted fields."""
