"""The `Dialect` abstraction: one native wire format, translated to vf for the trace.

A `Dialect[RespT]` is the per-format translator the interception server uses to build the
trace from the program's native request + the provider's native response. The server serves
every registered dialect's `routes` (see `dialects.DIALECTS`), so a request's format is resolved
from the endpoint the program's SDK posts to — the harness declares nothing.

The eval client preserves a request's native JSON fields except for eval-owned overrides, while a
dialect-owned `StreamParser` incrementally assembles a response copy for the trace; the renderer is chat-only.
A dialect is therefore mostly wire -> vf (`parse_request`/`parse_response`/`stream_parser`); the
exception is `apply_overrides` (impose the eval's model + sampling in this format's shape).
"""

import json
import logging
from abc import ABC, abstractmethod
from collections.abc import Callable, Mapping
from typing import Any, ClassVar, Generic, TypeVar
from urllib.parse import urlsplit

from pydantic import AnyHttpUrl, BaseModel, ValidationError
from pydantic_core import from_json

from verifiers.v1.configs.runtime import NetworkPolicyConfig
from verifiers.v1.types import (
    Request,
    Response,
    Sampling,
    SamplingConfig,
    ToolMessage,
    UserMessage,
)

RespT = TypeVar("RespT", bound=BaseModel)
RawRequest = dict[str, Any]

logger = logging.getLogger(__name__)

PROVIDER_CAPABILITY_POLICY_CODE = "provider_capability_unavailable"
CAPABILITY_NOTICE = (
    "Network protocol blocked fetching a resource. Continue without those capabilities; "
    "use local tools or inline data already present in the conversation, and do not retry "
    "the blocked provider-side operation."
)


def blocked_url(value: str, policy: NetworkPolicyConfig) -> bool:
    """Whether a provider-resolved resource is neither inline nor policy-permitted."""
    if value.lower().startswith("data:"):
        return False
    try:
        url = AnyHttpUrl(value)
    except ValidationError:
        return True
    host = url.host.lower().rstrip(".").strip("[]")
    return not policy.permits(url.scheme, host, url.port)


def blocked_path(
    value,
    path: str,
    policy: NetworkPolicyConfig,
    blocked_item: Callable[[dict, str, NetworkPolicyConfig], str | None],
) -> str | None:
    """The first policy-blocked path under `value`, or None. Lists recurse per index,
    non-dicts pass, a non-`direct` `caller` is blocked in every format, and `blocked_item`
    applies the format's own rules to each dict."""
    if isinstance(value, list):
        for index, item in enumerate(value):
            if blocked := blocked_path(item, f"{path}[{index}]", policy, blocked_item):
                return blocked
        return None
    if not isinstance(value, dict):
        return None
    caller = value.get("caller")
    if caller is not None and not (
        isinstance(caller, dict) and caller.get("type") == "direct"
    ):
        return f"{path}.caller.type"
    return blocked_item(value, path, policy)


def mediate_parts(
    value,
    path: str,
    policy: NetworkPolicyConfig,
    blocked: Callable[[Any, str, NetworkPolicyConfig], str | None],
    wrappers: tuple[str, ...] = (),
) -> tuple[Any, list[str]]:
    """Drop the policy-blocked parts of a content list and report their paths; a non-list
    value is kept whole or replaced by "". A part whose type is in `wrappers` is checked
    without its content, then its content is mediated in place."""
    if not isinstance(value, list):
        blocked_at = blocked(value, path, policy)
        return ("", [blocked_at]) if blocked_at else (value, [])

    mediated = []
    capabilities = []
    for index, part in enumerate(value):
        item_path = f"{path}[{index}]"
        if isinstance(part, dict) and part.get("type") in wrappers:
            if blocked_at := blocked({**part, "content": []}, item_path, policy):
                capabilities.append(blocked_at)
                continue
            content, removed = mediate_parts(
                part.get("content"), f"{item_path}.content", policy, blocked, wrappers
            )
            if removed:
                part["content"] = content or ""
                capabilities.extend(removed)
        elif blocked_at := blocked(part, item_path, policy):
            capabilities.append(blocked_at)
            continue
        mediated.append(part)
    return mediated, capabilities


def user_and_tool_messages(request: Request) -> list[UserMessage | ToolMessage]:
    """The rewritable messages of a request, in order."""
    return [m for m in request.messages if isinstance(m, (UserMessage, ToolMessage))]


def provider_allowed_domains(
    policy: NetworkPolicyConfig, requested: object = None
) -> list[str]:
    """Translate a network policy to provider domain-filter semantics without widening it."""
    if policy.block or not policy.allow or "*" in policy.allow:
        return []
    domains = []
    for rule in policy.allow:
        try:
            url = urlsplit(rule if "://" in rule else f"//{rule}")
            port = url.port
        except ValueError:
            return []
        host = (url.hostname or "").lower().rstrip(".")
        domain = host.removeprefix("*.")
        if (
            url.scheme
            or port is not None
            or not host.startswith("*.")
            or not domain
            or "*" in domain
            or not domain.isascii()
        ):
            return []
        domains.append(domain)
    domains = list(dict.fromkeys(domains))
    if requested is None:
        return domains
    if not isinstance(requested, list):
        return []
    requested_domains = []
    for domain in requested:
        if not isinstance(domain, str):
            return []
        try:
            url = urlsplit(domain if "://" in domain else f"//{domain}")
            port = url.port
        except ValueError:
            return []
        host = (url.hostname or "").lower().rstrip(".")
        if (
            url.scheme
            or port is not None
            or url.username is not None
            or url.path
            or url.query
            or url.fragment
            or not host
            or "*" in host
            or not host.isascii()
        ):
            return []
        requested_domains.append(host)
    intersection = []
    for allowed in domains:
        for requested_domain in requested_domains:
            if allowed == requested_domain or allowed.endswith(f".{requested_domain}"):
                intersection.append(allowed)
            elif requested_domain.endswith(f".{allowed}"):
                intersection.append(requested_domain)
    return list(dict.fromkeys(intersection))


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
    as mutable native JSON because the gateway preserves provider extensions while mediating and
    rewriting them. Implement a `Dialect` + register it in `dialects.DIALECTS` and a harness
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

    @abstractmethod
    def mediate_external_capabilities(
        self, body: RawRequest, policy: NetworkPolicyConfig
    ) -> tuple[RawRequest, list[str]]:
        """Remove provider-side capabilities during restricted execution. Implementations add
        the same policy context on every call because the agent does not retain injected request
        content. Returned paths never contain request values."""

    @abstractmethod
    def parse_request(self, body: RawRequest) -> Request:
        """The native request -> the typed model request."""

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

    @abstractmethod
    def rewrite_request(
        self, body: RawRequest, before: Request, after: Request
    ) -> None:
        """Patch rewritten user/tool messages into the native conversation."""

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
        eval's applied). Capability mediation may subsequently remove restricted fields."""
