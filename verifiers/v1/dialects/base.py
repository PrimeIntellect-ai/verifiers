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
from glob import has_magic
from typing import Any, ClassVar, Generic, TypeVar

from pydantic import AnyHttpUrl, BaseModel, ValidationError
from pydantic_core import from_json

from verifiers.v1.configs.runtime import (
    NetworkPolicyConfig,
    intersect_network_hosts,
    parse_network_rule,
)
from verifiers.v1.types import Request, Response, Sampling, SamplingConfig

RespT = TypeVar("RespT", bound=BaseModel)
RawRequest = dict[str, Any]

logger = logging.getLogger(__name__)

PROVIDER_CAPABILITY_POLICY_CODE = "provider_capability_unavailable"
CAPABILITY_NOTICE = (
    "Some request content or provider-side capabilities were omitted because they are "
    "blocked by the network policy or cannot enforce it."
)


class RequestFilter:
    """One request's omissions and blocked URLs; subclasses define native wire rules."""

    wrappers: tuple[str, ...] = ()

    def __init__(self, policy: NetworkPolicyConfig):
        self.policy = policy
        self.blocked_urls: list[str] = []
        self.capabilities: list[str] = []

    def blocked_url(self, value: object) -> bool:
        if not isinstance(value, str):
            return True
        if value.lower().startswith("data:"):
            return False
        try:
            url = AnyHttpUrl(value)
        except ValidationError:
            return True
        blocked = not self.policy.permits(url.scheme, url.host.strip("[]"), url.port)
        if blocked:
            self.blocked_urls.append(value)
        return blocked

    def blocked(self, value, path: str) -> str | None:
        """Find the first forbidden part, treating nested content as one unit."""
        if isinstance(value, list):
            for index, item in enumerate(value):
                if blocked := self.blocked(item, f"{path}[{index}]"):
                    return blocked
            return None
        if not isinstance(value, dict):
            return None
        caller = value.get("caller")
        if caller is not None and not (
            isinstance(caller, dict) and caller.get("type") == "direct"
        ):
            return f"{path}.caller.type"
        return self.blocked_part(value, path)

    def blocked_part(self, value: dict, path: str) -> str | None:
        raise NotImplementedError

    def mediate(self, value, path: str):
        """Remove forbidden parts while retaining supported wrapper blocks."""
        if not isinstance(value, list):
            if blocked := self.blocked(value, path):
                self.capabilities.append(blocked)
                return ""
            return value
        mediated = []
        for index, block in enumerate(value):
            item_path = f"{path}[{index}]"
            wrapper = isinstance(block, dict) and block.get("type") in self.wrappers
            scan = {**block, "content": []} if wrapper else block
            if blocked := self.blocked(scan, item_path):
                self.capabilities.append(blocked)
                continue
            if wrapper:
                self.content(block, "content", f"{item_path}.content")
            mediated.append(block)
        return mediated

    def content(self, parent: dict, key: str, path: str) -> bool:
        """Rewrite a content field only when filtering removes something."""
        before = len(self.capabilities)
        content = self.mediate(parent.get(key), path)
        changed = len(self.capabilities) != before
        if changed:
            parent[key] = content or ""
        return changed

    def tools(self, value, path: str = "tools") -> list[dict]:
        if value is not None and not isinstance(value, list):
            self.capabilities.append(path)
            return []
        tools = []
        for index, tool in enumerate(value or []):
            if (filtered := self.tool(tool, f"{path}[{index}]")) is not None:
                tools.append(filtered)
        return tools

    def tool(self, value, path: str) -> dict | None:
        raise NotImplementedError


def provider_domains(
    policy: NetworkPolicyConfig, requested: object = None
) -> list[str]:
    """Translate allow/block rules to provider filters without changing their scope.

    Provider filters include subdomains, so exact hosts need a covering wildcard rule.
    Empty results mean the policy cannot be represented; never send an empty filter.
    """
    rules = policy.block or policy.allow
    if requested is not None and not isinstance(requested, list):
        return []
    hosts, requested_domains = [], []
    for entries, output, is_filter in (
        (rules, hosts, False),
        (requested or [], requested_domains, True),
    ):
        for rule in entries:
            if not isinstance(rule, str):
                return []
            try:
                url, host, port = parse_network_rule(rule)
            except ValueError:
                return []
            if is_filter and (
                url.username is not None or url.path or url.query or url.fragment
            ):
                return []
            domain = host if is_filter else host.removeprefix("*.")
            if (
                url.scheme
                or port is not None
                or not domain
                or has_magic(domain)
                or not domain.isascii()
            ):
                return []
            output.append(host)
    for host in hosts:
        if not host.startswith("*.") and not (
            policy.block
            and any(
                wildcard.startswith("*.")
                and intersect_network_hosts(wildcard, host) == host
                for wildcard in hosts
            )
        ):
            return []
    domains = list(dict.fromkeys(host.removeprefix("*.") for host in hosts))
    if requested is None:
        return domains
    if policy.block:
        return list(dict.fromkeys([*domains, *requested_domains]))
    intersection = []
    for allowed in domains:
        for requested_domain in requested_domains:
            if host := intersect_network_hosts(f"*.{allowed}", f"*.{requested_domain}"):
                intersection.append(host.removeprefix("*."))
    return list(dict.fromkeys(intersection))


def append_user_notice(
    messages: list,
    *,
    blocked_urls: list[str],
    text_type: str = "text",
    message_type: str | None = None,
) -> None:
    """Explain an actual policy-driven omission in the earliest user input."""
    notice = CAPABILITY_NOTICE
    if blocked_urls:
        notice += "\nBlocked URLs: " + ", ".join(
            json.dumps(url) for url in dict.fromkeys(blocked_urls)
        )
        notice += "\nCircumventing this block is forbidden."
    part = {"type": text_type, "text": notice}
    for message in messages:
        if not isinstance(message, dict) or message.get("role") != "user":
            continue
        content = message.get("content")
        if isinstance(content, list):
            message["content"] = [*content, part]
        elif isinstance(content, str):
            message["content"] = f"{content}\n\n{notice}" if content else notice
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
        """Filter blocked content and constrain provider tools to the network policy.

        Provider tools execute outside runtime egress controls; remove them when their
        filters cannot express the policy. Add context only when something is removed.
        Returned paths never contain request values.
        """

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
