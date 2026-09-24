"""The `Dialect` abstraction: one native wire format, translated to vf for the trace.

A `Dialect` is the per-format translator the interception server uses to build the trace from
the program's native request + the provider's native response. The server serves every
registered dialect's `routes` (see `dialects.DIALECTS`), so a request's format is resolved from
the endpoint the program's SDK posts to — the harness declares nothing.

The client relays a request's native JSON with explicit sampling settings applied, and
the dialect reads copies of the JSON into vf types. The provider SDK assembles streams.
Task hooks edit the typed copy, and their edits flow back into the native request through
the setters `parse_request` returns. What a
restricted runtime may send lives apart, in `dialects.policy`.
"""

import json
from abc import ABC, abstractmethod
from collections.abc import Callable, Mapping
from typing import Any, ClassVar

from verifiers.v1.types import Message, Request, Response, Sampling, SamplingConfig

RawRequest = dict[str, Any]
Setter = Callable[[Message], None]
"""Writes an edited user or tool message back into the native request it was parsed from."""


def patch_content(content, replacement, native=None):
    """Keep each edited part's own native metadata through hook copies and reordering.

    Transient `_native` attributes survive deep copies without being model fields or
    private attributes, so equality and trace serialization ignore them. New parts have
    none. A scalar projection can only retain metadata from one original text block.
    """
    if isinstance(content, str):
        if (
            isinstance(native, list)
            and len(native) == 1
            and native[0].get("type") == "text"
        ):
            return [native[0] | {"text": content}]
        return replacement
    patched = []
    for source, part in zip(content, replacement, strict=True):
        original = getattr(source, "_native", {})
        if original.get("type") == part.get("type"):
            part = original | {
                key: original[key] | value
                if isinstance(value, dict)
                and isinstance(original.get(key), dict)
                and original[key].get("type") == value.get("type")
                else value
                for key, value in part.items()
            }
        patched.append(part)
    return patched


class Dialect(ABC):
    """One native API's wire format. Requests and responses stay native JSON because the
    gateway preserves provider extensions while relaying them. Implement a `Dialect` + register
    it in `dialects.DIALECTS` and a harness speaking that format works end-to-end."""

    sampling_fields: ClassVar[frozenset[str]] = frozenset()
    """Request keys that are call settings — what shapes generation given the same
    conversation: decoding knobs, budgets/stops, reasoning effort, output contract.
    A whitelist, so payload, conversation state, and tracking fields can never leak
    into the per-call record by omission; an unlisted knob is simply not recorded."""

    max_tokens_keys: ClassVar[tuple[str, ...]]
    """The request keys that cap output in this format, the one the eval's cap is sent as
    first (e.g. Responses' `max_output_tokens`)."""

    effort_path: ClassVar[tuple[str, ...]]
    """Where this format carries reasoning effort: a top-level key, or an object and its key
    (e.g. Responses' `("reasoning", "effort")`)."""

    routes: ClassVar[tuple[str, ...]]
    """The endpoint path(s) a program's SDK posts model turns to. The interception server serves
    one handler per route, so the wire format is resolved from the route the SDK chose (it
    commits to one when the client is picked) rather than declared by the harness."""

    aux_routes: ClassVar[tuple[str, ...]] = ()
    """Side endpoints the SDK may call that aren't model turns (e.g. Anthropic's
    `count_tokens`): relayed as native JSON by the client, never recorded on the trace."""

    upstream_path: ClassVar[str]
    """The provider endpoint the proxy forwards to for this format (e.g. `/chat/completions`)."""

    event_type: Any
    """The provider SDK event type used for streamed responses."""

    def auth_headers(self, api_key: str) -> dict[str, str]:
        """The provider auth headers for this format. Defaults to OAuth2 Bearer (every
        OpenAI-compatible provider); override for a different scheme (e.g. Anthropic's
        `x-api-key` + `anthropic-version`)."""
        return {"Authorization": f"Bearer {api_key}"}

    def secret(self, headers: Mapping[str, str]) -> str:
        """The per-rollout secret from the request, read from this format's auth carrier
        (default: an `Authorization: Bearer` token; Anthropic uses `x-api-key`)."""
        return headers.get("Authorization", "").removeprefix("Bearer ")

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

    @abstractmethod
    def parse_request(self, body: RawRequest) -> tuple[Request, list[Setter | None]]:
        """The native request -> the typed model request, plus one entry per message: a setter
        writing an edited user or tool message back into `body` (None for the rest)."""

    def parse_sampling(self, body: RawRequest) -> Sampling:
        """The native request's call settings -> the canonical `Sampling` (for the trace's
        per-call records): the `sampling_fields` whitelist, with this format's output cap and
        reasoning effort mapped onto the typed knobs; dialect-specific keys ride as extras."""
        settings = {k: v for k, v in body.items() if k in self.sampling_fields}
        caps = [
            cap
            for key in self.max_tokens_keys
            if (cap := settings.pop(key, None)) is not None
        ]
        if caps:
            settings["max_tokens"] = caps[0]
        *parent, leaf = self.effort_path
        if parent and isinstance(nested := settings.get(parent[0]), dict):
            nested = dict(nested)
            if nested.get(leaf):
                settings["reasoning_effort"] = nested.pop(leaf)
            if nested:
                settings[parent[0]] = nested
            else:
                settings.pop(parent[0])
        return Sampling.model_validate(settings)

    def apply_overrides(
        self, body: RawRequest, model: str, sampling: SamplingConfig
    ) -> RawRequest:
        """Return `body` with the eval's `model` imposed and its sampling applied in this
        format's shape. Explicit settings replace the program's values; unset settings keep
        them. Canonical output caps and effort take precedence over provider-specific aliases.
        Capability mediation may subsequently remove restricted fields."""
        overrides = sampling.wire_args()
        caps = [
            overrides.pop(key)
            for key in dict.fromkeys(("max_tokens", *self.max_tokens_keys))
            if key in overrides
        ]
        effort = overrides.pop("reasoning_effort", None)
        steered = {**body, **overrides, "model": model}
        if caps:
            # One cap rides upstream: the program's aliases would conflict with the eval's.
            for key in self.max_tokens_keys:
                steered.pop(key, None)
            steered[self.max_tokens_keys[0]] = caps[0]
        *parent, leaf = self.effort_path
        if not parent:
            if effort is not None:
                steered[leaf] = effort
            return steered
        # Effort nests in an object whose other keys (e.g. a reasoning summary) the program
        # and the eval both may set: merge them, the eval's winning.
        nested = {**(body.get(parent[0]) or {}), **(overrides.get(parent[0]) or {})}
        if effort is not None:
            nested[leaf] = effort
        if nested:
            steered[parent[0]] = nested
        return steered

    @abstractmethod
    def parse_response(self, response: dict) -> Response:
        """A native response -> the vf `Response` we consume."""

    @abstractmethod
    def rewrite_response(self, raw: dict, text: str) -> None:
        """Replace the native assistant response with inert text."""

    @abstractmethod
    def stream_events(self, raw: dict) -> list[bytes]:
        """Serialize a response verifiers produced (a train-client turn, a hook's inert
        rewrite) as a minimal native SSE stream."""

    @abstractmethod
    async def read_stream(self, stream) -> dict:
        """Assemble a native response using the provider SDK's stream helpers."""
