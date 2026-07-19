"""The per-rollout unit the interception layer serves.

One `RolloutSession` per rollout, registered on an interception server under the rollout's
secret. The rollout constructs it (model ctx, trace, task `@stop`s, limits) and the server
drives it: routes each intercepted model call to it, runs `refused()` before each turn,
injects the user simulator's replies, and stashes the real failure on `error`.
`RolloutLimits` is the framework's per-rollout budget (turns / tokens), checked between
turns.
"""

import asyncio
import hashlib
import logging
import threading
from collections import OrderedDict
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from verifiers.v1.clients import ModelContext
from verifiers.v1.trace import Trace
from verifiers.v1.types import Messages
from verifiers.v1.utils.textify import TextifyConfig, render_url

if TYPE_CHECKING:
    from verifiers.v1.errors import RolloutError
    from verifiers.v1.mcp import Respond

logger = logging.getLogger(__name__)

_TEXTIFY_CACHE_SIZE = 32
_TEXTIFY_SEEN_BYTES = 8192


@dataclass(frozen=True)
class RolloutLimits:
    """Per-rollout framework limits (None = no cap), checked before each turn is served.
    The first limit reached refuses the turn — halting any harness, the same mechanism as
    a @stop — and becomes the trace's stop condition. Each caps a trace computed property:
    `max_turns` -> num_turns, `max_input_tokens` -> num_input_tokens, `max_output_tokens` ->
    num_output_tokens, `max_total_tokens` -> num_total_tokens. Token caps are soft by one turn:
    they're checked between turns, so the turn that crosses a cap still completes."""

    max_turns: int | None = None
    max_input_tokens: int | None = None
    max_output_tokens: int | None = None
    max_total_tokens: int | None = None

    def reached(self, trace: Trace) -> str | None:
        """The name of the first limit `trace` has reached, or None if within all caps."""
        if self.max_turns is not None and trace.num_turns >= self.max_turns:
            return "max_turns"
        if (
            self.max_input_tokens is not None
            and trace.num_input_tokens >= self.max_input_tokens
        ):
            return "max_input_tokens"
        if (
            self.max_output_tokens is not None
            and trace.num_output_tokens >= self.max_output_tokens
        ):
            return "max_output_tokens"
        if (
            self.max_total_tokens is not None
            and trace.num_total_tokens >= self.max_total_tokens
        ):
            return "max_total_tokens"
        return None


@dataclass
class RolloutSession:
    ctx: ModelContext
    trace: Trace
    stops: list[Callable[[Trace], Awaitable[bool]]] = field(default_factory=list)
    limits: RolloutLimits = field(default_factory=RolloutLimits)
    textify: TextifyConfig = field(default_factory=TextifyConfig)
    """How the interception server renders this rollout's wire images to text
    (`verifiers.v1.utils.textify`); disabled by default, ascii when enabled."""
    _textify_cache: OrderedDict[bytes, str | None] = field(
        default_factory=OrderedDict, init=False
    )
    _textify_seen: bytearray = field(
        default_factory=lambda: bytearray(_TEXTIFY_SEEN_BYTES),
        init=False,
        repr=False,
    )
    """Bloom admission filter: false positives skip caching but never corrupt output."""
    _textify_lock: threading.Lock = field(
        default_factory=threading.Lock, init=False, repr=False
    )
    user: "Respond | None" = None
    """A user simulator the rollout sets before the harness runs (see `verifiers.v1.mcp.user`).
    When set, each model turn with no tool call is followed by the simulator's reply,
    injected as a user turn, and the model is re-prompted — all within one program request,
    transparently to the harness."""
    opening: Messages | None = None
    """Cached opening `respond("")` messages for a no-prompt task. Computed once and re-injected on
    every request until the first turn lands on the trace — so a retried opening request (e.g. the
    harness SDK retrying a transient model 502, before any turn is recorded) never calls `respond`
    twice and advances the simulator's queue past the opening."""
    _opening_textified: bool = field(default=False, init=False, repr=False)
    _opening_lock: asyncio.Lock = field(
        default_factory=asyncio.Lock, init=False, repr=False
    )
    error: "RolloutError | None" = None
    """The latest unresolved model-call failure. The harness only sees it as an HTTP error
    (and may swallow it, or exit non-zero), so the rollout re-raises this original error once the
    harness returns — recording the real `ProviderError` instead of a secondary `HarnessError`.
    Reset before each model turn, so a successful retry clears it."""
    last_request: bytes | None = None
    """Digest of the most recently served request body; with `last_response` /
    `last_response_error`, the replay cache that keeps the graph atomic under SDK retries. A retry re-sends the
    byte-identical request; when it matches, the interception server replays the recorded
    response instead of re-sampling and committing a second turn — which would fork the graph
    into a dead-end branch. Only a fully served request is cached, so a genuinely failed attempt
    still re-runs. Turns are issued sequentially (one outstanding request at a time), so a retry
    is always of the most recent request — keeping only the last one is sufficient and bounded."""
    last_response: dict | None = None
    """The response returned for `last_request`, replayed verbatim on a clean retry."""
    last_response_error: "RolloutError | None" = None
    """Post-commit failure replayed for `last_request` instead of silently returning success."""
    inflight: dict[bytes, "asyncio.Future[dict | RolloutError | None]"] = field(
        default_factory=dict
    )
    """Body digest -> the future of the attempt currently computing it. A retry that arrives
    while the first attempt is still in flight (a slow turn) awaits this future instead of
    starting a second inference — the other half of retry atomicity (the last-result fields
    cover retries after the attempt finished). Because a slow turn is coalesced rather than
    re-sampled, retries stay safe without an inflated client timeout. The future resolves to the
    served response, a post-commit error, or None if no servable result exists."""

    def render_image(self, url: str) -> str | None:
        """Render one wire image, retaining a scan-resistant recent working set."""
        key = hashlib.sha256(url.encode()).digest()
        with self._textify_lock:
            if key in self._textify_cache:
                self._textify_cache.move_to_end(key)
                return self._textify_cache[key]
            seen = True
            for offset in (0, 2):
                bit = int.from_bytes(key[offset : offset + 2], "big")
                byte, shift = divmod(bit, 8)
                mask = 1 << shift
                if not self._textify_seen[byte] & mask:
                    seen = False
                    self._textify_seen[byte] |= mask

        rendered = render_url(url, self.textify)
        with self._textify_lock:
            # A concurrent request may have rendered/admitted the same image meanwhile.
            if key in self._textify_cache:
                self._textify_cache.move_to_end(key)
                return self._textify_cache[key]
            if seen:
                # Old images beyond capacity may re-render, but never evict the recent set
                # as a harness scans its full retained history on every request.
                return rendered
            if len(self._textify_cache) >= _TEXTIFY_CACHE_SIZE:
                self._textify_cache.popitem(last=False)
            self._textify_cache[key] = rendered
            return rendered

    async def refused(self) -> str | None:
        """The framework's limits (turns / token budget) and `@stop` checks, run before each
        model call. Sets the stop condition and returns its name, else None. A refused first
        call halts the harness (its model call errors out); Harness.run treats it as clean. A task
        that ends a trajectory from `trace.state` does it with its own `@stop` (run here generically),
        so the interception server holds no opinion about the state's contents."""
        if (limit := self.limits.reached(self.trace)) is not None:
            self.trace.stop(limit)
            logger.debug("limit %r reached: id=%s", limit, self.trace.id)
            return limit
        for stop in self.stops:
            if await stop(self.trace):
                self.trace.stop(stop.__name__)
                logger.debug("stop %r fired: id=%s", stop.__name__, self.trace.id)
                return stop.__name__
        return None
