"""The per-rollout unit the interception layer serves.

One `RolloutSession` per rollout, registered on an interception server under the rollout's
secret. The rollout constructs it (model ctx, trace, task `@stop`s, limits) and the server
drives it: routes each intercepted model call to it, runs `refused()` before each turn,
injects the user simulator's replies, and stashes the real failure on `error`.
`RolloutLimits` is the framework's per-rollout budget (turns / tokens), checked between
turns.
"""

import asyncio
import inspect
import logging
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from verifiers.v1.clients import ModelContext
from verifiers.v1.decorators import invoke
from verifiers.v1.intercept import (
    Direction,
    InterceptExchange,
    InterceptRecord,
    Terminate,
    _response_tool_call_items,
    _snippet,
    drop_response_tool_calls,
)
from verifiers.v1.trace import Trace
from verifiers.v1.types import (
    AssistantMessage,
    Messages,
    SystemMessage,
    ToolMessage,
    UserMessage,
    content_text,
)

MESSAGE_TYPES = (SystemMessage, UserMessage, AssistantMessage, ToolMessage)
"""What an `@intercept` handler returns to block an exchange, answering with its text."""

if TYPE_CHECKING:
    from verifiers.v1.dialects import Dialect
    from verifiers.v1.errors import RolloutError
    from verifiers.v1.mcp import Respond

logger = logging.getLogger(__name__)


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
    intercepts: list[Callable[..., Awaitable[Any]]] = field(default_factory=list)
    """The task's `@intercept` handlers (see `verifiers.v1.intercept`), run over every
    model exchange — the request body inbound, the response outbound — by the interception
    server via `run_intercepts`. Empty means exchanges pass through untouched."""
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
    error: "RolloutError | None" = None
    """The latest unresolved model-call failure. The harness only sees it as an HTTP error
    (and may swallow it, or exit non-zero), so the rollout re-raises this original error once the
    harness returns — recording the real `ProviderError` instead of a secondary `HarnessError`.
    Reset before each model turn, so a successful retry clears it."""
    last_request: bytes | None = None
    """Digest of the most recently served request body; with `last_response`, the replay cache
    that keeps the message graph atomic under harness-SDK retries. A retry re-sends the
    byte-identical request; when it matches, the interception server replays the recorded
    response instead of re-sampling and committing a second turn — which would fork the graph
    into a dead-end branch. Only a fully served request is cached, so a genuinely failed attempt
    still re-runs. Turns are issued sequentially (one outstanding request at a time), so a retry
    is always of the most recent request — keeping only the last one is sufficient and bounded."""
    last_response: dict | None = None
    """The response returned for `last_request`, replayed verbatim on a retry."""
    inflight: dict[bytes, "asyncio.Future[dict | None]"] = field(default_factory=dict)
    """Body digest -> the future of the attempt currently computing it. A retry that arrives
    while the first attempt is still in flight (a slow turn) awaits this future instead of
    starting a second inference — the other half of retry atomicity (with `last_response`, which
    covers a retry after the attempt finished). Because a slow turn is coalesced rather than
    re-sampled, retries stay safe without an inflated client timeout. The future resolves to the
    served response, or to None if the attempt produced no servable response (error/refuse)."""

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

    async def run_intercepts(
        self, direction: Direction, raw: dict, dialect: "Dialect"
    ) -> Terminate | None:
        """The task's `@intercept` handlers, run over one wire exchange in priority order.
        `raw` is the native request body (inbound) or response object (outbound), mutated in
        place. Handlers get name-injected `task`/`trace`/`exchange` like scorers. Every action
        lands on `trace.interceptions`: an in-place rewrite (auto-detected by digest when a
        handler mutates `raw` but returns None and records nothing itself), a response-side
        block (the handler returned a
        `Message`; the tool calls are dropped here and the model gets its text as the answer),
        and any `Terminate` — which a request-side `Message` becomes. Returns the
        first `Terminate` (short-circuiting the rest): it stops the trace and records its reward
        when it carries one. Handler errors propagate to the server's boundary, like `@stop`s."""
        if not self.intercepts:
            return None
        exchange = InterceptExchange(direction, raw, self.trace, dialect)
        available = {
            "task": self.trace.task.data,
            "trace": self.trace,
            "exchange": exchange,
        }
        for handler in self.intercepts:
            before = exchange.digest()
            marks = len(self.trace.interceptions)
            action = invoke(handler, available)
            if inspect.isawaitable(action):
                action = await action
            record: InterceptRecord | None = None
            if action is None:
                # A mutation the handler (or a helper it called) already recorded speaks
                # for itself; an unrecorded one is auto-logged as a rewrite.
                if (
                    exchange.digest() != before
                    and len(self.trace.interceptions) == marks
                ):
                    record = InterceptRecord(
                        direction=direction,
                        handler=handler.__name__,
                        action="rewrite",
                        reason=handler.__name__,
                    )
            if isinstance(action, MESSAGE_TYPES):
                text = content_text(action.content)
                if direction == "request":
                    # Nothing to block inbound — a refused turn is the request-side block.
                    action = Terminate(reason=text)
                else:
                    snippet = _snippet(_response_tool_call_items(raw))
                    dropped = drop_response_tool_calls(raw, text)
                    record = InterceptRecord(
                        direction=direction,
                        handler=handler.__name__,
                        action="block",
                        target=", ".join(name for name in dropped if name),
                        reason=text,
                        before=snippet,
                    )
            if isinstance(action, Terminate):
                self.trace.record_interception(
                    InterceptRecord(
                        direction=direction,
                        handler=handler.__name__,
                        action="terminate",
                        reason=action.reason,
                        before=_snippet(raw),
                    )
                )
                if action.reward is not None:
                    self.trace.record_reward(
                        f"intercept/{handler.__name__}", action.reward, 1.0
                    )
                self.trace.stop(f"intercept/{handler.__name__}")
                logger.debug(
                    "intercept terminate: id=%s handler=%s",
                    self.trace.id,
                    handler.__name__,
                )
                return action
            if record is not None:
                self.trace.record_interception(record)
                logger.debug(
                    "intercept %s: id=%s handler=%s target=%s",
                    record.action,
                    self.trace.id,
                    handler.__name__,
                    record.target,
                )
        return None
