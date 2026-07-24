"""The per-rollout unit the interception layer serves.

One `RolloutSession` per rollout, registered on an interception server under the rollout's
secret. The rollout constructs it (model ctx, trace, task `@stop`s and `@intercept`s,
limits) and the server drives it: routes each intercepted model call to it, runs
`refused()` before each turn, and stashes the real failure on `error`. `RolloutLimits`
is the framework's per-rollout budget (turns / tokens), checked between turns.
"""

import asyncio
import inspect
import logging
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, get_args, get_type_hints

from verifiers.v1 import graph
from verifiers.v1.clients import ModelContext
from verifiers.v1.decorators import invoke
from verifiers.v1.errors import RolloutError, TaskError
from verifiers.v1.intercepts.core import (
    Direction,
    InterceptOutcome,
    InterceptRecord,
    Interceptor,
    PendingTermination,
    Terminate,
    snippet,
)
from verifiers.v1.trace import Trace
from verifiers.v1.types import (
    AssistantMessage,
    Messages,
    ToolMessage,
)

if TYPE_CHECKING:
    from verifiers.v1.dialects import Dialect

logger = logging.getLogger(__name__)

MESSAGE_TYPES = (AssistantMessage, ToolMessage)
RequestKey = tuple[str, bytes]


@dataclass(frozen=True, slots=True)
class StreamReplay:
    body: bytes
    content_type: str


ReplayResponse = dict | StreamReplay


def _message_types(handler: Callable[..., Any]) -> tuple[type, ...]:
    """Concrete message classes accepted by a handler's optional annotation."""
    hint = get_type_hints(handler, localns={"Trace": Trace}).get("message")
    accepted = tuple(
        kind for kind in (get_args(hint) or (hint,)) if kind in MESSAGE_TYPES
    )
    return accepted or MESSAGE_TYPES


def _directions(handler: Callable[..., Any]) -> tuple[Direction, ...]:
    if marked := getattr(handler, "intercept_directions", None):
        return marked
    accepted = _message_types(handler)
    if accepted == (AssistantMessage,):
        return ("response",)
    if accepted == (ToolMessage,):
        return ("request",)
    return ("request", "response")


def _handler_name(handler: Callable[..., Any]) -> str:
    return getattr(handler, "__name__", type(handler).__name__)


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
    intercepts: list[Interceptor] = field(default_factory=list)
    terminated: asyncio.Event = field(default_factory=asyncio.Event)
    termination_complete: asyncio.Event = field(default_factory=asyncio.Event)
    rewritten_response_ids: set[str] = field(default_factory=set)
    error: "RolloutError | None" = None
    """The latest unresolved model-call failure. The harness only sees it as an HTTP error
    (and may swallow it, or exit non-zero), so the rollout re-raises this original error once the
    harness returns — recording the real `ProviderError` instead of a secondary `HarnessError`.
    Reset before each model turn, so a successful retry clears it."""
    last_request: RequestKey | None = None
    """Route plus digest of the most recently served request; with `last_response`, the cache
    that keeps the message graph atomic under harness-SDK retries. A retry re-sends the
    byte-identical request; when it matches, the interception server replays the recorded
    response instead of re-sampling and committing a second turn — which would fork the graph
    into a dead-end branch. Only a fully served request is cached, so a genuinely failed attempt
    still re-runs. Turns are issued sequentially (one outstanding request at a time), so a retry
    is always of the most recent request — keeping only the last one is sufficient and bounded."""
    last_response: ReplayResponse | None = None
    """The response returned for `last_request`, replayed verbatim on a retry."""
    inflight: dict[RequestKey, "asyncio.Future[ReplayResponse | None]"] = field(
        default_factory=dict
    )
    """Route/body digest -> the future of the attempt currently computing it. A retry that arrives
    while the first attempt is still in flight (a slow turn) awaits this future instead of
    starting a second inference — the other half of retry atomicity (with `last_response`, which
    covers a retry after the attempt finished). Because a slow turn is coalesced rather than
    re-sampled, retries stay safe without an inflated client timeout. The future resolves to the
    served response, or to None if the attempt produced no servable response (error/refuse)."""
    released: bool = False
    """Set when the rollout unregisters the session: the trace is sealed (its conclusion is
    what scored and persisted), so a handler still in flight must not commit turns, record
    calls, or write state onto it — the in-memory trace must stay what the run produced."""
    tasks: set["asyncio.Task"] = field(default_factory=set)
    """Handler tasks currently serving this session. aiohttp does not cancel a handler when
    its client disconnects, so a request whose program died at teardown would keep driving
    the exchange (upstream call, simulator turn) — unregistering cancels these instead."""

    def adopt(self, task: "asyncio.Task | None") -> None:
        """Track a handler task serving this session, for cancellation at release.
        Callers adopt in the same synchronous stretch that fetched the session, so
        `release()` can't interleave; the released check keeps the seal even if a
        future caller breaks that invariant (an await before adopting)."""
        if task is None:
            return
        if self.released:  # sealed while this handler was scheduled — don't serve
            task.cancel()
            return
        self.tasks.add(task)
        task.add_done_callback(self.tasks.discard)

    def release(self) -> None:
        """Seal the session: no further trace mutation, and in-flight handlers cancel."""
        self.released = True
        for task in list(self.tasks):
            task.cancel()

    @property
    def has_response_intercepts(self) -> bool:
        """Whether a complete provider response must be classified before delivery."""
        return any("response" in _directions(handler) for handler in self.intercepts)

    def signal_termination(self, pending: PendingTermination) -> None:
        """Record a terminal policy result and wake the rollout lifecycle."""
        if self.terminated.is_set():
            return
        name = f"intercept/{pending.handler}"
        self.error = None
        self.trace.record_reward(name, pending.result.reward)
        self.trace.stop(name)
        self.terminated.set()

    async def refused(self) -> str | None:
        """The framework's limits (turns / token budget) and `@stop` checks, run before each
        model call. Sets the stop condition and returns its name, else None. A refused first
        call halts the harness (its model call errors out); Harness.run treats it as clean. A task
        that ends a trajectory from `trace.state` does it with its own `@stop` (run here generically),
        so the interception server holds no opinion about the state's contents."""
        if self.terminated.is_set():
            return self.trace.stop_condition or "intercepted"
        if (limit := self.limits.reached(self.trace)) is not None:
            self.trace.stop(limit)
            logger.debug("limit %r reached: id=%s", limit, self.trace.id)
            return limit
        for stop in self.stops:
            if await stop(self.trace):
                name = _handler_name(stop)
                self.trace.stop(name)
                logger.debug("stop %r fired: id=%s", name, self.trace.id)
                return name
        return None

    async def run_intercepts(
        self,
        direction: Direction,
        raw: dict,
        dialect: "Dialect",
        prompt: Messages | None = None,
    ) -> InterceptOutcome:
        """Run typed handlers in priority order, mutating only the native wire copy."""
        rewritten = False
        try:
            for handler in self.intercepts:
                if direction not in _directions(handler):
                    continue
                name = _handler_name(handler)
                if getattr(handler, "intercept_raw", False):
                    action = invoke(
                        handler,
                        {
                            "task": self.trace.task.data,
                            "trace": self.trace,
                            "raw": raw,
                            "dialect": dialect,
                        },
                    )
                    if inspect.isawaitable(action):
                        action = await action
                    if isinstance(action, str):
                        raise TypeError("a string result requires a message parameter")
                    if action is not None and not isinstance(action, Terminate):
                        raise TypeError(type(action).__name__)
                    if isinstance(action, Terminate):
                        self.trace.record_interception(
                            InterceptRecord(
                                direction=direction,
                                handler=name,
                                action="terminate",
                                reason=action.reason,
                                before=snippet(raw),
                                reward=action.reward,
                            )
                        )
                        return InterceptOutcome(
                            rewritten=rewritten,
                            termination=PendingTermination(name, action),
                        )
                    continue

                messages = prompt or []
                if direction == "request":
                    messages = dialect.parse_request(raw)[0]
                    tool_names = {
                        call.id: call.name
                        for item in [*self.trace.assistant_messages, *messages]
                        if isinstance(item, AssistantMessage)
                        for call in item.tool_calls or []
                    }
                    candidates = [
                        item.model_copy(
                            update={"name": tool_names.get(item.tool_call_id)}
                        )
                        if item.name is None and item.tool_call_id in tool_names
                        else item
                        for item in graph.prepare_turn(self.trace, messages).tail
                        if isinstance(item, ToolMessage)
                    ]
                else:
                    candidates = [
                        dialect.parse_response(dialect.validate_response(raw)).message
                    ]

                accepted = _message_types(handler)
                for message in candidates:
                    if not isinstance(message, accepted):
                        continue
                    action = invoke(
                        handler,
                        {
                            "task": self.trace.task.data,
                            "trace": self.trace,
                            "message": message.model_copy(deep=True),
                            "prompt": messages,
                        },
                    )
                    if inspect.isawaitable(action):
                        action = await action
                    if action is not None and not isinstance(action, (str, Terminate)):
                        raise TypeError(type(action).__name__)
                    target = (
                        ", ".join(call.name for call in message.tool_calls or [])
                        if isinstance(message, AssistantMessage)
                        else message.name or message.tool_call_id
                    )
                    if isinstance(action, Terminate):
                        self.trace.record_interception(
                            InterceptRecord(
                                direction=direction,
                                handler=name,
                                action="terminate",
                                target=target,
                                reason=action.reason,
                                before=snippet(message.model_dump(mode="json")),
                                reward=action.reward,
                            )
                        )
                        return InterceptOutcome(
                            rewritten=rewritten,
                            termination=PendingTermination(name, action),
                        )
                    if isinstance(action, str):
                        if isinstance(message, AssistantMessage):
                            dialect.rewrite_response(raw, action)
                        else:
                            dialect.rewrite_tool_result(
                                raw, message.tool_call_id, action
                            )
                        self.trace.record_interception(
                            InterceptRecord(
                                direction=direction,
                                handler=name,
                                action="rewrite",
                                target=target,
                                reason=name,
                                before=snippet(message.model_dump(mode="json")),
                            )
                        )
                        rewritten = True
        except RolloutError:
            raise
        except Exception as e:
            raise TaskError(f"@intercept failed: {type(e).__name__}: {e}") from e
        return InterceptOutcome(rewritten=rewritten)
