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
from pathlib import Path
from typing import TYPE_CHECKING, Any, get_args, get_type_hints

from verifiers.v1 import graph
from verifiers.v1.clients import ModelContext
from verifiers.v1.decorators import invoke
from verifiers.v1.errors import RolloutError, TaskError
from verifiers.v1.intercepts.core import (
    Direction,
    Interceptor,
    InterceptRecord,
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
RequestKey = tuple[str, bytes, str]


@dataclass(frozen=True, slots=True)
class StreamReplay:
    path: Path
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
    if getattr(handler, "intercept_raw", False):
        raise TypeError("raw @intercept requires an explicit direction")
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
    rewritten_response_ids: set[str] = field(default_factory=set)
    error: "RolloutError | None" = None
    """The latest unresolved model-call failure. The harness only sees it as an HTTP error
    (and may swallow it, or exit non-zero), so the rollout re-raises this original error once the
    harness returns — recording the real `ProviderError` instead of a secondary `HarnessError`.
    Reset before each model turn, so a successful retry clears it."""
    replays: dict[RequestKey, ReplayResponse] = field(default_factory=dict)
    """Completed responses keyed by route, body digest, and Idempotency-Key. Explicitly keyed
    retries replay the matching response; unkeyed requests never enter this time-bounded cache."""
    replay_expirations: dict[RequestKey, asyncio.TimerHandle] = field(
        default_factory=dict
    )
    """Expiry timers for completed replay responses."""
    inflight: dict[RequestKey, "asyncio.Future[ReplayResponse | None]"] = field(
        default_factory=dict
    )
    """Request key -> the future of the attempt currently computing it. A retry that arrives
    while the first attempt is still in flight (a slow turn) awaits this future instead of
    starting a second inference — the other half of retry atomicity (with `replays`, which covers
    a retry after the attempt finished). Because a slow turn is coalesced rather than
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
        for expiration in self.replay_expirations.values():
            expiration.cancel()
        self.replay_expirations.clear()
        for response in self.replays.values():
            if isinstance(response, StreamReplay):
                response.path.unlink(missing_ok=True)
        self.replays.clear()
        for task in list(self.tasks):
            task.cancel()

    @property
    def has_response_intercepts(self) -> bool:
        """Whether a complete provider response must be classified before delivery."""
        return any("response" in _directions(handler) for handler in self.intercepts)

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
    ) -> bool:
        """Run matching handlers in priority order against the native wire object."""
        rewritten = False
        try:
            for handler in self.intercepts:
                if direction not in _directions(handler):
                    continue
                name = _handler_name(handler)
                raw_handler = getattr(handler, "intercept_raw", False)
                messages = (
                    dialect.parse_request(raw)[0]
                    if direction == "request"
                    else prompt or []
                )
                if raw_handler:
                    candidates = [None]
                elif direction == "request":
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

                for message in candidates:
                    if message is not None and not isinstance(
                        message, _message_types(handler)
                    ):
                        continue
                    action = invoke(
                        handler,
                        {
                            "task": self.trace.task.data,
                            "trace": self.trace,
                            "raw": raw,
                            "dialect": dialect,
                            "message": message.model_copy(deep=True)
                            if message is not None
                            else None,
                            "prompt": messages,
                        },
                    )
                    if inspect.isawaitable(action):
                        action = await action
                    if raw_handler:
                        if action is not None and action is not raw:
                            raise TypeError(type(action).__name__)
                        if action is raw:
                            self.trace.interceptions.append(
                                InterceptRecord(
                                    direction=direction,
                                    handler=name,
                                    action="rewrite",
                                )
                            )
                            rewritten = True
                        continue
                    if action is not None and not isinstance(action, str):
                        raise TypeError(type(action).__name__)
                    if isinstance(action, str):
                        if message is None:
                            raise TypeError(
                                "a string result requires a message parameter"
                            )
                        if isinstance(message, AssistantMessage):
                            dialect.rewrite_response(raw, action)
                        else:
                            dialect.rewrite_tool_result(
                                raw, message.tool_call_id, action
                            )
                        self.trace.interceptions.append(
                            InterceptRecord(
                                direction=direction, handler=name, action="rewrite"
                            )
                        )
                        rewritten = True
        except RolloutError:
            raise
        except Exception as e:
            raise TaskError(f"@intercept failed: {type(e).__name__}: {e}") from e
        return rewritten
