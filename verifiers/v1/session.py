"""The per-rollout unit the interception layer serves.

One `RolloutSession` per rollout, registered on an interception server under the rollout's
secret. The rollout constructs it (model ctx, trace, task `@stop`s, limits) and the server
drives it: routes each intercepted model call to it, runs `refused()` before each turn,
and stashes the real failure on `error`. `RolloutLimits` is the framework's per-rollout
budget (turns / tokens), checked between turns.
"""

import asyncio
import inspect
import logging
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from functools import cached_property

from pydantic import BaseModel, ConfigDict, TypeAdapter

from verifiers.v1 import graph
from verifiers.v1.clients import Client, ModelContext
from verifiers.v1.errors import RolloutError, TaskError
from verifiers.v1.intercepts.core import (
    Direction,
    InterceptDecision,
    Interceptor,
    InterceptRecord,
    Terminate,
)
from verifiers.v1.trace import Trace
from verifiers.v1.types import (
    AssistantMessage,
    Messages,
    Response,
    SystemMessage,
)
from verifiers.v1.utils.decorators import invoke

logger = logging.getLogger(__name__)


class StreamReplay(BaseModel):
    """The exact buffered SSE body retained for an explicitly keyed retry."""

    model_config = ConfigDict(frozen=True)

    content_type: str
    body: bytes


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
    client: Client
    trace: Trace
    stops: list[Callable[[Trace], Awaitable[bool]]] = field(default_factory=list)
    limits: RolloutLimits = field(default_factory=RolloutLimits)
    intercepts: list[Interceptor] = field(default_factory=list)
    terminated: bool = False
    stream_replays: dict[
        str, tuple[tuple[str, bytes], "asyncio.Future[StreamReplay | None]"]
    ] = field(default_factory=dict)
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
    released: bool = False
    """Set when the rollout unregisters the session: the trace is sealed (its conclusion is
    what scored and persisted), so a handler still in flight must not commit turns, record
    calls, or write state onto it — the in-memory trace must stay what the run produced."""
    tasks: set["asyncio.Task"] = field(default_factory=set)
    """Handler tasks currently serving this session. aiohttp does not cancel a handler when
    its client disconnects, so a request whose program died at teardown would keep driving
    the exchange (upstream call, simulator turn) — unregistering cancels these instead."""

    @cached_property
    def state_adapter(self) -> TypeAdapter:
        """The rollout's state codec, built only when a state channel is used."""
        return TypeAdapter(type(self.trace.state))

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
        for _, future in self.stream_replays.values():
            if not future.done():
                future.set_result(None)
        self.stream_replays.clear()
        for task in list(self.tasks):
            task.cancel()

    async def refused(self) -> str | None:
        """The framework's limits (turns / token budget) and `@stop` checks, run before each
        model call. Sets the stop condition and returns its name, else None. A refused first
        call halts the harness (its model call errors out); Harness.run treats it as clean. A task
        that ends a trajectory from `trace.state` does it with its own `@stop` (run here generically),
        so the interception server holds no opinion about the state's contents."""
        if self.terminated:
            return self.trace.stop_condition or "intercepted"
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

    def matching_intercepts(self, direction: Direction) -> list[Interceptor]:
        return [
            handler
            for handler in self.intercepts
            if direction in handler.intercept_directions
        ]

    async def intercept_request(
        self, trace: Trace, request: Messages
    ) -> InterceptDecision:
        """Run request hooks in priority order on ``request[-1]``."""
        if not request:
            return InterceptDecision()
        records: list[InterceptRecord] = []
        try:
            for handler in self.matching_intercepts("request"):
                current = request[-1]
                candidate = [current.model_copy(deep=True)]
                result = invoke(handler, {"trace": trace, "request": candidate})
                if inspect.isawaitable(result):
                    result = await result
                name = getattr(handler, "__name__", type(handler).__name__)
                if isinstance(result, Terminate):
                    return InterceptDecision(
                        records=records,
                        termination=InterceptRecord(
                            direction="request",
                            handler=name,
                            action="terminate",
                            reason=result.reason,
                            reward=result.reward,
                        ),
                    )
                if result is None or result == current:
                    continue
                if type(result) is not type(current):
                    raise TypeError(
                        f"request hooks must return {type(current).__name__}, "
                        "vf.Terminate, or None"
                    )
                request[-1] = result
                records.append(
                    InterceptRecord(
                        direction="request",
                        handler=name,
                        action="rewrite",
                    )
                )
                trace.nodes[-1].message = result
        except RolloutError:
            raise
        except Exception as e:
            raise TaskError(f"@on_request failed: {type(e).__name__}: {e}") from e
        return InterceptDecision(records=records)

    async def intercept_response(
        self, trace: Trace, response: Response
    ) -> tuple[InterceptDecision, str | None]:
        """Run response hooks and reduce a replacement to one inert text message."""
        records: list[InterceptRecord] = []
        try:
            for handler in self.matching_intercepts("response"):
                result = invoke(
                    handler,
                    {"trace": trace, "response": response.model_copy(deep=True)},
                )
                if inspect.isawaitable(result):
                    result = await result
                name = getattr(handler, "__name__", type(handler).__name__)
                if isinstance(result, Terminate):
                    if records:
                        response.finish_reason = "stop"
                        response.tokens = None
                    return (
                        InterceptDecision(
                            records=records,
                            termination=InterceptRecord(
                                direction="response",
                                handler=name,
                                action="terminate",
                                reason=result.reason,
                                reward=result.reward,
                            ),
                        ),
                        None,
                    )
                if result is None or result == response.message:
                    continue
                if not isinstance(result, AssistantMessage):
                    raise TypeError(
                        "response hooks must return vf.AssistantMessage, "
                        "vf.Terminate, or None"
                    )
                response.message = result
                records.append(
                    InterceptRecord(
                        direction="response", handler=name, action="rewrite"
                    )
                )
                trace.nodes[-1].message = response.message
        except RolloutError:
            raise
        except Exception as e:
            raise TaskError(f"@on_response failed: {type(e).__name__}: {e}") from e
        decision = InterceptDecision(records=records)
        if not records:
            return decision, None
        message = response.message
        if (
            not isinstance(message, AssistantMessage)
            or not isinstance(message.content, str)
            or message.reasoning_content
            or message.tool_calls
            or message.provider_tools
            or message.provider_state
        ):
            raise TaskError(
                "response interception must produce text-only assistant output"
            )
        response.finish_reason = "stop"
        response.tokens = None
        return decision, message.content

    async def intercept_users(self, messages: Messages) -> Messages:
        """Run request hooks on a new user turn before the harness starts."""
        if not self.matching_intercepts("request"):
            return messages
        branch = self.trace.branches[-1].messages if self.trace.branches else []
        if not branch and self.trace.task.data.system_prompt is not None:
            branch = [SystemMessage(content=self.trace.task.data.system_prompt)]
        request = list(messages)
        for index, message in enumerate(request):
            current = [message]
            turn = graph.prepare_turn(self.trace, [*branch, *request[: index + 1]])
            decision = await self.intercept_request(turn.scoped_trace(), current)
            request[index] = current[-1]
            self.trace.interceptions.extend(decision.records)
            if decision.termination is not None:
                graph.prepare_turn(
                    self.trace, [*branch, *request[: index + 1]]
                ).commit_prompt()
                self.terminate(decision.termination)
                break
        return request

    def terminate(self, record: InterceptRecord) -> None:
        """Store a terminal decision so every later exchange is refused."""
        if self.terminated:
            return
        self.trace.interceptions.append(record)
        self.trace.record_reward(f"intercept/{record.handler}", record.reward or 0.0)
        self.trace.stop_condition = record.reason
        self.terminated = True
