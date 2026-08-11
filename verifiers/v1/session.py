"""The per-rollout unit the interception layer serves.

One `RolloutSession` per rollout, registered on an interception server under the rollout's
secret. The rollout constructs it (model ctx, trace, task `@stop`s, limits) and the server
drives it: assigns its model client at register, routes each intercepted model call to it,
runs `refused()` before each turn, and stashes the real failure on `error`. `RolloutLimits` is the framework's per-rollout
budget (turns / tokens), checked between turns.
"""

import asyncio
import inspect
import logging
from collections import Counter
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from functools import cached_property

from pydantic import TypeAdapter

from verifiers.v1 import graph
from verifiers.v1.clients import Client, ModelContext
from verifiers.v1.errors import RolloutError, TaskError
from verifiers.v1.on_request import RequestHandler, RequestRewrite
from verifiers.v1.on_response import ResponseHandler, ResponseRewrite
from verifiers.v1.terminate import Terminate, Termination
from verifiers.v1.trace import Trace
from verifiers.v1.types import (
    AssistantMessage,
    Messages,
    Response,
    ToolMessage,
    UserMessage,
)
from verifiers.v1.utils.decorators import invoke

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
    request_handlers: list[RequestHandler] = field(default_factory=list)
    response_handlers: list[ResponseHandler] = field(default_factory=list)
    supports_tool_rewrite: bool = False
    client: Client | None = None
    """The model client serving this rollout's turns. The interception server assigns it at
    `register` (one server-owned client per distinct endpoint config), so every rollout it
    multiplexes shares one keepalive connection pool instead of opening its own."""
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
    pending_tool_results: dict[str, ToolMessage] = field(default_factory=dict)
    prepared_tool_results: dict[str, ToolMessage] = field(default_factory=dict)
    prepared_users: Counter[str] = field(default_factory=Counter)

    @property
    def terminated(self) -> bool:
        return self.trace.termination is not None

    def terminate(self, termination: Termination) -> None:
        """Apply one hook termination to the canonical trace."""
        if self.terminated:
            return
        self.trace.rewards.clear()
        self.trace.record_reward(f"terminate/{termination.handler}", termination.reward)
        self.trace.termination = termination
        self.trace.stop(termination.reason)

    async def rewrite_request(
        self, messages: Messages, *, from_position: int | None = None
    ) -> tuple[Messages, list[RequestRewrite], Termination | None]:
        """Run request hooks sequentially over each new user or tool message."""
        if not self.request_handlers:
            return messages, [], None
        turn = graph.prepare_turn(self.trace, messages)
        rewritten = list(messages)
        records: list[RequestRewrite] = []
        try:
            start = (
                turn.tail_start
                if from_position is None
                else max(turn.tail_start, from_position)
            )
            for position in range(start, len(rewritten)):
                original = rewritten[position]
                if not isinstance(original, (UserMessage, ToolMessage)):
                    continue
                if isinstance(original, UserMessage):
                    key = graph.message_hash(original)
                    if self.prepared_users[key]:
                        self.prepared_users[key] -= 1
                        continue
                if isinstance(original, ToolMessage):
                    prepared = self.prepared_tool_results.get(original.tool_call_id)
                    if prepared is not None and prepared.content == original.content:
                        continue
                for handler in self.request_handlers:
                    view = turn.scoped_trace(rewritten, end=position + 1)
                    result = invoke(handler, {"request": view.messages, "trace": view})
                    if inspect.isawaitable(result):
                        result = await result
                    if result is None:
                        continue
                    name = getattr(handler, "__name__", type(handler).__name__)
                    if isinstance(result, Terminate):
                        self.pending_tool_results.clear()
                        return (
                            rewritten[: position + 1],
                            records,
                            Termination(
                                **result.model_dump(),
                                handler=name,
                                boundary="request",
                            ),
                        )
                    if isinstance(result, str):
                        result = original.model_copy(update={"content": result})
                    if not isinstance(result, type(original)):
                        raise TypeError(
                            f"expected {type(original).__name__}, got "
                            f"{type(result).__name__}"
                        )
                    if isinstance(result, ToolMessage):
                        assert isinstance(original, ToolMessage)
                        if result.tool_call_id != original.tool_call_id:
                            raise ValueError(
                                "request hooks cannot change a tool-call ID"
                            )
                    if result == original:
                        continue
                    rewritten[position] = original = result
                    records.append(
                        RequestRewrite(
                            handler=name,
                            target=original.role,
                        )
                    )
        except RolloutError:
            raise
        except Exception as error:
            raise TaskError(
                f"@on_request failed: {type(error).__name__}: {error}"
            ) from error
        return rewritten, records, None

    def consume_prepared_tool_results(self, messages: Messages) -> None:
        """Forget hook-prepared tool results after their model request commits."""
        for message in messages:
            if isinstance(message, ToolMessage):
                self.prepared_tool_results.pop(message.tool_call_id, None)

    async def prepare_users(
        self, messages: Messages
    ) -> tuple[Messages, list[RequestRewrite]]:
        """Rewrite caller-owned user turns before the harness stores them."""
        branch = self.trace.messages
        rewritten, records, termination = await self.rewrite_request(
            [*branch, *messages]
        )
        tail = rewritten[len(branch) :]
        if termination is not None:
            graph.prepare_turn(self.trace, rewritten).commit_prompt()
            self.terminate(termination)
        self.prepared_users.update(
            graph.message_hash(message)
            for message in tail
            if isinstance(message, UserMessage)
        )
        return tail, records

    async def rewrite_response(
        self, response: Response, turn: graph.PendingTurn
    ) -> tuple[Response, list[ResponseRewrite], Termination | None]:
        """Run response hooks sequentially over the harness-visible response."""
        records: list[ResponseRewrite] = []
        try:
            for handler in self.response_handlers:
                view = turn.scoped_trace(response=response.message)
                result = invoke(handler, {"response": response, "trace": view})
                if inspect.isawaitable(result):
                    result = await result
                if result is None:
                    continue
                name = getattr(handler, "__name__", type(handler).__name__)
                if isinstance(result, Terminate):
                    self.pending_tool_results.clear()
                    return (
                        response,
                        records,
                        Termination(
                            **result.model_dump(),
                            handler=name,
                            boundary="response",
                        ),
                    )
                if isinstance(result, str):
                    result = AssistantMessage(content=result)
                if isinstance(result, ToolMessage):
                    if not self.supports_tool_rewrite:
                        raise ValueError(
                            "this harness cannot inject synthetic tool results; "
                            "return vf.Terminate to block the call"
                        )
                    calls = {
                        call.id: call.name for call in response.message.tool_calls or []
                    }
                    if result.tool_call_id not in calls:
                        raise ValueError(
                            "a synthetic result must reference this response's tool call"
                        )
                    self.pending_tool_results[result.tool_call_id] = result
                    target = calls[result.tool_call_id]
                elif isinstance(result, AssistantMessage):
                    if (
                        result.reasoning_content
                        or result.tool_calls
                        or result.provider_state
                    ):
                        raise ValueError(
                            "response rewrites must be an inert text-only message"
                        )
                    response = response.model_copy(
                        update={"message": result, "finish_reason": "stop"}
                    )
                    self.pending_tool_results.clear()
                    target = "assistant"
                else:
                    raise TypeError(
                        "response hooks must return AssistantMessage, ToolMessage, "
                        "str, or None"
                    )
                records.append(
                    ResponseRewrite(
                        handler=name,
                        target=target,
                    )
                )
        except RolloutError:
            raise
        except Exception as error:
            raise TaskError(
                f"@on_response failed: {type(error).__name__}: {error}"
            ) from error
        return response, records, None

    async def handle_tool(
        self, phase: str, message: ToolMessage, can_rewrite: bool
    ) -> dict:
        """Return the model-visible result before a supporting harness records it."""
        synthetic = self.pending_tool_results.pop(message.tool_call_id, None)
        if phase == "before" and synthetic is None:
            return {"action": "allow"}
        candidate = synthetic or message
        branches = [
            branch
            for branch in self.trace.branches
            if branch.nodes
            and isinstance(branch.nodes[-1].message, AssistantMessage)
            and any(
                call.id == message.tool_call_id
                for call in branch.nodes[-1].message.tool_calls or []
            )
        ]
        if len(branches) != 1:
            raise TaskError(
                f"tool call {message.tool_call_id!r} matched {len(branches)} branches"
            )
        branch = branches[0]
        assistant = branch.nodes[-1].message
        assert isinstance(assistant, AssistantMessage)
        # Keep earlier results in the hook's trace, but commit them only when the model
        # request arrives and can supply their token attribution.
        previous = [
            self.prepared_tool_results[call.id]
            for call in assistant.tool_calls or []
            if call.id in self.prepared_tool_results
        ]
        prompt = [*branch.messages, *previous, candidate]
        rewritten, records, termination = await self.rewrite_request(
            prompt, from_position=len(prompt) - 1
        )
        candidate = rewritten[-1]
        assert isinstance(candidate, ToolMessage)
        self.trace.request_rewrites.extend(records)
        if termination is None and candidate != message and not can_rewrite:
            termination = Termination(
                reason="harness cannot replace this tool result",
                handler=(records or self.trace.response_rewrites)[-1].handler,
                boundary="request",
            )
        if termination is not None:
            graph.prepare_turn(self.trace, [*prompt[:-1], candidate]).commit_prompt()
            self.terminate(termination)
            return {"action": "stop", "reason": termination.reason}
        self.prepared_tool_results[candidate.tool_call_id] = candidate
        if synthetic is not None or candidate != message:
            return {
                "action": "rewrite",
                "message": candidate.model_dump(exclude_none=True),
            }
        return {"action": "allow"}

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
        for task in list(self.tasks):
            task.cancel()

    async def refused(self) -> str | None:
        """The framework's limits (turns / token budget) and `@stop` checks, run before each
        model call. Sets the stop condition and returns its name, else None. A refused first
        call halts the harness (its model call errors out); HarnessSession.turn treats it as clean. A task
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
