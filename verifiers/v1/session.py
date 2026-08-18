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
from typing import get_origin, get_type_hints

from pydantic import TypeAdapter

from verifiers.v1 import graph
from verifiers.v1.clients import Client, ModelContext
from verifiers.v1.configs.runtime import NetworkPolicyConfig
from verifiers.v1.errors import HarnessError, RolloutError, TaskError
from verifiers.v1.trace import InterceptRecord, Trace
from verifiers.v1.types import (
    AssistantMessage,
    Request,
    Response,
    ToolMessage,
    UserMessage,
)
from verifiers.v1.utils.decorators import invoke

logger = logging.getLogger(__name__)


def hook_boundary(handler: Callable, *, allow_trace: bool) -> type:
    """Select a hook boundary solely from its annotated parameters."""
    hints = get_type_hints(handler)
    annotations = [
        get_origin(hints[name]) or hints[name]
        for name in inspect.signature(handler).parameters
        if name in hints
    ]
    boundaries = [kind for kind in annotations if kind in (Request, Response)]
    if len(boundaries) == 1 and annotations.count(Trace) <= 1:
        return boundaries[0]
    if not boundaries and allow_trace and annotations.count(Trace) == 1:
        return Trace
    expected = "Request, Response, or Trace" if allow_trace else "Request or Response"
    raise TypeError(f"{handler.__name__} must have exactly one {expected} parameter")


async def call_hook(handler: Callable, available: dict[type, object]) -> object:
    result = invoke(handler, available)
    return await result if inspect.isawaitable(result) else result


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
class IdempotentRequest:
    """One non-streaming model request shared by its original call and retries."""

    binding: tuple[str, bytes]
    response: "ReplayResponse | None" = None
    completed_at: float | None = None
    inflight: "asyncio.Future[ReplayResponse | None] | None" = None


@dataclass(frozen=True)
class ReplayResponse:
    """The exact HTTP result handed to coalesced in-flight request attempts."""

    status: int
    body: bytes


@dataclass
class RolloutSession:
    ctx: ModelContext
    trace: Trace
    network_policy: NetworkPolicyConfig = field(default_factory=NetworkPolicyConfig)
    """The resolved execution policy, including task-level restrictions."""
    trace_stops: list[Callable[..., Awaitable[bool] | bool]] = field(
        default_factory=list
    )
    limits: RolloutLimits = field(default_factory=RolloutLimits)
    request_interceptors: list[Callable] = field(default_factory=list)
    response_interceptors: list[Callable] = field(default_factory=list)
    request_stops: list[Callable] = field(default_factory=list)
    response_stops: list[Callable] = field(default_factory=list)
    native_tool_interception: bool = False
    """Buffer streamed model turns until their assistant node is committed, so a
    native pre-execution hook cannot outrun the graph entry that identifies its call."""
    client: Client | None = None
    """The model client serving this rollout's turns. The interception server assigns it at
    `register` (one server-owned client per distinct endpoint config), so every rollout it
    multiplexes shares one keepalive connection pool instead of opening its own."""
    error: "RolloutError | None" = None
    """The latest unresolved model-call failure. The harness only sees it as an HTTP error, so
    when its program dies on it the rollout records this original error instead of a secondary
    `HarnessError`. A harness that completes cleanly after the failure handled it. Reset before
    each model turn, so a successful retry clears it."""
    idempotent_requests: dict[str, IdempotentRequest] = field(default_factory=dict)
    """Explicit keys or marked SDK retries mapped to their replay state."""
    fatal_error: "RolloutError | None" = None
    """A tool-boundary failure. Unlike retryable model errors, a later model call cannot
    clear it because the native agent may continue after rejecting a tool permission."""
    released: bool = False
    """Set when the rollout unregisters the session: the trace is sealed (its conclusion is
    what scored and persisted), so a handler still in flight must not commit turns, record
    calls, or write state onto it — the in-memory trace must stay what the run produced."""
    tasks: set["asyncio.Task"] = field(default_factory=set)
    """Handler tasks currently serving this session. aiohttp does not cancel a handler when
    its client disconnects, so a request whose program died at teardown would keep driving
    the exchange (upstream call, simulator turn) — unregistering cancels these instead."""
    prepared_tool_results: dict[tuple[int, str], ToolMessage | None] = field(
        default_factory=dict
    )
    """Native call state keyed by issuing assistant node and call ID: None while an
    allowed call awaits its post-execution hook, then its intercepted result."""
    tool_interception_lock: asyncio.Lock = field(
        default_factory=asyncio.Lock, repr=False
    )
    """Native agents may finish sibling tools concurrently; serialize trace mutation."""
    prepared_messages: Counter[str] = field(default_factory=Counter)

    @property
    def stopped(self) -> bool:
        return self.trace.stop_condition is not None

    async def rewrite_request(
        self,
        request: Request,
        *,
        run_stops: bool = True,
        require_native_results: bool = True,
    ) -> tuple[Request, list[InterceptRecord], str | None]:
        """Run typed request interceptors and stops over one canonical request."""
        if not self.request_interceptors and (not run_stops or not self.request_stops):
            return request, [], None
        current = request
        turn = graph.prepare_turn(self.trace, request.messages)
        tail_start = turn.tail_start
        assistant_node = (
            turn.prefix_node_ids[-1]
            if turn.prefix_node_ids
            and isinstance(
                self.trace.nodes[turn.prefix_node_ids[-1]].message, AssistantMessage
            )
            else None
        )
        # Provider-only mediation can change an earlier user message without changing
        # the harness's transcript. Its canonical current assistant still anchors the tail.
        if assistant_node is None and self.network_policy.network_restricted:
            leaves = graph.leaves(self.trace)
            for position in range(len(current.messages) - 1, -1, -1):
                message = current.messages[position]
                if not isinstance(message, AssistantMessage):
                    continue
                matches = [
                    leaf
                    for leaf in leaves
                    if graph.message_hash(self.trace.nodes[leaf].message)
                    == graph.message_hash(message)
                ]
                if len(matches) == 1:
                    assistant_node = matches[0]
                    tail_start = position + 1
                    break
        prepared_messages = self.prepared_messages.copy()
        prepared: set[int] = set()
        native_prepared: set[int] = set()
        candidates: set[int] = set()
        for position in range(tail_start, len(current.messages)):
            message = current.messages[position]
            if isinstance(message, (UserMessage, ToolMessage)):
                candidates.add(position)
                key = graph.message_hash(message)
                if prepared_messages[key]:
                    prepared_messages[key] -= 1
                    prepared.add(position)
                    continue
            if isinstance(message, ToolMessage):
                prepared_result = (
                    self.prepared_tool_results.get(
                        (assistant_node, message.tool_call_id)
                    )
                    if assistant_node is not None
                    else None
                )
                is_prepared = (
                    prepared_result == message
                    # Native hooks include the tool name, but Anthropic and Responses
                    # omit it when they parse the result into the next model request.
                    or prepared_result is not None
                    and message.name is None
                    and prepared_result.model_copy(update={"name": None}) == message
                )
                candidates.add(position)
                if is_prepared:
                    prepared.add(position)
                    native_prepared.add(position)
                elif prepared_result is not None:
                    raise HarnessError(
                        "native tool interception did not preserve the approved result "
                        f"for call {message.tool_call_id!r}"
                    )
                elif (
                    self.native_tool_interception
                    and require_native_results
                    and assistant_node is not None
                ):
                    raise HarnessError(
                        "native tool result reached the model request before its "
                        f"post-execution hook for call {message.tool_call_id!r}"
                    )
        if not candidates and (not run_stops or not self.request_stops):
            return request, [], None
        already_intercepted = candidates == prepared
        if (
            candidates
            and candidates == native_prepared
            and all(
                isinstance(current.messages[position], ToolMessage)
                for position in candidates
            )
        ):
            # The native hook already ran both interceptors and stops. This request only
            # commits the result that the harness admitted to its next model turn.
            return request, [], None

        records: list[InterceptRecord] = []
        try:
            interceptors = [] if already_intercepted else self.request_interceptors
            for handler in interceptors:
                candidate = current.model_copy(deep=True)
                result = await call_hook(
                    handler, {Request: candidate, Trace: self.trace}
                )
                if result is None:
                    continue
                if not isinstance(result, Request):
                    raise TypeError(f"expected Request, got {type(result).__name__}")
                if len(result.messages) != len(current.messages):
                    raise ValueError(
                        "request interceptors cannot add or remove messages"
                    )
                if result.tools != current.tools:
                    raise ValueError("request interceptors cannot rewrite tools")
                for position, (before, after) in enumerate(
                    zip(current.messages, result.messages, strict=True)
                ):
                    if before == after:
                        continue
                    if position not in candidates - prepared:
                        raise ValueError(
                            "request interceptors can only rewrite new user or tool messages"
                        )
                    if type(after) is not type(before):
                        raise TypeError(
                            f"expected {type(before).__name__}, got {type(after).__name__}"
                        )
                    if isinstance(before, ToolMessage) and (
                        after.tool_call_id != before.tool_call_id
                        or after.name != before.name
                    ):
                        raise ValueError(
                            "request interceptors cannot change a tool-call ID or name"
                        )
                if result != current:
                    current = result
                    records.append(InterceptRecord(handler=handler.__name__))

            stops = self.request_stops if run_stops else []
            for stop in stops:
                candidate = current.model_copy(deep=True)
                result = await call_hook(stop, {Request: candidate, Trace: self.trace})
                if not isinstance(result, bool):
                    raise TypeError(
                        f"@stop must return bool, got {type(result).__name__}"
                    )
                if result:
                    return current, records, stop.__name__
        except RolloutError:
            raise
        except Exception as error:
            raise TaskError(
                f"request interception failed: {type(error).__name__}: {error}"
            ) from error
        return current, records, None

    def consume_prepared(self, turn: graph.PendingTurn) -> None:
        """Forget pre-harness rewrites only after their model request commits."""
        assistant_node = (
            turn.prefix_node_ids[-1]
            if turn.prefix_node_ids
            and isinstance(
                self.trace.nodes[turn.prefix_node_ids[-1]].message, AssistantMessage
            )
            else None
        )
        if assistant_node is None and self.network_policy.network_restricted:
            leaves = graph.leaves(self.trace)
            for message in reversed(turn.tail):
                if not isinstance(message, AssistantMessage):
                    continue
                matches = [
                    leaf
                    for leaf in leaves
                    if graph.message_hash(self.trace.nodes[leaf].message)
                    == graph.message_hash(message)
                ]
                if len(matches) == 1:
                    assistant_node = matches[0]
                    break
        for message in turn.tail:
            if isinstance(message, (UserMessage, ToolMessage)):
                key = graph.message_hash(message)
                if self.prepared_messages[key]:
                    self.prepared_messages[key] -= 1
            if isinstance(message, ToolMessage) and assistant_node is not None:
                tool_call = (assistant_node, message.tool_call_id)
                self.prepared_tool_results.pop(tool_call, None)

    async def prepare_users(
        self, request: Request
    ) -> tuple[Request, list[InterceptRecord]]:
        """Intercept caller-owned user turns before the harness stores them."""
        branch = self.trace.messages
        rewritten, records, _ = await self.rewrite_request(
            Request(messages=[*branch, *request.messages]),
            run_stops=False,
            require_native_results=False,
        )
        tail = rewritten.messages[len(branch) :]
        self.prepared_messages.update(
            graph.message_hash(message)
            for message in tail
            if isinstance(message, (UserMessage, ToolMessage))
        )
        return Request(messages=tail), records

    async def rewrite_response(
        self, response: Response
    ) -> tuple[Response, list[InterceptRecord], str | None]:
        """Run typed response interceptors and stops before harness delivery."""
        records: list[InterceptRecord] = []
        try:
            for handler in self.response_interceptors:
                candidate = response.model_copy(deep=True)
                result = await call_hook(
                    handler, {Response: candidate, Trace: self.trace}
                )
                if result is None:
                    continue
                if not isinstance(result, Response):
                    raise TypeError(f"expected Response, got {type(result).__name__}")
                if result == response:
                    continue
                unchanged = result.model_copy(
                    update={
                        "message": response.message,
                        "finish_reason": response.finish_reason,
                    }
                )
                if unchanged != response:
                    raise ValueError(
                        "response interceptors can only replace the assistant message"
                    )
                if (
                    result.message.reasoning_content
                    or result.message.tool_calls
                    or result.message.provider_state
                ):
                    raise ValueError(
                        "response interceptors must return an inert text-only message"
                    )
                response = result.model_copy(update={"finish_reason": "stop"})
                records.append(InterceptRecord(handler=handler.__name__))

            for stop in self.response_stops:
                candidate = response.model_copy(deep=True)
                result = await call_hook(stop, {Response: candidate, Trace: self.trace})
                if not isinstance(result, bool):
                    raise TypeError(
                        f"@stop must return bool, got {type(result).__name__}"
                    )
                if result:
                    return response, records, stop.__name__
        except RolloutError:
            raise
        except Exception as error:
            raise TaskError(
                f"response interception failed: {type(error).__name__}: {error}"
            ) from error
        return response, records, None

    async def handle_tool(
        self,
        phase: str,
        message: ToolMessage,
        content: str = "any",
    ) -> dict:
        """Run native tool policy before execution or before the next model turn."""
        leaves = graph.leaves(self.trace)
        matches = [
            leaf
            for leaf in leaves
            if isinstance(self.trace.nodes[leaf].message, AssistantMessage)
            and any(
                call.id == message.tool_call_id
                for call in self.trace.nodes[leaf].message.tool_calls or []
            )
        ]
        if len(matches) != 1:
            raise HarnessError(
                f"{phase} tool call {message.tool_call_id!r} matched "
                f"{len(matches)} branches"
            )
        assistant_node = matches[0]
        tool_call = (assistant_node, message.tool_call_id)
        if phase != "before":
            if tool_call not in self.prepared_tool_results:
                raise HarnessError(
                    f"tool call {message.tool_call_id!r} reached a post-execution hook "
                    "without crossing its pre-execution hook"
                )
            if self.prepared_tool_results.pop(tool_call) is not None:
                raise HarnessError(
                    f"harness reported tool call {message.tool_call_id!r} after its "
                    "pre-execution result was replaced"
                )
        assistant = self.trace.nodes[assistant_node].message
        assert isinstance(assistant, AssistantMessage)
        tool_name = next(
            call.name
            for call in assistant.tool_calls or []
            if call.id == message.tool_call_id
        )
        if message.name != tool_name:
            raise HarnessError(
                f"native hook reported tool {message.name!r}, expected {tool_name!r}"
            )
        branch = []
        node: int | None = assistant_node
        while node is not None:
            branch.append(self.trace.nodes[node].message)
            node = self.trace.nodes[node].parent
        branch.reverse()
        # Keep earlier results in the hook's trace, with the active result last so request
        # hooks can identify it. Commit them only when token attribution becomes available.
        previous: list[ToolMessage] = []
        for call in assistant.tool_calls or []:
            if call.id == message.tool_call_id:
                continue
            result = self.prepared_tool_results.get((assistant_node, call.id))
            if result is not None:
                previous.append(result)
        request, records, stopped = await self.rewrite_request(
            Request(
                messages=[*branch, *previous, message],
                tools=self.trace.tools or None,
            ),
            require_native_results=False,
        )
        if self.released:
            raise HarnessError("rollout concluded during tool interception")
        candidate = request.messages[-1]
        assert isinstance(candidate, ToolMessage)
        # Pi's transports reshape image results before the next model request, so only
        # text has a stable identity that the hook and canonical trace can both verify.
        if (
            stopped is None
            and content == "nonempty_text"
            and (phase != "before" or candidate != message)
            and (not isinstance(candidate.content, str) or not candidate.content)
        ):
            raise HarnessError(
                "this native hook can only preserve non-empty text tool results"
            )
        self.trace.request_rewrites.extend(records)
        if stopped is not None:
            results = {
                result.tool_call_id: result
                for result in request.messages[len(branch) :]
                if isinstance(result, ToolMessage)
            }
            if phase == "before":
                results.pop(message.tool_call_id)
            committed = [
                *request.messages[: len(branch)],
                *(
                    results[call.id]
                    for call in assistant.tool_calls or []
                    if call.id in results
                ),
            ]
            turn = graph.prepare_turn(self.trace, committed)
            turn.commit_prompt()
            self.consume_prepared(turn)
            self.trace.stop(stopped)
            return {"action": "stop", "reason": stopped}
        if phase == "before" and candidate == message:
            self.prepared_tool_results.setdefault(tool_call, None)
            return {"action": "allow"}
        self.prepared_tool_results[tool_call] = candidate
        if candidate != message:
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
        for stop in self.trace_stops:
            result = await call_hook(stop, {Trace: self.trace})
            if not isinstance(result, bool):
                raise TaskError(f"@stop must return bool, got {type(result).__name__}")
            if result:
                self.trace.stop(stop.__name__)
                logger.debug("stop %r fired: id=%s", stop.__name__, self.trace.id)
                return stop.__name__
        return None
