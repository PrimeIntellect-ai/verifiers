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
from verifiers.v1.interception.tool import ToolRewriteCapabilities
from verifiers.v1.trace import InterceptRecord, ToolPolicyEvent, Trace
from verifiers.v1.types import (
    AssistantMessage,
    ImageUrlContentPart,
    Request,
    Response,
    TextContentPart,
    ToolMessage,
    UserMessage,
)
from verifiers.v1.utils.decorators import invoke

logger = logging.getLogger(__name__)

JS_TRIM_CHARACTERS = (
    "\u0009\u000b\u000c\u0020\u00a0\u1680\u2000\u2001\u2002\u2003\u2004"
    "\u2005\u2006\u2007\u2008\u2009\u200a\u202f\u205f\u3000\ufeff\n\r\u2028\u2029"
)


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
    prepared_tool_results: dict[tuple[int, str], ToolMessage] = field(
        default_factory=dict
    )
    """Native results already intercepted, keyed by issuing assistant node and call ID."""
    blocked_tool_calls: set[tuple[int, str]] = field(default_factory=set)
    """Call occurrences vetoed before execution; a successful post hook is a violation."""
    pending_tool_calls: dict[str, int] = field(default_factory=dict)
    """Latest call occurrence awaiting its native post-execution hook, keyed by call ID."""
    tool_interception_lock: asyncio.Lock = field(
        default_factory=asyncio.Lock, repr=False
    )
    """Native agents may finish sibling tools concurrently; serialize trace mutation."""
    prepared_users: Counter[str] = field(default_factory=Counter)

    @property
    def stopped(self) -> bool:
        return self.trace.stop_condition is not None

    async def rewrite_request(
        self,
        request: Request,
        *,
        run_stops: bool = True,
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
        prepared_users = self.prepared_users.copy()
        prepared: set[int] = set()
        candidates: set[int] = set()
        for position in range(tail_start, len(current.messages)):
            message = current.messages[position]
            if isinstance(message, UserMessage):
                candidates.add(position)
                key = graph.message_hash(message)
                if prepared_users[key]:
                    prepared_users[key] -= 1
                    prepared.add(position)
            elif isinstance(message, ToolMessage):
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
        if not candidates and (not run_stops or not self.request_stops):
            return request, [], None
        already_intercepted = candidates == prepared
        if (
            candidates
            and already_intercepted
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
                    if (
                        isinstance(before, ToolMessage)
                        and after.tool_call_id != before.tool_call_id
                    ):
                        raise ValueError(
                            "request interceptors cannot change a tool-call ID"
                        )
                    if isinstance(before, ToolMessage) and after.name != before.name:
                        raise ValueError(
                            "request interceptors cannot change a tool name"
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
            if isinstance(message, UserMessage):
                key = graph.message_hash(message)
                if self.prepared_users[key]:
                    self.prepared_users[key] -= 1
            elif isinstance(message, ToolMessage) and assistant_node is not None:
                tool_call = (assistant_node, message.tool_call_id)
                if self.prepared_tool_results.pop(tool_call, None) is not None:
                    if (
                        self.pending_tool_calls.get(message.tool_call_id)
                        == assistant_node
                    ):
                        self.pending_tool_calls.pop(message.tool_call_id)
                    self.blocked_tool_calls.discard(tool_call)

    async def prepare_users(
        self, request: Request
    ) -> tuple[Request, list[InterceptRecord]]:
        """Intercept caller-owned user turns before the harness stores them."""
        branch = self.trace.messages
        rewritten, records, _ = await self.rewrite_request(
            Request(messages=[*branch, *request.messages]), run_stops=False
        )
        tail = rewritten.messages[len(branch) :]
        self.prepared_users.update(
            graph.message_hash(message)
            for message in tail
            if isinstance(message, UserMessage)
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
        rewrite: ToolRewriteCapabilities | None,
    ) -> dict:
        """Run native tool policy before execution or before the next model turn."""
        record_phase = "after" if phase == "after_failure" else phase
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
        # Claude may report a failed tool only after its hidden agent loop has advanced
        # the issuing assistant off the leaf; the preceding before hook identifies it.
        if phase == "after_failure" and not matches:
            pending_node = self.pending_tool_calls.get(message.tool_call_id)
            matches = [pending_node] if pending_node is not None else []
        if len(matches) != 1:
            raise TaskError(
                f"{phase} tool call {message.tool_call_id!r} matched "
                f"{len(matches)} branches"
            )
        assistant_node = matches[0]
        late = assistant_node not in leaves
        tool_call = (assistant_node, message.tool_call_id)
        if phase != "before":
            if self.pending_tool_calls.get(message.tool_call_id) != assistant_node:
                raise HarnessError(
                    f"tool call {message.tool_call_id!r} reached a post-execution hook "
                    "without crossing its pre-execution hook"
                )
            self.pending_tool_calls.pop(message.tool_call_id)
            if tool_call in self.blocked_tool_calls:
                self.blocked_tool_calls.remove(tool_call)
                if phase == "after_failure":
                    return {"action": "allow"}
                raise HarnessError(
                    f"harness executed tool call {message.tool_call_id!r} after its "
                    "pre-execution result was replaced"
                )
        assistant = self.trace.nodes[assistant_node].message
        assert isinstance(assistant, AssistantMessage)
        branch = []
        node: int | None = assistant_node
        while node is not None:
            branch.append(self.trace.nodes[node].message)
            node = self.trace.nodes[node].parent
        branch.reverse()
        # Keep earlier results in the hook's trace, with the active result last so request
        # hooks can identify it. Commit them only when token attribution becomes available.
        previous = [
            self.prepared_tool_results[(assistant_node, call.id)]
            for call in assistant.tool_calls or []
            if call.id != message.tool_call_id
            and (assistant_node, call.id) in self.prepared_tool_results
        ]
        request, records, stopped = await self.rewrite_request(
            Request(
                messages=[*branch, *previous, message],
                tools=self.trace.tools or None,
            )
        )
        if self.released:
            raise HarnessError("rollout concluded during tool interception")
        self.trace.tool_policy_events.append(
            ToolPolicyEvent(tool_call_id=message.tool_call_id, phase=phase)
        )
        candidate = request.messages[-1]
        assert isinstance(candidate, ToolMessage)
        if candidate != message:
            if late or rewrite is None:
                raise HarnessError(
                    "request interception rewrote a tool result that this harness "
                    "cannot replace before the agent observes it"
                )
            if rewrite.content == "text" and not isinstance(candidate.content, str):
                raise HarnessError(
                    "request interception returned structured content to a "
                    "text-only tool hook"
                )
            if isinstance(candidate.content, str):
                if not rewrite.allow_empty_text and not candidate.content.strip(
                    JS_TRIM_CHARACTERS
                ):
                    raise HarnessError(
                        "request interception returned empty text that the tool hook "
                        "cannot preserve"
                    )
            else:
                if not rewrite.allow_empty_parts and not candidate.content:
                    raise HarnessError(
                        "request interception returned empty structured content that "
                        "the tool hook cannot preserve"
                    )
                if (
                    not rewrite.preserve_single_text_part
                    and len(candidate.content) == 1
                    and isinstance(candidate.content[0], TextContentPart)
                ):
                    raise HarnessError(
                        "request interception returned a single text part that the "
                        "tool hook canonicalizes to plain text"
                    )
                for part in candidate.content:
                    if not isinstance(part, ImageUrlContentPart):
                        continue
                    if rewrite.image_urls == "none":
                        raise HarnessError(
                            "request interception returned an image that the tool hook "
                            "cannot preserve"
                        )
                    metadata, separator, _ = part.image_url.url.partition(",")
                    media_type = metadata.removeprefix("data:").removesuffix(";base64")
                    is_data_url = bool(
                        separator
                        and metadata == f"data:{media_type};base64"
                        and media_type
                        and ";" not in media_type
                    )
                    if rewrite.image_urls == "data" and not is_data_url:
                        raise HarnessError(
                            "request interception returned a remote image URL that "
                            "the tool hook cannot preserve"
                        )
            if rewrite.max_text_utf16_units is not None:
                text_units = (
                    sum(
                        1 + (ord(character) > 0xFFFF) for character in candidate.content
                    )
                    if isinstance(candidate.content, str)
                    else sum(
                        1 + (ord(character) > 0xFFFF)
                        for part in candidate.content
                        if isinstance(part, TextContentPart)
                        for character in part.text
                    )
                )
                if text_units > rewrite.max_text_utf16_units:
                    raise HarnessError(
                        "request interception returned more text than the tool hook "
                        "can replace synchronously"
                    )
        self.trace.request_rewrites.extend(
            record.model_copy(update={"boundary": "tool", "phase": record_phase})
            for record in records
        )
        if phase == "before":
            self.pending_tool_calls[message.tool_call_id] = assistant_node
        if stopped is not None:
            if not late:
                results = {
                    result.tool_call_id: result
                    for result in request.messages[len(branch) :]
                    if isinstance(result, ToolMessage)
                }
                if record_phase == "before":
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
        if late:
            return {"action": "allow"}
        if phase == "before" and candidate == message:
            return {"action": "allow"}
        self.prepared_tool_results[tool_call] = candidate
        if candidate != message:
            if record_phase == "before":
                self.blocked_tool_calls.add(tool_call)
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
