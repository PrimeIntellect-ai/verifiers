"""The per-rollout unit the interception layer serves.

One `RolloutSession` per rollout, registered on an interception server under the rollout's
secret. The rollout constructs it (model ctx, trace, task `@stop`s, limits) and the server
drives it: assigns its model client at register, routes each intercepted model call to it,
runs `refused()` before each turn, and stashes the real failure on `error`. `RolloutLimits` is the framework's per-rollout
budget (turns / tokens), checked between turns.

Tool calls are policed from the same two boundaries. When a response commits, the request
hooks run once per proposed call (`gate_tool_calls`) — before the harness receives the
response — and their verdicts are served to the harness's gate (`decide_tool`). Tool results
are rewritten when the harness's next request carries them (`rewrite_request`), and every
rewrite is pinned to its call occurrence so the harness's own copy never reaches the model
or the graph.
"""

import asyncio
import hashlib
import inspect
import json
from collections import Counter
from collections.abc import Awaitable, Callable, Sequence
from contextlib import AbstractAsyncContextManager, suppress
from dataclasses import dataclass, field, replace
from functools import cached_property
from typing import get_origin, get_type_hints

from pydantic import TypeAdapter

from verifiers.v1 import graph
from verifiers.v1.clients import Client, ModelContext, ModelRoute
from verifiers.v1.configs.harness import DialectName
from verifiers.v1.configs.runtime import NetworkPolicyConfig
from verifiers.v1.errors import HarnessError, RolloutError, TaskError
from verifiers.v1.harnesses.utils.compaction import bound_tool_message
from verifiers.v1.runtimes.bridge import ExecutionBridge
from verifiers.v1.trace import InterceptRecord, Trace
from verifiers.v1.types import (
    AssistantMessage,
    Messages,
    Request,
    Response,
    ToolCall,
    ToolMessage,
    UserMessage,
)
from verifiers.v1.utils.decorators import invoke
from verifiers.v1.utils.logging import (
    LogScope,
    current_scope,
    enter_scope,
    get_logger,
)

logger = get_logger(__name__)

MCPProxy = Callable[[str, str], AbstractAsyncContextManager[str]]
ToolResultKey = tuple[int | str, str]


def _parse_tool_key(raw: list) -> ToolResultKey:
    first = raw[0]
    if isinstance(first, int):
        return (first, str(raw[1]))
    if isinstance(first, str) and first.isdigit():
        return (int(first), str(raw[1]))
    return (str(first), str(raw[1]))


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


def same_arguments(call: ToolCall, arguments: object) -> bool:
    """Whether a gate's view of a call's arguments is the model's call, ignoring JSON form."""
    try:
        return json.dumps(json.loads(call.arguments), sort_keys=True) == json.dumps(
            arguments, sort_keys=True
        )
    except (json.JSONDecodeError, TypeError):
        return call.arguments == arguments


@dataclass(frozen=True)
class RolloutLimits:
    """Per-rollout framework limits (None = no cap), checked before each turn is served.
    The first limit reached refuses the turn — halting any harness, the same mechanism as
    a @stop — and becomes the trace's stop condition. Each caps a trace computed property:
    `max_turns` -> num_turns, `max_input_tokens` -> num_input_tokens, `max_output_tokens` ->
    num_output_tokens, `max_total_tokens` -> num_total_tokens. Token caps are soft by one turn:
    they're checked between turns, so the turn that crosses a cap still completes.
    `max_turn_tokens` caps each response (applied through sampling) and refuses the turn
    after a response that hit it."""

    max_turns: int | None = None
    max_input_tokens: int | None = None
    max_output_tokens: int | None = None
    max_total_tokens: int | None = None
    max_turn_tokens: int | None = None

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
        if self.max_turn_tokens is not None:
            last = next((c for c in reversed(trace.calls) if c.error is None), None)
            # A length finish below the cap ran out of context, not turn tokens.
            if (
                last is not None
                and last.finish_reason == "length"
                and (
                    last.usage is None
                    or last.usage.completion_tokens >= self.max_turn_tokens
                )
            ):
                return "max_turn_tokens"
        return None


@dataclass
class IdempotentRequest:
    """One buffered model request shared by its original call and retries."""

    binding: tuple[str, bytes]
    response: "ReplayResponse | None" = None
    completed_at: float | None = None
    inflight: "asyncio.Future[ReplayResponse | None] | None" = None


@dataclass(frozen=True)
class ReplayResponse:
    """The exact HTTP result handed to coalesced in-flight request attempts."""

    status: int
    body: bytes
    content_type: str


@dataclass
class RolloutSession:
    route: ModelRoute
    """Where its model turns go; `ctx` is the route's current context."""
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
    on_turn_input: Callable[[], None] | None = None
    """One-shot acknowledgement when a resumed harness reaches model inference."""
    log_scope: LogScope = field(default_factory=current_scope)
    """The rollout's log scope, re-entered by the handler tasks that serve it."""
    gates_tools: bool = False
    """Whether the harness asks `/tool` before executing each call (see
    `Harness.SUPPORTS_TOOL_INTERCEPTION`). Without that gate a pre-execution rewrite
    cannot be enforced, so it ends the rollout instead of letting the call run."""
    client: Client | None = None
    """The model client serving this rollout's turns. The interception server gives it the
    server-owned client for its route's endpoint config (one per distinct config) at
    `register` and again when the route changes, so every rollout it multiplexes shares one
    keepalive connection pool instead of opening its own."""
    state_base_url: str | None = None
    """State URL for consumers outside an interception server's private LAN."""
    state_service_secrets: frozenset[str] = field(default_factory=frozenset, repr=False)
    proxy_mcp: MCPProxy | None = field(default=None, repr=False)
    """Register a host MCP backend on this session's interception server while serving."""
    error: "RolloutError | None" = None
    """The latest unresolved model-call failure. The harness only sees it as an HTTP error, so
    when its program dies on it the rollout records this original error instead of a secondary
    `HarnessError`. A harness that completes cleanly after the failure handled it. Reset before
    each model turn, so a successful retry clears it."""
    idempotent_requests: dict[str, IdempotentRequest] = field(default_factory=dict)
    """Explicit keys or marked SDK retries mapped to their replay state."""
    released: bool = False
    """Set when the rollout unregisters the session: the trace is sealed (its conclusion is
    what scored and persisted), so a handler still in flight must not commit turns, record
    calls, or write state onto it — the in-memory trace must stay what the run produced."""
    tasks: set["asyncio.Task"] = field(default_factory=set)
    """Handler tasks currently serving this session. aiohttp does not cancel a handler when
    its client disconnects, so a request whose program died at teardown would keep driving
    the exchange (upstream call, simulator turn) — unregistering cancels these instead."""
    tool_decisions: dict[ToolResultKey, ToolMessage | None] = field(
        default_factory=dict
    )
    """The pre-execution verdict per proposed call occurrence: None to run it, else the result
    the policy put in its place — that call must not execute."""
    remote_tool_handler: Callable[[ToolCall], Awaitable[ToolMessage]] | None = None
    execution_bridge: ExecutionBridge | None = None
    tool_gate_can_deny: Callable[[ToolCall], bool] | None = None
    check_tool_call: Callable | None = None
    native_tool_decisions: dict[tuple[int, str], tuple[str, dict]] = field(
        default_factory=dict
    )
    _tool_gate_lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    pinned_tool_results: dict[ToolResultKey, ToolMessage] = field(default_factory=dict)
    """The results the model must see, by issuing turn (or prefix) and call id: rewrites and
    results it substituted for blocked calls. The harness keeps its own copy of each in its
    history and re-sends it forever, so every request has these re-applied before the graph is
    matched — the trace holds the pinned version, and a differing copy would fork it at that node."""
    gated_tools: set[ToolResultKey] = field(default_factory=set)
    """Calls the harness's gate asked about before executing."""
    tool_dialects: dict[int, DialectName] = field(default_factory=dict)
    """Wire dialect of each issuing node, used when its harness asks the tool gate."""
    tool_issuers: dict[tuple[str, str], set[int]] = field(default_factory=dict)
    """Sampled nodes that issued a call ID under a replay-equivalent prefix."""
    tool_call_replays: dict[tuple[str, str], list[tuple[object, ToolCall]]] = field(
        default_factory=dict
    )
    """Native argument forms observed at the gate, scoped to their conversation prefix.
    Remote adapters execute the sampled call; replay must retain that call too."""
    _remote_gate_queue: asyncio.Queue[dict | None] = field(
        default_factory=asyncio.Queue
    )
    _pending_pins: dict[
        ToolResultKey, tuple[ToolCall, list[str], asyncio.Future[None]]
    ] = field(default_factory=dict)
    prepared_users: Counter[str] = field(default_factory=Counter)
    last_request: tuple[list, Messages] = ((), [])
    """The last chat request's raw messages (copied: the server rewrites a body in place) and
    their parse, which the next request's equal prefix reuses instead of parsing again."""

    @property
    def ctx(self) -> ModelContext:
        """The model, client config and sampling of its next model turn: the route's, with
        `max_turn_tokens` capping its `max_tokens` whatever sampling the route was given."""
        ctx = self.route.ctx
        cap = self.limits.max_turn_tokens
        max_tokens = ctx.sampling.max_tokens
        if cap is None or (max_tokens is not None and max_tokens <= cap):
            return ctx
        return replace(
            ctx, sampling=ctx.sampling.model_copy(update={"max_tokens": cap})
        )

    @property
    def stopped(self) -> bool:
        return self.trace.stop_condition is not None

    def _prefix_hash(self, messages: Sequence, dialect: DialectName | None) -> str:
        digest = hashlib.blake2b(digest_size=16)
        for message in messages:
            digest.update(
                graph.replay_message_hash(self.trace, message, dialect).encode()
            )
        return digest.hexdigest()

    def _tool_messages(
        self,
        request: Request,
        dialect: DialectName | None = None,
        known_path: Sequence[int] = (),
    ):
        """Use the sampled node and call ID for a model-issued tool, restoring any
        native argument normalization observed at the split gate. A harness-supplied
        call outside the sampled graph uses its conversation prefix instead."""
        prefix = hashlib.blake2b(digest_size=16)
        calls: dict[str, ToolResultKey] = {}
        for position, message in enumerate(request.messages):
            if (
                isinstance(message, AssistantMessage)
                and message.tool_calls
                and self.tool_call_replays
            ):
                p_hex = prefix.hexdigest()
                restored = []
                for call in message.tool_calls:
                    aliases = self.tool_call_replays.get((p_hex, call.id), ())
                    restored.append(
                        next(
                            (
                                original
                                for arguments, original in aliases
                                if original.name == call.name
                                and same_arguments(call, arguments)
                            ),
                            call,
                        )
                    )
                if restored != message.tool_calls:
                    message = message.model_copy(update={"tool_calls": restored})
            key: ToolResultKey | None = None
            if isinstance(message, ToolMessage) and message.tool_call_id in calls:
                key = calls[message.tool_call_id]
                issuer, call_id = key
                if isinstance(issuer, str) and (
                    issuers := self.tool_issuers.get((issuer, call_id))
                ):
                    # A follow-up can change tools independently of its conversation.
                    if len(issuers) != 1:
                        raise HarnessError(
                            f"Ambiguous issuing turn for tool call {call_id!r}"
                        )
                    key = (next(iter(issuers)), call_id)
            yield position, message, key
            canonical = (
                self.pinned_tool_results.get(key, message)
                if key is not None
                else message
            )
            prefix.update(
                graph.replay_message_hash(self.trace, canonical, dialect).encode()
            )
            if isinstance(message, AssistantMessage) and message.tool_calls:
                prefix_hash = prefix.hexdigest()
                for call in message.tool_calls:
                    if (
                        position < len(known_path)
                        and self.trace.nodes[known_path[position]].sampled
                    ):
                        calls[call.id] = (known_path[position], call.id)
                    else:
                        calls[call.id] = (prefix_hash, call.id)

    def tool_result_keys(
        self,
        request: Request,
        dialect: DialectName | None = None,
        known_path: Sequence[int] = (),
    ) -> dict[int, ToolResultKey]:
        return {
            position: key
            for position, _, key in self._tool_messages(request, dialect, known_path)
            if key is not None
        }

    def pin_tool_results(
        self,
        request: Request,
        dialect: DialectName | None = None,
        known_path: Sequence[int] = (),
    ) -> Request:
        """Re-impose the pinned results on the harness's request. A result for a blocked call
        the harness never gated means the call ran anyway: the transcript would no longer
        describe the sandbox, so that fails the rollout instead of being papered over."""
        if not self.pinned_tool_results and not self.tool_call_replays:
            return request
        messages = list(request.messages)
        changed = False
        for position, message, key in self._tool_messages(request, dialect, known_path):
            pinned = self.pinned_tool_results.get(key) if key is not None else None
            if (
                key is not None
                and pinned is not None
                and self.tool_decisions.get(key) is not None
                and key not in self.gated_tools
            ):
                raise HarnessError(
                    f"tool call {key[1]!r} executed although the policy "
                    "blocked it: the harness never consulted the tool gate"
                )
            messages[position] = pinned if pinned is not None else message
            changed |= messages[position] is not request.messages[position]
        return request.model_copy(update={"messages": messages}) if changed else request

    async def rewrite_request(
        self,
        request: Request,
        *,
        run_stops: bool = True,
        dialect: DialectName | None = None,
        known_path: Sequence[int] = (),
    ) -> tuple[Request, list[InterceptRecord], str | None]:
        """Run typed request interceptors and stops over one canonical request. Only the
        uncommitted tail's user and tool messages may change; a rewritten tool result is
        pinned for the rest of the rollout."""
        if self._pending_pins:
            present_ids = {
                message.tool_call_id
                for message in request.messages
                if isinstance(message, ToolMessage)
            }
            pending = [
                asyncio.shield(future)
                for _, (call, _, future) in list(self._pending_pins.items())
                if call.id in present_ids and not future.done()
            ]
            if pending:
                with suppress(TimeoutError, asyncio.CancelledError):
                    await asyncio.wait_for(asyncio.gather(*pending), timeout=10.0)
        request = self.pin_tool_results(request, dialect, known_path)
        if not self.request_interceptors and (not run_stops or not self.request_stops):
            return request, [], None
        tool_keys = self.tool_result_keys(request, dialect, known_path)
        candidates: set[int] = set()
        tail_start = graph.message_prefix_len(self.trace, request.messages, dialect)
        if self.request_interceptors:
            prepared_users = self.prepared_users.copy()
            for position in range(tail_start, len(request.messages)):
                message = request.messages[position]
                if isinstance(message, UserMessage):
                    if prepared_users:
                        key = graph.message_hash(message)
                        if prepared_users[key]:
                            prepared_users[key] -= 1
                            continue
                    candidates.add(position)
                elif (
                    isinstance(message, ToolMessage)
                    and tool_keys.get(position) not in self.pinned_tool_results
                ):
                    candidates.add(position)

        # A tool hook observes each result at the tail, including parallel calls.
        boundaries = [
            position + 1
            for position in range(tail_start, len(request.messages))
            if isinstance(request.messages[position], ToolMessage)
            and tool_keys.get(position) not in self.pinned_tool_results
        ]
        if not boundaries or boundaries[-1] != len(request.messages):
            boundaries.append(len(request.messages))
        messages = list(request.messages)
        records: list[InterceptRecord] = []
        previous = 0
        try:
            for end in boundaries:
                active = {
                    position for position in candidates if previous <= position < end
                }
                current = request.model_copy(update={"messages": messages[:end]})
                for handler in self.request_interceptors if active else []:
                    candidate = current.model_copy(deep=True)
                    result = await call_hook(
                        handler, {Request: candidate, Trace: self.trace}
                    )
                    if result is None:
                        continue
                    if not isinstance(result, Request):
                        raise TypeError(
                            f"expected Request, got {type(result).__name__}"
                        )
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
                        if position not in active:
                            raise ValueError(
                                "request interceptors can only rewrite new user or tool messages"
                            )
                        if type(after) is not type(before):
                            raise TypeError(
                                f"expected {type(before).__name__}, got {type(after).__name__}"
                            )
                        if isinstance(after, ToolMessage):
                            after = ToolMessage.model_validate(
                                bound_tool_message(after.model_dump())
                            )
                            result.messages[position] = after
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

                messages[:end] = current.messages
                stops = self.request_stops if run_stops else []
                for stop in stops:
                    candidate = current.model_copy(deep=True)
                    result = await call_hook(
                        stop, {Request: candidate, Trace: self.trace}
                    )
                    if not isinstance(result, bool):
                        raise TypeError(
                            f"@stop must return bool, got {type(result).__name__}"
                        )
                    if result:
                        return (
                            request.model_copy(update={"messages": messages}),
                            records,
                            stop.__name__,
                        )
                previous = end
        except RolloutError:
            raise
        except Exception as error:
            raise TaskError(
                f"request interception failed: {type(error).__name__}: {error}"
            ) from error
        current = request.model_copy(update={"messages": messages})
        tool_keys = self.tool_result_keys(current, dialect, known_path)
        for position in candidates:
            after = current.messages[position]
            if (
                isinstance(after, ToolMessage)
                and after != request.messages[position]
                and position in tool_keys
            ):
                self.pinned_tool_results[tool_keys[position]] = after
        return current, records, None

    def consume_prepared(self, messages: Messages) -> None:
        """Forget pre-harness user rewrites only after their model request commits."""
        for message in messages:
            if isinstance(message, UserMessage) and self.prepared_users:
                key = graph.message_hash(message)
                if self.prepared_users[key]:
                    self.prepared_users[key] -= 1

    async def prepare_users(
        self, request: Request
    ) -> tuple[Request, list[InterceptRecord]]:
        """Intercept caller-owned user turns before the harness stores them."""
        known_path = (
            graph.path_node_ids(self.trace, len(self.trace.nodes) - 1)
            if self.trace.nodes
            else []
        )
        branch = [self.trace.nodes[i].message for i in known_path]
        rewritten, records, _ = await self.rewrite_request(
            Request(messages=[*branch, *request.messages]),
            run_stops=False,
            known_path=known_path,
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

    async def gate_tool_calls(
        self,
        node: int,
        dialect: DialectName | None = None,
        harness_history: Messages | None = None,
    ) -> str | None:
        """Police the tool calls the assistant committed at `node` before the harness sees
        them: the request hooks run over the branch plus an empty result for each call, so a
        `@stop` refuses the whole response and a rewrite becomes that call's verdict — the
        harness's gate denies it and the model sees the rewritten result instead. Returns
        the name of the stop that fired, if any."""
        assistant = self.trace.nodes[node].message
        if dialect is not None:
            self.tool_dialects[node] = dialect
        if not (
            self.request_interceptors or self.request_stops or self.remote_tool_handler
        ) or not (isinstance(assistant, AssistantMessage) and assistant.tool_calls):
            return None
        known_path = graph.path_node_ids(self.trace, node)
        branch = [self.trace.nodes[i].message for i in known_path]
        tools = self.trace.nodes[known_path[0]].tools
        # Live harnesses replay their own input. A resumed harness may replay the
        # model-facing transcript, including any policy notices or rewritten user turns.
        histories = [branch]
        if harness_history is not None and harness_history != branch[:-1]:
            histories.append([*harness_history, assistant])
        alias_prefixes: list[str] = []
        for history in histories:
            before_assistant = self._prefix_hash(history[:-1], dialect)
            if before_assistant not in alias_prefixes:
                alias_prefixes.append(before_assistant)
            prefix_hash = self._prefix_hash(history, dialect)
            for call in assistant.tool_calls:
                self.tool_issuers.setdefault((prefix_hash, call.id), set()).add(node)
        uses_local_gate = bool(
            getattr(self.remote_tool_handler, "uses_local_gate", False)
        )
        native_tools: frozenset[str] | None = getattr(
            self.remote_tool_handler, "native_tools", None
        )
        queued: list[dict] = []
        pending_entries: list[tuple[ToolResultKey, ToolCall]] = []
        for call in assistant.tool_calls:
            if self.check_tool_call is not None:
                self.check_tool_call(
                    call,
                    intercepted=bool(self.request_interceptors or self.request_stops),
                )
            probe = ToolMessage(tool_call_id=call.id, content="", name=call.name)
            request = Request(messages=[*branch, probe], tools=tools or None)
            key: ToolResultKey = (node, call.id)
            if key in self.tool_decisions:
                raise HarnessError(
                    f"tool call id {call.id!r} was reused in the same assistant turn"
                )
            request, records, stopped = await self.rewrite_request(
                request, dialect=dialect, known_path=known_path
            )
            self.trace.request_rewrites.extend(records)
            if stopped is not None:
                return stopped
            verdict = request.messages[-1]
            assert isinstance(verdict, ToolMessage)
            is_native = native_tools is None or call.name in native_tools
            if verdict == probe:
                self.tool_decisions[key] = None
                denied = None
            else:
                if not self.gates_tools or (
                    self.tool_gate_can_deny is not None
                    and not self.tool_gate_can_deny(call)
                ):
                    return records[-1].handler
                self.tool_decisions[key] = verdict
                denied = verdict.model_dump(exclude_none=True)
            if uses_local_gate and (is_native or denied is not None):
                queued.append(
                    {
                        "key": list(key),
                        "call": call.model_dump(),
                        "denied": denied,
                    }
                )
                pending_entries.append((key, call))
        if self.remote_tool_handler is not None and not self.gates_tools:
            raise HarnessError("split tool execution requires a harness tool gate")
        if uses_local_gate and queued:
            loop = asyncio.get_running_loop()
            for key, call in pending_entries:
                self._pending_pins[key] = (
                    call,
                    list(alias_prefixes),
                    loop.create_future(),
                )
            self._remote_gate_queue.put_nowait({"items": queued})
        return None

    def _finish_pending_pin(self, key: ToolResultKey) -> None:
        entry = self._pending_pins.pop(key, None)
        if entry is not None and not entry[2].done():
            entry[2].set_result(None)

    def _record_replay_alias(
        self, alias_prefixes: Sequence[str], call: ToolCall, arguments: object
    ) -> None:
        if not same_arguments(call, arguments):
            for alias_prefix in alias_prefixes:
                aliases = self.tool_call_replays.setdefault((alias_prefix, call.id), [])
                if not any(observed == arguments for observed, _ in aliases):
                    aliases.append((arguments, call))

    def _pin_tool_verdict(
        self, key: ToolResultKey, call: ToolCall, raw_or_msg: ToolMessage | dict
    ) -> ToolMessage:
        payload = (
            raw_or_msg.model_dump(exclude_none=True)
            if isinstance(raw_or_msg, ToolMessage)
            else raw_or_msg
        )
        verdict = ToolMessage.model_validate(bound_tool_message(payload))
        if verdict.tool_call_id != call.id or verdict.name != call.name:
            raise HarnessError("remote tool handler returned the wrong call")
        self.tool_decisions[key] = verdict
        self.pinned_tool_results[key] = verdict
        return verdict

    async def poll_remote_gate(self, timeout: float = 20.0) -> dict:
        if self.released or self.stopped:
            return {"closed": True}
        try:
            batch = await asyncio.wait_for(
                self._remote_gate_queue.get(), timeout=timeout
            )
        except TimeoutError:
            return {"items": []}
        if batch is None or self.released:
            return {"closed": True}
        return batch

    def record_remote_pin(
        self,
        raw_key: list,
        raw_message: dict,
        arguments: object = None,
    ) -> None:
        key = _parse_tool_key(raw_key)
        pending = self._pending_pins.get(key)
        if pending is None or self.tool_decisions.get(key) is not None:
            raise HarnessError("unexpected or unapproved remote tool pin")
        call, alias_prefixes, _ = pending
        self._pin_tool_verdict(key, call, raw_message)
        self._record_replay_alias(alias_prefixes, call, arguments)
        self.gated_tools.add(key)
        self._finish_pending_pin(key)

    def record_remote_denial_ack(
        self,
        raw_key: list,
        arguments: object = None,
    ) -> None:
        key = _parse_tool_key(raw_key)
        pending = self._pending_pins.get(key)
        if pending is None or self.tool_decisions.get(key) is None:
            raise HarnessError("unexpected remote tool denial acknowledgement")
        call, alias_prefixes, _ = pending
        self._record_replay_alias(alias_prefixes, call, arguments)
        self.gated_tools.add(key)
        self._finish_pending_pin(key)

    async def decide_tool(
        self, tool_call_id: str, name: str | None = None, arguments: object = None
    ) -> dict:
        async with self._tool_gate_lock:
            return await self._decide_tool(tool_call_id, name, arguments)

    async def _decide_tool(
        self, tool_call_id: str, name: str | None, arguments: object
    ) -> dict:
        """Answer the harness's gate for a call it is about to execute with the verdict
        `gate_tool_calls` recorded. Native gates may wrap the model's call id or carry it
        in their input, and some only know the arguments. A call the model never made — a
        tool a Codex Code Mode script invokes on its behalf — had no earlier point to be
        judged at, so the request hooks judge it now, appended to the turn that spawned it."""
        if self.stopped:
            return {"action": "stop", "reason": self.trace.stop_condition}
        leaves = [
            (leaf, message)
            for leaf in graph.leaves(self.trace)
            if isinstance(message := self.trace.nodes[leaf].message, AssistantMessage)
        ]
        calls = [
            (leaf, call)
            for leaf, message in leaves
            for call in message.tool_calls or []
        ]
        keys = {tool_call_id}
        if isinstance(arguments, dict):
            keys.update(
                arguments[key]
                for key in ("tool_call_id", "call_id", "toolCallId")
                if isinstance(arguments.get(key), str)
            )
        matches = [
            (leaf, call)
            for leaf, call in calls
            if any(
                key == call.id
                or key.startswith(f"{call.id}|")
                or key.endswith(f":{call.id}")
                for key in keys
            )
        ] or [
            (leaf, call)
            for leaf, call in calls
            if (name is None or call.name == name) and same_arguments(call, arguments)
        ]
        if len(matches) > 1:
            matches = [
                (leaf, call)
                for leaf, call in matches
                if (name is None or call.name == name)
                and same_arguments(call, arguments)
            ] or matches
        if len(matches) == 1:
            leaf, call = matches[0]
            dialect = self.tool_dialects.get(leaf)
            deferred = getattr(self.remote_tool_handler, "defer_until_gate", False)
            if (
                self.remote_tool_handler is not None
                and not deferred
                and name is not None
                and name != call.name
            ):
                raise HarnessError("split tool gate call differs from the sampled call")
            branch = graph.path(self.trace, leaf)
            if self.remote_tool_handler is not None and not deferred:
                self._record_replay_alias(
                    [self._prefix_hash(branch[:-1], dialect)], call, arguments
                )
            key: ToolResultKey = (leaf, call.id)
            self.gated_tools.add(key)
            verdict = self.tool_decisions.get(key)
            native_tools: frozenset[str] | None = getattr(
                self.remote_tool_handler, "native_tools", None
            )
            is_native = native_tools is None or call.name in native_tools
            if deferred and verdict is None:
                return await self._decide_native_tool(
                    leaf, tool_call_id, name or call.name, arguments
                )
            if self.remote_tool_handler is not None and verdict is None and is_native:
                if key not in self.tool_decisions:
                    raise HarnessError("split tool gate has no policy decision")
                verdict = self._pin_tool_verdict(
                    key, call, await self.remote_tool_handler(call)
                )
            self._finish_pending_pin(key)
        elif matches or not name:
            raise HarnessError(
                f"tool gate asked about {tool_call_id!r}, which matches "
                f"{len(matches)} proposed calls"
            )
        else:
            deferred = getattr(self.remote_tool_handler, "defer_until_gate", False)
            if self.remote_tool_handler is not None and not deferred:
                raise HarnessError("split tool gate received an unsampled call")
            native_worker = self.execution_bridge is not None
            native_leaves = (
                graph.leaves(self.trace)
                if deferred or native_worker
                else [leaf for leaf, _ in leaves]
            )
            if len(native_leaves) > 1 or (
                not native_leaves and not (deferred or native_worker)
            ):
                raise HarnessError(
                    "native tool gate requires one active conversation branch"
                )
            return await self._decide_native_tool(
                native_leaves[0] if native_leaves else -1, tool_call_id, name, arguments
            )
        if verdict is None:
            return {"action": "allow"}
        return {"action": "deny", "message": verdict.model_dump(exclude_none=True)}

    async def _decide_native_tool(
        self, leaf: int, tool_call_id: str, name: str, arguments: object
    ) -> dict:
        """Judge native commands after their harness has resolved arguments.

        Some environments also issue setup commands before the first model turn.
        These use the same policy boundary and retry receipts as sampled calls.
        """
        native_key = (leaf, tool_call_id)
        fingerprint = json.dumps([name, arguments], sort_keys=True)
        if receipt := self.native_tool_decisions.get(native_key):
            previous, decision = receipt
            if previous != fingerprint:
                raise HarnessError(
                    "native tool call id was reused with different arguments"
                )
            return decision
        known_path = graph.path_node_ids(self.trace, leaf) if leaf >= 0 else []
        branch = [self.trace.nodes[i].message for i in known_path]
        tools = (
            self.trace.nodes[known_path[0]].tools if known_path else self.trace.tools
        )
        call = ToolCall(id=tool_call_id, name=name, arguments=json.dumps(arguments))
        if branch and isinstance(branch[-1], AssistantMessage):
            assistant = branch.pop()
            known_path = known_path[:-1]
            assistant = assistant.model_copy(
                update={
                    "tool_calls": [
                        *(
                            previous
                            for previous in assistant.tool_calls or []
                            if previous.id != call.id
                        ),
                        call,
                    ]
                }
            )
        else:
            assistant = AssistantMessage(content="", tool_calls=[call])
        probe = ToolMessage(tool_call_id=call.id, content="", name=name)
        request, records, stopped = await self.rewrite_request(
            Request(messages=[*branch, assistant, probe], tools=tools or None),
            dialect=self.tool_dialects.get(leaf),
            known_path=known_path,
        )
        self.trace.request_rewrites.extend(records)
        if stopped is not None:
            self.trace.stop(stopped)
            decision = {"action": "stop", "reason": stopped}
        else:
            verdict = request.messages[-1]
            assert isinstance(verdict, ToolMessage)
            if verdict == probe:
                verdict = None
                if self.remote_tool_handler is not None:
                    verdict = await self.remote_tool_handler(call)
                    if verdict.tool_call_id != call.id or verdict.name != call.name:
                        raise HarnessError(
                            "remote tool handler returned the wrong native call"
                        )
                    verdict = ToolMessage.model_validate(
                        bound_tool_message(verdict.model_dump(exclude_none=True))
                    )
            decision = (
                {"action": "allow"}
                if verdict is None
                else {
                    "action": "deny",
                    "message": verdict.model_dump(exclude_none=True),
                }
            )
        self.native_tool_decisions[native_key] = (fingerprint, decision)
        return decision

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
        if task is asyncio.current_task():
            enter_scope(self.log_scope)
        self.tasks.add(task)
        task.add_done_callback(self.tasks.discard)

    def release(self) -> None:
        """Seal the session: no further trace mutation, and in-flight handlers cancel."""
        self.released = True
        self._remote_gate_queue.put_nowait(None)
        for _, (_, _, future) in list(self._pending_pins.items()):
            if not future.done():
                future.cancel()
        self._pending_pins.clear()
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
            logger.debug("Reached the %s limit", limit)
            return limit
        for stop in self.trace_stops:
            result = await call_hook(stop, {Trace: self.trace})
            if not isinstance(result, bool):
                raise TaskError(f"@stop must return bool, got {type(result).__name__}")
            if result:
                self.trace.stop(stop.__name__)
                logger.debug("Stopped by @stop %s", stop.__name__)
                return stop.__name__
        return None
