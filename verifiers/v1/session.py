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
import copy
import hashlib
import inspect
import json
from collections import Counter
from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
from contextlib import AbstractAsyncContextManager
from dataclasses import dataclass, field, replace
from difflib import SequenceMatcher
from functools import cached_property
from itertools import chain
from typing import Any, TypeVar, get_origin, get_type_hints

from pydantic import TypeAdapter

from verifiers.v1 import graph
from verifiers.v1.clients import Client, ModelContext, ModelRoute
from verifiers.v1.configs.harness import DialectName
from verifiers.v1.configs.runtime import NetworkPolicyConfig
from verifiers.v1.dialects.base import CAPABILITY_NOTICE
from verifiers.v1.errors import HarnessError, RolloutError, TaskError
from verifiers.v1.harnesses.utils.compaction import bound_tool_message
from verifiers.v1.trace import InterceptRecord, Trace
from verifiers.v1.types import (
    AssistantMessage,
    Message,
    Messages,
    Request,
    Response,
    TextContentPart,
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

MCPProxy = Callable[[str], AbstractAsyncContextManager[str]]
ToolResultKey = tuple[int | str, str]
StopVerdict = tuple[str, dict[str, Any]]
PendingRewrite = tuple[int | None, InterceptRecord]
HookT = TypeVar("HookT", Request, Response, bool)


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


def unpack_hook(result: object, expected: type[HookT]) -> tuple[HookT, dict[str, Any]]:
    """Unpack `value` or `(value, metadata)` returned by an `@intercept` or `@stop` hook."""
    match result:
        case (value, Mapping() as info) if all(isinstance(k, str) for k in info):
            pass
        case tuple():
            raise TypeError(f"expected ({expected.__name__}, dict[str, Any])")
        case value:
            info = {}
    if not isinstance(value, expected):
        raise TypeError(f"expected {expected.__name__}, got {type(value).__name__}")
    return value, copy.deepcopy(dict(info))


def _tool_calls(messages: Iterable[Message]) -> dict[str, ToolCall]:
    calls: dict[str, ToolCall] = {}
    for msg in messages:
        if isinstance(msg, AssistantMessage) and msg.tool_calls:
            calls.update((call.id, call) for call in msg.tool_calls)
    return calls


def _tool_call(calls: dict[str, ToolCall], message: Message) -> ToolCall | None:
    if not isinstance(message, ToolMessage):
        return None
    return calls.get(
        message.tool_call_id,
        ToolCall(id=message.tool_call_id, name=message.name or "", arguments=""),
    )


def _make_record(
    handler: Callable,
    node: int | None,
    pos: int,
    calls: dict[str, ToolCall],
    message: Message,
    info: dict[str, Any],
) -> PendingRewrite:
    rec = InterceptRecord(
        handler=handler.__name__,
        node=node,
        tool_call=_tool_call(calls, message),
        info=copy.deepcopy(info),
    )
    return (pos if node is None else None, rec)


def _role_key(message: Message) -> tuple[type, str]:
    return (
        type(message),
        message.tool_call_id if isinstance(message, ToolMessage) else "",
    )


def _align_role_slice(
    before: Messages, after: Messages, nodes: list[int]
) -> Iterator[tuple[int, int]]:
    b_keys = [_role_key(m) for m in before]
    a_keys = [_role_key(m) for m in after]
    for tag, i1, i2, j1, j2 in SequenceMatcher(
        None, b_keys, a_keys, autojunk=False
    ).get_opcodes():
        if tag == "equal":
            yield from zip(range(i1, i2), nodes[j1:j2])


def _strip_notice(message: Message) -> Message:
    if not isinstance(message, UserMessage):
        return message
    if isinstance(message.content, str):
        if message.content.startswith(CAPABILITY_NOTICE):
            return message.model_copy(update={"content": ""})
        head, sep, _ = message.content.partition(f"\n\n{CAPABILITY_NOTICE}")
        return message.model_copy(update={"content": head}) if sep else message
    if (
        message.content
        and isinstance(last := message.content[-1], TextContentPart)
        and last.text.startswith(CAPABILITY_NOTICE)
    ):
        return message.model_copy(update={"content": message.content[:-1]})
    return message


def _align_nodes(
    before: Messages, after: Messages, prompt_nodes: list[int]
) -> dict[int, int]:
    """Map pre-mediation message indices in `before` to committed `prompt_nodes`."""
    trimmed = (
        after[:-1]
        if after
        and isinstance(after[-1], UserMessage)
        and not any(isinstance(m, UserMessage) for m in before)
        else after
    )
    if [_role_key(m) for m in before] == [_role_key(m) for m in trimmed]:
        return {i: prompt_nodes[i] for i in range(len(trimmed))}
    b_hashes = [graph.message_hash(m) for m in before]
    a_hashes = [graph.message_hash(_strip_notice(m)) for m in trimmed]
    mapping: dict[int, int] = {}
    for tag, i1, i2, j1, j2 in SequenceMatcher(
        None, b_hashes, a_hashes, autojunk=False
    ).get_opcodes():
        if tag == "equal":
            mapping.update(zip(range(i1, i2), prompt_nodes[j1:j2]))
        elif tag == "replace":
            mapping.update(
                (i1 + rel_i, nid)
                for rel_i, nid in _align_role_slice(
                    before[i1:i2], trimmed[j1:j2], prompt_nodes[j1:j2]
                )
            )
    return mapping


def _diff_request_rewrite(
    current: Request, result: Request, active: set[int]
) -> list[int]:
    """Validate a request rewrite in place and return the positions that changed."""
    if len(result.messages) != len(current.messages):
        raise ValueError("request interceptors cannot add or remove messages")
    if result.tools != current.tools:
        raise ValueError("request interceptors cannot rewrite tools")
    changed: list[int] = []
    for pos, (before, after) in enumerate(
        zip(current.messages, result.messages, strict=True)
    ):
        if before == after:
            continue
        if pos not in active:
            raise ValueError(
                "request interceptors can only rewrite new user or tool messages"
            )
        if type(after) is not type(before):
            raise TypeError(
                f"expected {type(before).__name__}, got {type(after).__name__}"
            )
        if isinstance(after, ToolMessage):
            assert isinstance(before, ToolMessage)
            after = ToolMessage.model_validate(bound_tool_message(after.model_dump()))
            result.messages[pos] = after
            if after.tool_call_id != before.tool_call_id or after.name != before.name:
                raise ValueError(
                    "request interceptors cannot change a tool-call ID or name"
                )
            if after == before:
                continue
        changed.append(pos)
    return changed


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
    trace_stops: list[Callable] = field(default_factory=list)
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
    pinned_tool_results: dict[ToolResultKey, ToolMessage] = field(default_factory=dict)
    """The results the model must see, by conversation prefix and call id: rewrites and results it
    substituted for blocked calls. The harness keeps its own copy of each in its history and
    re-sends it forever, so every request has these re-applied before the graph is matched —
    the trace holds the pinned version, and a differing copy would fork it at that node."""
    gated_tools: set[ToolResultKey] = field(default_factory=set)
    """Calls the harness's gate asked about before executing."""
    tool_dialects: dict[int, DialectName] = field(default_factory=dict)
    """Wire dialect of each issuing node, used when its harness asks the tool gate."""
    tool_issuers: dict[tuple[str, str], set[int]] = field(default_factory=dict)
    """Sampled nodes that issued a call ID under a replay-equivalent prefix."""
    prepared_users: Counter[str] = field(default_factory=Counter)
    prepared_rewrites: list[tuple[int, InterceptRecord]] = field(default_factory=list)
    """`(user_slot, record)` pairs from `prepare_users` awaiting their prompt commit."""
    _prepared_user_count: int = 0
    pinned_rewrites: list[tuple[ToolResultKey, InterceptRecord]] = field(
        default_factory=list
    )
    """`(tool_key, record)` pairs for pinned tool results awaiting their node commit."""
    inflight_rewrites: dict[str, list[tuple[int, InterceptRecord]]] = field(
        default_factory=dict
    )
    """Unpinned `(pos, record)` pairs keyed by `replay_key`, flushed if the turn never commits."""
    last_request: tuple[list, Messages] = ((), [])
    """The last chat request's raw messages (copied: the server rewrites a body in place) and
    their parse, which the next request's equal prefix reuses instead of parsing again."""
    judge_only: bool = False
    """Serve packaged grader calls without touching the solver graph, limits, or state."""
    network_policy_source: Callable[[], NetworkPolicyConfig] | None = field(
        default=None, repr=False
    )
    """Optional live network policy reader (e.g. when a task lifts egress mid-scoring)."""

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

    def tool_result_keys(
        self,
        request: Request,
        dialect: DialectName | None = None,
        known_path: Sequence[int] | Mapping[int, int] = (),
    ) -> dict[int, ToolResultKey]:
        """Use the sampled node and call ID for a model-issued tool. A harness-supplied
        call outside the sampled graph uses its conversation prefix instead."""
        prefix = hashlib.blake2b(digest_size=16)
        calls: dict[str, ToolResultKey] = {}
        keys: dict[int, ToolResultKey] = {}
        for position, message in enumerate(request.messages):
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
                keys[position] = key
                message = self.pinned_tool_results.get(key, message)
            prefix.update(
                graph.replay_message_hash(self.trace, message, dialect).encode()
            )
            if isinstance(message, AssistantMessage) and message.tool_calls:
                prefix_hash = prefix.hexdigest()
                issuer_node: int | None = (
                    known_path.get(position)
                    if isinstance(known_path, Mapping)
                    else (known_path[position] if position < len(known_path) else None)
                )
                for call in message.tool_calls:
                    alias = (prefix_hash, call.id)
                    if (
                        issuer_node is not None
                        and self.trace.nodes[issuer_node].sampled
                    ):
                        calls[call.id] = (issuer_node, call.id)
                    else:
                        calls[call.id] = alias
        return keys

    def pin_tool_results(
        self,
        request: Request,
        dialect: DialectName | None = None,
        known_path: Sequence[int] = (),
    ) -> Request:
        """Re-impose the pinned results on the harness's request. A result for a blocked call
        the harness never gated means the call ran anyway: the transcript would no longer
        describe the sandbox, so that fails the rollout instead of being papered over."""
        if not self.pinned_tool_results:
            return request
        messages = list(request.messages)
        changed = False
        for position, key in self.tool_result_keys(
            request, dialect, known_path
        ).items():
            pinned = self.pinned_tool_results.get(key)
            if pinned is None:
                continue
            if self.tool_decisions.get(key) is not None and key not in self.gated_tools:
                raise HarnessError(
                    f"tool call {key[1]!r} executed although the policy "
                    "blocked it: the harness never consulted the tool gate"
                )
            messages[position] = pinned
            changed = True
        return request.model_copy(update={"messages": messages}) if changed else request

    async def _rewrite_boundary(
        self,
        current: Request,
        active: set[int],
        node: int | None,
    ) -> tuple[Request, list[PendingRewrite]]:
        calls = _tool_calls((*(n.message for n in self.trace.nodes), *current.messages))
        records: list[PendingRewrite] = []
        for handler in self.request_interceptors if active else ():
            raw = await call_hook(
                handler, {Request: current.model_copy(deep=True), Trace: self.trace}
            )
            if raw is None:
                continue
            result, info = unpack_hook(raw, Request)
            changed = _diff_request_rewrite(current, result, active)
            if not changed:
                continue
            current = result
            records.extend(
                _make_record(handler, node, pos, calls, current.messages[pos], info)
                for pos in changed
            )
        return current, records

    async def _check_request_stops(self, current: Request) -> StopVerdict | None:
        for stop in self.request_stops:
            triggered, info = unpack_hook(
                await call_hook(
                    stop, {Request: current.model_copy(deep=True), Trace: self.trace}
                ),
                bool,
            )
            if triggered:
                return stop.__name__, info
        return None

    async def rewrite_request(
        self,
        request: Request,
        *,
        run_stops: bool = True,
        dialect: DialectName | None = None,
        known_path: Sequence[int] = (),
        node: int | None = None,
    ) -> tuple[Request, list[PendingRewrite], StopVerdict | None]:
        """Run typed request interceptors and stops over one canonical request. Only the
        uncommitted tail's user and tool messages may change; a rewritten tool result is
        pinned for the rest of the rollout."""
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
        records: list[PendingRewrite] = []
        previous = 0
        try:
            for end in boundaries:
                active = {pos for pos in candidates if previous <= pos < end}
                current = request.model_copy(update={"messages": messages[:end]})
                current, step_records = await self._rewrite_boundary(
                    current, active, node
                )
                records.extend(step_records)
                messages[:end] = current.messages
                if run_stops and (stopped := await self._check_request_stops(current)):
                    return (
                        request.model_copy(update={"messages": messages}),
                        records,
                        stopped,
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
        pinned_positions: dict[int, ToolResultKey] = {}
        for position in candidates:
            after = current.messages[position]
            if (
                isinstance(after, ToolMessage)
                and after != request.messages[position]
                and position in tool_keys
            ):
                tkey = tool_keys[position]
                self.pinned_tool_results[tkey] = after
                pinned_positions[position] = tkey
        if node is None and pinned_positions:
            replaced = set(pinned_positions.values())
            self.pinned_rewrites = [
                item for item in self.pinned_rewrites if item[0] not in replaced
            ]
            for pos, rec in records:
                if pos in pinned_positions:
                    self.pinned_rewrites.append((pinned_positions[pos], rec))
        return current, records, None

    def stage_rewrites(
        self, replay_key: str, records: Iterable[PendingRewrite]
    ) -> None:
        """Hold in-request rewrites in case the model call fails before commit."""
        staged = [(pos, rec) for pos, rec in records if pos is not None]
        if staged:
            self.inflight_rewrites[replay_key] = staged
        else:
            self.inflight_rewrites.pop(replay_key, None)

    def commit_rewrites(
        self,
        turn: graph.PendingTurn,
        last_prompt_node: int | None,
        request_rewrites: Iterable[PendingRewrite] = (),
        response_rewrites: Iterable[InterceptRecord] = (),
        assistant_node: int | None = None,
        *,
        pre_request: Request | None = None,
        replay_key: str | None = None,
    ) -> None:
        """Consume prepared user turns and attach committed node ids in execution order."""
        pending_in_turn = list(
            self.inflight_rewrites.pop(replay_key, ())
            if replay_key is not None
            else request_rewrites
        )
        for message in turn.tail:
            if isinstance(message, UserMessage) and self.prepared_users:
                key = graph.message_hash(message)
                if self.prepared_users[key]:
                    self.prepared_users[key] -= 1
        prompt_nodes = (
            []
            if last_prompt_node is None
            else graph.path_node_ids(self.trace, last_prompt_node)
        )
        tail_nodes = prompt_nodes[turn.tail_start :]
        base = pre_request if pre_request is not None else Request(messages=turn.prompt)
        node_by_pos = _align_nodes(base.messages, turn.prompt, prompt_nodes)
        in_turn: list[InterceptRecord] = []
        in_turn_ids: set[int] = set()
        in_turn_nodes: set[int] = set()
        for pos, rec in pending_in_turn:
            if pos is None:
                continue
            nid = node_by_pos.get(pos)
            rec.node = nid
            if nid is not None:
                in_turn_nodes.add(nid)
            in_turn.append(rec)
            in_turn_ids.add(id(rec))
        aligned_nodes = set(node_by_pos.values())
        users = [
            nid
            for nid in tail_nodes
            if nid in aligned_nodes
            and isinstance(self.trace.nodes[nid].message, UserMessage)
            and nid not in in_turn_nodes
        ]
        if self._prepared_user_count and users:
            if self.prepared_rewrites:
                targets = users[-self._prepared_user_count :]
                for slot, rec in self.prepared_rewrites:
                    rec.node = targets[min(slot, len(targets) - 1)]
                    self.trace.request_rewrites.append(rec)
                self.prepared_rewrites.clear()
            self._prepared_user_count = 0
        if self.pinned_rewrites:
            tool_nodes = {
                tkey: node_by_pos[pos]
                for pos, tkey in self.tool_result_keys(
                    base, turn.dialect, node_by_pos
                ).items()
                if pos in node_by_pos
            }
            if any(
                tkey not in tool_nodes and isinstance(tkey[0], str)
                for tkey, _ in self.pinned_rewrites
            ):
                for pos, tkey in self.tool_result_keys(base, turn.dialect).items():
                    if pos in node_by_pos:
                        tool_nodes.setdefault(tkey, node_by_pos[pos])
            remaining: list[tuple[ToolResultKey, InterceptRecord]] = []
            committed_pinned: set[int] = set()
            for tkey, rec in self.pinned_rewrites:
                if id(rec) in in_turn_ids:
                    continue
                if (nid := tool_nodes.get(tkey)) is None:
                    remaining.append((tkey, rec))
                    continue
                rec.node = nid
                self.trace.request_rewrites.append(rec)
                committed_pinned.add(id(rec))
            self.pinned_rewrites = remaining
            if committed_pinned and self.inflight_rewrites:
                self.inflight_rewrites = {
                    key: kept
                    for key, items in self.inflight_rewrites.items()
                    if (
                        kept := [
                            item
                            for item in items
                            if id(item[1]) not in committed_pinned
                        ]
                    )
                }
        self.trace.request_rewrites.extend(in_turn)
        for rec in response_rewrites:
            rec.node = assistant_node
            self.trace.response_rewrites.append(rec)

    def flush_rewrites(self) -> None:
        """Flush any request rewrites whose prompt never committed (`node=None`)."""
        seen = {id(rec) for rec in self.trace.request_rewrites}
        pending = chain(
            (rec for _, rec in self.prepared_rewrites),
            (rec for _, rec in self.pinned_rewrites),
            (rec for _, rec in chain.from_iterable(self.inflight_rewrites.values())),
        )
        for rec in pending:
            if id(rec) not in seen:
                seen.add(id(rec))
                self.trace.request_rewrites.append(rec)
        self.prepared_rewrites.clear()
        self._prepared_user_count = 0
        self.pinned_rewrites.clear()
        self.inflight_rewrites.clear()

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
        rewritten, pending, _ = await self.rewrite_request(
            Request(messages=[*branch, *request.messages]),
            run_stops=False,
            known_path=known_path,
        )
        tail = rewritten.messages[len(branch) :]
        user_slots: dict[int, int] = {}
        for offset, message in enumerate(tail):
            if isinstance(message, UserMessage):
                self.prepared_users[graph.message_hash(message)] += 1
                user_slots[len(branch) + offset] = self._prepared_user_count
                self._prepared_user_count += 1
        records: list[InterceptRecord] = []
        for pos, rec in pending:
            records.append(rec)
            if pos in user_slots:
                self.prepared_rewrites.append((user_slots[pos], rec))
        return Request(messages=tail), records

    async def rewrite_response(
        self, response: Response
    ) -> tuple[Response, list[InterceptRecord], StopVerdict | None]:
        """Run typed response interceptors and stops before harness delivery."""
        records: list[InterceptRecord] = []
        try:
            for handler in self.response_interceptors:
                candidate = response.model_copy(deep=True)
                raw = await call_hook(handler, {Response: candidate, Trace: self.trace})
                if raw is None:
                    continue
                result, info = unpack_hook(raw, Response)
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
                records.append(InterceptRecord(handler=handler.__name__, info=info))

            for stop in self.response_stops:
                candidate = response.model_copy(deep=True)
                triggered, info = unpack_hook(
                    await call_hook(stop, {Response: candidate, Trace: self.trace}),
                    bool,
                )
                if triggered:
                    return response, records, (stop.__name__, info)
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
    ) -> StopVerdict | None:
        """Police the tool calls the assistant committed at `node` before the harness sees
        them: the request hooks run over the branch plus an empty result for each call, so a
        `@stop` refuses the whole response and a rewrite becomes that call's verdict — the
        harness's gate denies it and the model sees the rewritten result instead. Returns
        the stop verdict that fired, if any."""
        assistant = self.trace.nodes[node].message
        if not (self.request_interceptors or self.request_stops):
            return None
        if dialect is not None:
            self.tool_dialects[node] = dialect
        if not isinstance(assistant, AssistantMessage) or not assistant.tool_calls:
            return None
        known_path = graph.path_node_ids(self.trace, node)
        branch = [self.trace.nodes[i].message for i in known_path]
        tools = self.trace.nodes[known_path[0]].tools
        # Live harnesses replay their own input. A resumed harness may replay the
        # model-facing transcript, including any policy notices or rewritten user turns.
        histories = [branch]
        if harness_history is not None and harness_history != branch[:-1]:
            histories.append([*harness_history, assistant])
        for history in histories:
            prefix = hashlib.blake2b(digest_size=16)
            for message in history:
                prefix.update(
                    graph.replay_message_hash(self.trace, message, dialect).encode()
                )
            prefix_hash = prefix.hexdigest()
            for call in assistant.tool_calls:
                self.tool_issuers.setdefault((prefix_hash, call.id), set()).add(node)
        for call in assistant.tool_calls:
            probe = ToolMessage(tool_call_id=call.id, content="", name=call.name)
            request = Request(messages=[*branch, probe], tools=tools or None)
            key = (node, call.id)
            if key in self.tool_decisions:
                raise HarnessError(
                    f"tool call id {call.id!r} was reused in the same assistant turn"
                )
            request, pending, stopped = await self.rewrite_request(
                request, dialect=dialect, known_path=known_path, node=node
            )
            records = [rec for _, rec in pending]
            self.trace.request_rewrites.extend(records)
            if stopped is not None:
                return stopped
            verdict = request.messages[-1]
            assert isinstance(verdict, ToolMessage)
            if verdict == probe:
                self.tool_decisions[key] = None
                continue
            if not self.gates_tools:
                # The harness would run the call regardless; only ending the rollout keeps
                # the transcript and the sandbox in agreement.
                return records[-1].handler, records[-1].info
            self.tool_decisions[key] = verdict
        return None

    async def decide_tool(
        self, tool_call_id: str, name: str | None = None, arguments: object = None
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
            key = (leaf, call.id)
            self.gated_tools.add(key)
            verdict = self.tool_decisions.get(key)
        elif matches or not name or len(leaves) != 1:
            raise HarnessError(
                f"tool gate asked about {tool_call_id!r}, which matches "
                f"{len(matches)} proposed calls"
            )
        else:
            spawn_node = leaves[0][0]
            known_path = graph.path_node_ids(self.trace, spawn_node)
            *branch, assistant = [self.trace.nodes[i].message for i in known_path]
            assert isinstance(assistant, AssistantMessage)
            args = arguments if isinstance(arguments, str) else json.dumps(arguments)
            call = ToolCall(id=tool_call_id, name=name, arguments=args)
            assistant = assistant.model_copy(
                update={"tool_calls": [*(assistant.tool_calls or []), call]}
            )
            probe = ToolMessage(tool_call_id=call.id, content="", name=name)
            request, pending, stopped = await self.rewrite_request(
                Request(
                    messages=[*branch, assistant, probe],
                    tools=self.trace.nodes[known_path[0]].tools or None,
                ),
                dialect=self.tool_dialects.get(spawn_node),
                known_path=known_path[:-1],
                node=spawn_node,
            )
            records = [rec for _, rec in pending]
            self.trace.request_rewrites.extend(records)
            if stopped is not None:
                stop_name, stop_info = stopped
                self.trace.stop(stop_name, info=stop_info)
                self.trace.notify()
                return {"action": "stop", "reason": stop_name}
            if records:
                self.trace.notify()
            verdict = request.messages[-1]
            assert isinstance(verdict, ToolMessage)
            if verdict == probe:
                verdict = None
        if verdict is None:
            return {"action": "allow"}
        return {"action": "deny", "message": verdict.model_dump(exclude_none=True)}

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
        for task in list(self.tasks):
            task.cancel()
        self.flush_rewrites()

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
            try:
                triggered, info = unpack_hook(
                    await call_hook(stop, {Trace: self.trace}), bool
                )
            except RolloutError:
                raise
            except Exception as error:
                raise TaskError(
                    f"@stop {stop.__name__} failed: {type(error).__name__}: {error}"
                ) from error
            if triggered:
                self.trace.stop(stop.__name__, info=info)
                logger.debug("Stopped by @stop %s", stop.__name__)
                return stop.__name__
        return None
