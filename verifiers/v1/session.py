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
import logging
import time
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
from verifiers.v1.harnesses.utils.compaction import bound_tool_message
from verifiers.v1.runtimes.base import Runtime
from verifiers.v1.trace import InterceptRecord, SteerRecord, Trace
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

logger = logging.getLogger(__name__)


def hook_boundary(handler: Callable, *, allow_trace: bool) -> type:
    """Select a hook boundary solely from its annotated parameters. A `Response` hook
    may also take the `Request` it answers."""
    hints = get_type_hints(handler)
    annotations = [
        get_origin(hints[name]) or hints[name]
        for name in inspect.signature(handler).parameters
        if name in hints
    ]
    requests, responses = annotations.count(Request), annotations.count(Response)
    if annotations.count(Trace) <= 1:
        if responses == 1 and requests <= 1:
            return Response
        if requests == 1 and responses == 0:
            return Request
        if allow_trace and requests == responses == 0 and Trace in annotations:
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
class PendingSteer:
    """A steer delivered through the interception: added as a user message to the
    rollout's model requests. The harness never learns of it, so it keeps sending its
    own conversation; the message is re-inserted at the same place (its anchor) on every
    later request that extends that conversation, until the harness's own request
    carries it."""

    record: SteerRecord
    message: UserMessage
    anchor: tuple[int, str] | None = None
    """Where it goes: before message `index` of the harness's own request, whose
    messages before it digest to `prefix`. None until a request first carries it."""
    prepared: bool = False
    """Counted in `prepared_users`, so request interceptors leave it alone."""
    echoed: bool = False
    """The harness's own request carries it: never inserted again."""


def _prefix_digests(messages: Messages) -> list[str]:
    """`digests[i]` identifies `messages[:i]`."""
    digest = hashlib.blake2b(digest_size=16)
    digests = [digest.hexdigest()]
    for message in messages:
        digest.update(graph.message_hash(message).encode())
        digests.append(digest.hexdigest())
    return digests


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
    on_commit: Callable[[], None] | None = None
    """Called once right after the next model call commits to the trace (its request
    and reply are on the trace), then cleared: per-call saves report from here."""
    request_stops: list[Callable] = field(default_factory=list)
    response_stops: list[Callable] = field(default_factory=list)
    on_turn_input: Callable[[], None] | None = None
    """One-shot acknowledgement when a resumed harness reaches model inference."""
    runtime: Runtime | None = None
    """The rollout's box once it is up, for hooks that take a `Runtime`."""
    gates_tools: bool = False
    """Whether the harness asks `/tool` before executing each call (see
    `Harness.SUPPORTS_TOOL_INTERCEPTION`). Without that gate a pre-execution rewrite
    cannot be enforced, so it ends the rollout instead of letting the call run."""
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
    released: bool = False
    """Set when the rollout unregisters the session: the trace is sealed (its conclusion is
    what scored and persisted), so a handler still in flight must not commit turns, record
    calls, or write state onto it — the in-memory trace must stay what the run produced."""
    tasks: set["asyncio.Task"] = field(default_factory=set)
    """Handler tasks currently serving this session. aiohttp does not cancel a handler when
    its client disconnects, so a request whose program died at teardown would keep driving
    the exchange (upstream call, simulator turn) — unregistering cancels these instead."""
    tool_decisions: dict[tuple[str, str], ToolMessage | None] = field(
        default_factory=dict
    )
    """The pre-execution verdict per proposed call occurrence: None to run it, else the result
    the policy put in its place — that call must not execute."""
    pinned_tool_results: dict[tuple[str, str], ToolMessage] = field(
        default_factory=dict
    )
    """The results the model must see, by conversation prefix and call id: rewrites and results it
    substituted for blocked calls. The harness keeps its own copy of each in its history and
    re-sends it forever, so every request has these re-applied before the graph is matched —
    the trace holds the pinned version, and a differing copy would fork it at that node."""
    gated_tools: set[tuple[str, str]] = field(default_factory=set)
    """Calls the harness's gate asked about before executing."""
    prepared_users: Counter[str] = field(default_factory=Counter)
    steers: list[PendingSteer] = field(default_factory=list)
    """Steers delivered through the interception, in arrival order (`queue_steer`)."""
    steer_peer: Callable[[str, str], Awaitable[dict]] | None = None
    """Steers a sibling agent of this rollout's episode by seat name, `(seat, text)` to a
    receipt — set when the agent's `message_agents` tool is enabled."""

    @property
    def stopped(self) -> bool:
        return self.trace.stop_condition is not None

    def queue_steer(
        self, text: str, *, message_id: str | None = None, sender: str | None = None
    ) -> dict:
        """Accept a steer for the rollout's next model request. A repeated
        `message_id` with the same text is the same steer."""
        if message_id is not None:
            for steer in self.steers:
                if steer.record.id == message_id:
                    if steer.record.text != text:
                        raise ValueError(
                            "steering message ID reused with different content"
                        )
                    return {"outcome": "injected", "delivery": "interception"}
        record = SteerRecord(
            id=message_id,
            text=text,
            sender=sender,
            time=time.time(),
            delivery="interception",
        )
        self.trace.steers.append(record)
        self.steers.append(PendingSteer(record, UserMessage(content=text)))
        self.trace.notify()
        return {"outcome": "injected", "delivery": "interception"}

    def place_steers(
        self, request: Request
    ) -> tuple[list[tuple[int, UserMessage]], list[PendingSteer]]:
        """Where this harness request needs steer messages: `(index, message)` pairs, each
        inserted before `request.messages[index]` (in list order at equal indexes), and
        the steers placed. A new steer is anchored at the end of the first request that
        carries it; an anchored one is re-inserted whenever the request extends the
        conversation it was anchored to and does not already carry it. A request on
        another branch (a subagent, a compacted history) gets nothing."""
        live = [steer for steer in self.steers if not steer.echoed]
        if not live:
            return [], []
        messages = request.messages
        digests = _prefix_digests(messages)
        placed: list[PendingSteer] = []
        for steer in live:
            if steer.anchor is None:
                steer.anchor = (len(messages), digests[-1])
            index, prefix = steer.anchor
            if index >= len(digests) or digests[index] != prefix:
                continue
            key = graph.message_hash(steer.message)
            if any(
                isinstance(message, UserMessage) and graph.message_hash(message) == key
                for message in messages[index:]
            ):
                steer.echoed = True
                continue
            if self.request_interceptors and not steer.prepared:
                # Framework input, like a caller's prepared turn: hooks leave it alone.
                self.prepared_users[key] += 1
                steer.prepared = True
            placed.append(steer)
        inserts = sorted(
            ((steer.anchor[0], steer.message) for steer in placed if steer.anchor),
            key=lambda insert: insert[0],
        )
        return inserts, placed

    def land_steers(self, steers: list[PendingSteer], node: int, call: int) -> None:
        """Record that `steers` reached the model in the call committing `node`."""
        landed = False
        for steer in steers:
            if steer.record.call is not None:
                continue
            steer.record.call = call
            key = graph.message_hash(steer.message)
            parent = self.trace.nodes[node].parent
            while parent is not None:
                candidate = self.trace.nodes[parent]
                if (
                    not candidate.sampled
                    and graph.message_hash(candidate.message) == key
                ):
                    steer.record.node = parent
                    break
                parent = candidate.parent
            landed = True
        if landed:
            self.trace.notify()

    def tool_result_keys(self, request: Request) -> dict[int, tuple[str, str]]:
        """Identify results by their issuing conversation prefix and call id. Canonical
        pinned history gives native and rewritten replays the same occurrence keys."""
        prefix = hashlib.blake2b(digest_size=16)
        calls: dict[str, tuple[str, str]] = {}
        keys: dict[int, tuple[str, str]] = {}
        for position, message in enumerate(request.messages):
            if isinstance(message, ToolMessage) and message.tool_call_id in calls:
                key = calls[message.tool_call_id]
                keys[position] = key
                message = self.pinned_tool_results.get(key, message)
            prefix.update(graph.message_hash(message).encode())
            if isinstance(message, AssistantMessage):
                calls.update(
                    (call.id, (prefix.hexdigest(), call.id))
                    for call in message.tool_calls or []
                )
        return keys

    def pin_tool_results(self, request: Request) -> Request:
        """Re-impose the pinned results on the harness's request. A result for a blocked call
        the harness never gated means the call ran anyway: the transcript would no longer
        describe the sandbox, so that fails the rollout instead of being papered over."""
        messages = list(request.messages)
        changed = False
        for position, key in self.tool_result_keys(request).items():
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

    async def rewrite_request(
        self, request: Request, *, run_stops: bool = True
    ) -> tuple[Request, list[InterceptRecord], str | None]:
        """Run typed request interceptors and stops over one canonical request. Only the
        uncommitted tail's user and tool messages may change; a rewritten tool result is
        pinned for the rest of the rollout."""
        request = self.pin_tool_results(request)
        tool_keys = self.tool_result_keys(request)
        if not self.request_interceptors and (not run_stops or not self.request_stops):
            return request, [], None
        candidates: set[int] = set()
        tail_start = graph.message_prefix_len(self.trace, request.messages)
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
                        handler,
                        {Request: candidate, Trace: self.trace, Runtime: self.runtime},
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
                        stop,
                        {Request: candidate, Trace: self.trace, Runtime: self.runtime},
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
        tool_keys = self.tool_result_keys(current)
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
        self, response: Response, request: Request
    ) -> tuple[Response, list[InterceptRecord], str | None]:
        """Run typed response interceptors and stops before harness delivery. They may
        also read the `request` the response answers."""
        records: list[InterceptRecord] = []
        try:
            for handler in self.response_interceptors:
                candidate = response.model_copy(deep=True)
                result = await call_hook(
                    handler,
                    {
                        Response: candidate,
                        Request: request.model_copy(deep=True),
                        Trace: self.trace,
                        Runtime: self.runtime,
                    },
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
                result = await call_hook(
                    stop,
                    {
                        Response: candidate,
                        Request: request.model_copy(deep=True),
                        Trace: self.trace,
                        Runtime: self.runtime,
                    },
                )
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

    async def gate_tool_calls(self, node: int) -> str | None:
        """Police the tool calls the assistant committed at `node` before the harness sees
        them: the request hooks run over the branch plus an empty result for each call, so a
        `@stop` refuses the whole response and a rewrite becomes that call's verdict — the
        harness's gate denies it and the model sees the rewritten result instead. Returns
        the name of the stop that fired, if any."""
        assistant = self.trace.nodes[node].message
        if not (self.request_interceptors or self.request_stops) or not (
            isinstance(assistant, AssistantMessage) and assistant.tool_calls
        ):
            return None
        branch = graph.path(self.trace, node)
        for call in assistant.tool_calls:
            probe = ToolMessage(tool_call_id=call.id, content="", name=call.name)
            request = Request(messages=[*branch, probe], tools=self.trace.tools or None)
            key = self.tool_result_keys(request)[len(branch)]
            if key in self.tool_decisions:
                raise HarnessError(
                    f"tool call id {call.id!r} was reused in the same assistant turn"
                )
            request, records, stopped = await self.rewrite_request(request)
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
                return records[-1].handler
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
            branch = graph.path(self.trace, leaf)
            request = Request(
                messages=[*branch, ToolMessage(tool_call_id=call.id, content="")]
            )
            key = self.tool_result_keys(request)[len(branch)]
            self.gated_tools.add(key)
            verdict = self.tool_decisions.get(key)
        elif matches or not name or len(leaves) != 1:
            raise HarnessError(
                f"tool gate asked about {tool_call_id!r}, which matches "
                f"{len(matches)} proposed calls"
            )
        else:
            *branch, assistant = graph.path(self.trace, leaves[0][0])
            assert isinstance(assistant, AssistantMessage)
            call = ToolCall(id=tool_call_id, name=name, arguments=json.dumps(arguments))
            assistant = assistant.model_copy(
                update={"tool_calls": [*(assistant.tool_calls or []), call]}
            )
            probe = ToolMessage(tool_call_id=call.id, content="", name=name)
            request, records, stopped = await self.rewrite_request(
                Request(
                    messages=[*branch, assistant, probe], tools=self.trace.tools or None
                )
            )
            self.trace.request_rewrites.extend(records)
            if stopped is not None:
                self.trace.stop(stopped)
                return {"action": "stop", "reason": stopped}
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
            result = await call_hook(stop, {Trace: self.trace, Runtime: self.runtime})
            if not isinstance(result, bool):
                raise TaskError(f"@stop must return bool, got {type(result).__name__}")
            if result:
                self.trace.stop(stop.__name__)
                logger.debug("stop %r fired: id=%s", stop.__name__, self.trace.id)
                return stop.__name__
        return None
