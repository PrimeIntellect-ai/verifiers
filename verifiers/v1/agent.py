"""The Agent: a reusable (harness x model x runtime) value with one executable
arrow — `agent.run(task) -> Trace`; `runtime=` borrows a live box,
`provision(task)` hands you one. `agent.interaction(task)` holds the rollout open
turn-by-turn, with the caller as the run's user — one `turn()` per harness segment;
who computes the turns is control flow, not a framework concept.
Inject a live `Interception` to share servers across agents (a pool belongs to
what spans agents, never to one agent); an entered agent (`async with`) owns one
server; un-entered, each run brings its own."""

import asyncio
import inspect
import logging
from collections.abc import AsyncIterator, Awaitable, Callable, Iterable, Mapping
from contextlib import (
    AbstractAsyncContextManager,
    AsyncExitStack,
    asynccontextmanager,
    nullcontext,
)
from contextvars import ContextVar
from dataclasses import dataclass, field, replace
from typing import Any, Generic, Literal, Self, cast
from weakref import WeakValueDictionary

from typing_extensions import TypeVar

from verifiers.v1.box import Box, box_of, provision_box
from verifiers.v1.clients import (
    EvalClientConfig,
    ModelContext,
)
from verifiers.v1.configs.agent import (
    AgentConfig,
    TimeoutConfig,
    agent_config_fields,
    resolve_agent,
)
from verifiers.v1.configs.runtime import NetworkPolicyConfig
from verifiers.v1.dialects import parse_message
from verifiers.v1.durable import (
    BOX_LABEL,
    CALL_LABEL,
    Durable,
    Resolved,
    TraceSink,
    box_checkpoint,
    box_parts,
    box_ref,
    call_parts,
    fingerprint,
    merge_labels,
    observe,
    plan_call,
    resolve_call,
)
from verifiers.v1.errors import SandboxError
from verifiers.v1.harness import Harness
from verifiers.v1.interception import Interception, InterceptionServer
from verifiers.v1.mcp import SharedToolServer
from verifiers.v1.rollout import OnSave, Rollout, RolloutTimeouts, messaging_toolset
from verifiers.v1.runtimes import (
    Runtime,
    RuntimeConfig,
    SubprocessConfig,
    restore_config,
    runtime_is_local,
)
from verifiers.v1.session import RolloutLimits
from verifiers.v1.task import Task
from verifiers.v1.trace import Checkpoint, Trace
from verifiers.v1.types import (
    AssistantMessage,
    Messages,
    ToolMessage,
    UserMessage,
)
from verifiers.v1.utils.aio import run_shielded
from verifiers.v1.utils.compile import (
    cap_remote_agent_timeout,
    resolve_runtime_config,
    validate_pairing,
)
from verifiers.v1.utils.retries import (
    MODEL_RETRY_NOTE,
    RETRY_NOTE,
    RetryState,
    backoff,
    retryable_error,
)

__all__ = [
    "Agent",
    "AgentConfig",
    "Agents",
    "TimeoutConfig",
    "make_agent",
]

logger = logging.getLogger(__name__)

# Only nested runs in this scoring attempt participate in its rollback.
_scoring_runs: ContextVar[list[tuple["_EpisodeAgent", Trace]] | None] = ContextVar(
    "scoring_runs", default=None
)


def resolve_rollout_timeouts(timeout: TimeoutConfig, task: Task) -> RolloutTimeouts:
    """Apply an agent's stage-timeout precedence to one task."""
    return RolloutTimeouts(
        setup=timeout.setup if timeout.setup is not None else task.data.timeout.setup,
        agent=(
            timeout.rollout if timeout.rollout is not None else task.data.timeout.agent
        ),
        finalize=(
            timeout.finalize
            if timeout.finalize is not None
            else task.data.timeout.finalize
        ),
        scoring=(
            timeout.scoring
            if timeout.scoring is not None
            else task.data.timeout.scoring
        ),
    )


def _check_borrowed_placement(
    task: Task, runtime: Runtime, base_config: RuntimeConfig
) -> None:
    """A borrowed box is never re-provisioned, so a task's placement fields can't
    be honored. Reject requirements that cannot be applied to the running box; an
    image mismatch on a container only warns, since sharing its world is the point."""
    task_policy = "*" not in task.data.network_allow or bool(task.data.network_block)
    base_policy = base_config if isinstance(base_config, NetworkPolicyConfig) else None
    if task_policy or (base_policy is not None and base_policy.network_restricted):
        config = runtime.config
        if not isinstance(config, NetworkPolicyConfig):
            raise ValueError(
                f"task {task.data.idx!r} requires a framework-aware network policy, "
                f"but borrowed runtime {runtime.name!r} does not support one; use "
                "agent.provision(task)"
            )
        if base_policy is not None and type(config) is not type(base_policy):
            raise ValueError(
                f"the configured {base_policy.type} network policy cannot be applied "
                f"to borrowed {config.type} runtime {runtime.name!r}; use "
                "agent.provision(task)"
            )
        policy_base = base_policy or config.model_copy(
            update={"allow": ["*"], "block": []}
        )
        expected = resolve_runtime_config(policy_base, task)
        assert isinstance(expected, NetworkPolicyConfig)
        # Do not inherit extra destinations from a box provisioned for another task.
        if set(config.allow) != set(expected.allow) or set(config.block) != set(
            expected.block
        ):
            raise ValueError(
                f"task {task.data.idx!r} requires allow={expected.allow!r} and "
                f"block={expected.block!r}, but borrowed runtime {runtime.name!r} "
                f"has allow={config.allow!r} and block={config.block!r}; use "
                "agent.provision(task)"
            )
    if task.data.image is None:
        return
    if isinstance(runtime.config, SubprocessConfig):
        raise TypeError(
            f"task {task.data.idx!r} requires image {task.data.image!r}, but the "
            "borrowed runtime is subprocess-backed (no container); borrow a container "
            "box (e.g. agent.provision(task)) or drop the task's image"
        )
    box_image = getattr(runtime.config, "image", None)
    if box_image != task.data.image:
        logger.warning(
            "task %r requires image %r, but borrowed box %r runs %r; a borrowed box "
            "is never re-provisioned, so the run proceeds in the box's world",
            task.data.idx,
            task.data.image,
            runtime.name,
            box_image,
        )


@dataclass(frozen=True)
class Segment:
    """One harness segment's agent/tool output, as `Interaction.turn` returns it.

    `messages` carries every model-sampled assistant message and intervening tool
    result produced by the segment, in order. `root_reply` identifies the root
    agent's response when the trace also contains descendant branches.
    `terminated` marks the exchange over — the run ended (a
    limit, a `@stop`, or the harness finishing) instead of producing another
    segment; a terminated `Segment` carries no messages (the last real segment was
    already delivered), and the interaction's `trace` holds the full exchange.
    """

    messages: Messages
    root_reply: str | None = None
    terminated: bool = False

    @property
    def last_reply(self) -> str:
        """The root agent's response, falling back to the final assistant message."""
        if self.root_reply is not None:
            return self.root_reply.strip()
        for message in reversed(self.messages):
            if isinstance(message, AssistantMessage):
                return (message.content or "").strip()
        return ""


@dataclass(frozen=True)
class _Opening:
    """Where a rollout starts: its task (prompt resolved), the checkpoint its box
    starts from, and a conversation the first caller turn continues (a reopened trace
    waiting on its user)."""

    task: Task
    checkpoint: str | None = None
    prefix: Messages | None = None
    source: Trace | None = None
    """The trace a `resume=` continues."""


Reopen = Callable[[Task, str | None, Trace, "Rollout"], Awaitable["Rollout"]]
"""Build and open a replacement rollout: `(task, checkpoint, the trace it replaces,
the rollout it replaces)`."""


@dataclass
class _TurnMark:
    """Where the running turn's input lives, for an interaction retry: the trace it
    was sent on, that trace's turns before it, and whether the trace's opening prompt
    already holds it (a replacement restored during this turn)."""

    trace: Trace
    turns: int
    in_prompt: bool = False


@dataclass
class _PendingMessage:
    text: str
    message_id: str | None
    receipt: dict | None = None
    error: BaseException | None = None
    done: asyncio.Event = field(default_factory=asyncio.Event)


class Interaction:
    """An agent's rollout, held open turn-by-turn: the caller IS the run's user.

    `agent.interaction(task)` opens the rollout; `await interaction.turn("...")`
    sends one user turn and runs ONE harness segment — the
    program, resumed onto the conversation, until it yields — returning the
    resulting `Segment`. A prompt-less (or masked) task is opened by the first
    `turn(message)`; a prompted task speaks first — a bare `turn()` takes its
    opening reply. Turns are serialized; `steer(..., wake=True)` can run a
    background turn when idle. `interaction.trace` is live from
    the moment the interaction exists: watch tokens and turns mid-exchange, read
    rewards after close. Leaving the `interaction()` context closes the exchange
    as `user_closed` and finishes the rollout — hooks and scoring included.

    With `AgentConfig.interaction_retries`, a turn that fails on infrastructure runs
    again: on the same box after a model-side failure, else on a fresh box restored
    from the trace's latest checkpoint (`latest_checkpoint`; or the opening state):
    the failed attempt's rollout is closed with its errors (`failed_attempts`), and
    `interaction.trace` becomes the replacement's. A borrowed box is restored by its
    owner (`box.Box.replace`, from the latest checkpoint any seat on it took), so every
    seat on it continues on the restored box; it needs a checkpoint of this trace."""

    def __init__(
        self,
        run: "Rollout",
        gate: asyncio.Semaphore | None = None,
        *,
        opening: _Opening | None = None,
        reopen: Reopen | None = None,
        retries: int = 0,
        box: Box | None = None,
    ) -> None:
        self._run = run
        self._box = box
        self._gate = gate
        self._opening = opening
        self._prefix = opening.prefix if opening is not None else None
        self._reopen = reopen
        self._retries = retries
        self.failed_attempts: list[Trace] = []
        """Completed traces of attempts an interaction retry replaced, errors kept."""
        self._history_merged = False
        self._over = False  # a terminated segment was already delivered
        self._started = False  # a segment has run (the exchange is under way)
        self._lock = asyncio.Lock()
        self._steering_locks: WeakValueDictionary[str, asyncio.Lock] = (
            WeakValueDictionary()
        )
        self._steering_receipts: dict[str, tuple[str, dict]] = {}
        self._steering_tasks: dict[object, tuple[str, asyncio.Task[dict]]] = {}
        self._pending_messages: list[_PendingMessage] = []
        self._pending_by_id: dict[str, _PendingMessage] = {}
        self._message_drain: asyncio.Task[None] | None = None
        self._closing = False

    @property
    def trace(self) -> Trace:
        return self._run.trace

    @property
    def runtime(self) -> Runtime | None:
        """The box the current attempt runs on (a retry may replace it)."""
        return self._run.runtime

    @property
    def task(self) -> Task:
        """The task the current attempt runs."""
        return self._run.task

    @property
    def latest_checkpoint(self) -> Checkpoint | None:
        """The current attempt's latest checkpoint (`trace.checkpoints`), if any: a
        per-call save, a `checkpoint()`, or the one the attempt's box started from."""
        checkpoints = self.trace.checkpoints
        return checkpoints[-1] if checkpoints else None

    async def checkpoint(self) -> Checkpoint:
        """Checkpoint the interaction's box, between turns, and record it on the
        trace (`trace.checkpoints`), matching the conversation so far: what
        `agent.run(resume=trace)` / `agent.interaction(resume=trace)` continue from.
        Waits for a running turn to finish. A borrowed box is checkpointed through its
        owner (`box.Box`); one with no owner raises."""
        async with self._lock:
            run = self._run
            if run.closed:
                raise RuntimeError(
                    "this interaction is closed; checkpoint it while it is open"
                )
            if self._prefix is not None and not self.trace.nodes:
                raise ValueError(
                    "the reopened conversation is not on the trace until its first "
                    "turn; checkpoint after a turn"
                )
            return await run.checkpoint()

    async def send(
        self,
        message: str,
        *,
        mode: Literal["steer", "queue"] = "steer",
        message_id: str | None = None,
    ) -> dict:
        """Deliver a user message to this open interaction.

        ``steer`` delivers at the harness's next safe point during active work.
        ``queue`` waits for the current turn to yield, then delivers all pending
        messages together in arrival order in one new turn. It does not wake a
        wait tool inside the active turn. Messages arriving after the batch is
        drained wait for the following turn.
        Both resume an idle session. Queue mode also works without ACP steering.
        Sends return on acceptance, not turn completion. For queued sends and
        idle wake-ups, the first model request acknowledges the resumed input.
        Cancellation withdraws a queued message that has not started. After a batch starts, cancelling
        a sender does not cancel the shared turn. Callers must await or schedule
        this coroutine to submit input; closing the interaction cancels any
        outstanding delivery tasks.
        The receipt confirms delivery, not that the interaction is finished;
        keep its context open
        while more messages may arrive. A prompted task must start its opening
        turn first. Stable IDs deduplicate successful deliveries.
        A harness without native steering takes a steered message from its inbox
        before its next model call; one with neither cannot be steered
        (`Rollout.steer`).
        """
        if mode not in ("steer", "queue"):
            raise ValueError("message mode must be 'steer' or 'queue'")
        return await self._send(message, message_id=message_id, mode=mode, wake=True)

    async def steer(
        self,
        message: str,
        *,
        message_id: str | None = None,
        wake: bool = False,
    ) -> dict:
        """Inject a user message, optionally resuming an idle interaction.

        With ``wake=True``, an idle session runs another turn on this message.
        The interaction must remain open, and a prompted task must have started
        its opening turn. Wake-up uses the normal turn budget and lifecycle.
        Successful deliveries with a stable ID are deduplicated in this interaction.
        """
        return await self._send(message, message_id=message_id, mode="steer", wake=wake)

    async def _send(
        self,
        message: str,
        *,
        message_id: str | None,
        mode: Literal["steer", "queue"],
        wake: bool,
    ) -> dict:
        if not isinstance(message, str) or not message.strip():
            raise ValueError("delivery requires a nonempty user message")
        # Only retries of the same message share a lock. A wake-up can run a
        # full turn; unrelated messages must still reach that turn through ACP.
        lock = (
            self._steering_locks.setdefault(message_id, asyncio.Lock())
            if message_id is not None
            else nullcontext()
        )
        async with lock:
            if self._closing or self._run.closed:
                raise RuntimeError("this interaction is closed")
            if message_id is not None and message_id in self._steering_receipts:
                previous, receipt = self._steering_receipts[message_id]
                if previous != message:
                    raise ValueError(
                        "steering message ID reused with different content"
                    )
                return dict(receipt)
            if message_id is not None and message_id in self._pending_by_id:
                pending = self._pending_by_id[message_id]
                if pending.text != message:
                    raise ValueError(
                        "steering message ID reused with different content"
                    )
                await pending.done.wait()
                if pending.error is not None:
                    raise pending.error
                assert pending.receipt is not None
                return dict(pending.receipt)
            if message_id in self._steering_tasks:
                receipt = await self._deliver_steer(message, message_id)
            elif self._over or not self._run.ok:
                return {"outcome": "promptRequired", "reason": "interactionStopped"}
            elif mode == "steer":
                receipt = await self._deliver_steer(message, message_id)
            else:
                receipt = {"outcome": "promptRequired", "reason": "noRunningTurn"}
            if wake and receipt.get("outcome") == "promptRequired":
                receipt = await self._resume_messages(message, message_id)
            if receipt.get("outcome") == "injected" and message_id is not None:
                self._steering_receipts[message_id] = (message, dict(receipt))
            return receipt

    async def _deliver_steer(self, message: str, message_id: str | None) -> dict:
        key = message_id if message_id is not None else object()
        if key in self._steering_tasks:
            previous, task = self._steering_tasks[key]
            if previous != message:
                raise ValueError("steering message ID reused with different content")
        else:

            async def deliver() -> dict:
                try:
                    receipt = await self._run.steer(message, message_id=message_id)
                    if receipt.get("outcome") == "injected" and message_id is not None:
                        self._steering_receipts[message_id] = (message, dict(receipt))
                    return receipt
                finally:
                    self._steering_tasks.pop(key, None)

            # The harness may already have accepted input when the caller is
            # cancelled. Finish and cache that delivery before allowing retries.
            task = asyncio.create_task(deliver())
            self._steering_tasks[key] = (message, task)
            task.add_done_callback(
                lambda done: None if done.cancelled() else done.exception()
            )
        return await asyncio.shield(task)

    async def _cancel_deliveries(self) -> None:
        self._closing = True
        tasks = [task for _, task in self._steering_tasks.values()]
        if self._message_drain is not None:
            tasks.append(self._message_drain)
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        self._steering_tasks.clear()
        # A task cancelled before it starts never enters its finally block.
        self._finish_pending(asyncio.CancelledError())

    def _finish_pending(self, error: BaseException) -> None:
        for item in self._pending_messages:
            item.error = error
            item.done.set()
            if item.message_id is not None:
                self._pending_by_id.pop(item.message_id, None)
        self._pending_messages.clear()

    async def _resume_messages(self, message: str, message_id: str | None) -> dict:
        if self._closing:
            raise RuntimeError("this interaction is closed")
        pending = _PendingMessage(message, message_id)
        self._pending_messages.append(pending)
        if message_id is not None:
            self._pending_by_id[message_id] = pending
        if self._message_drain is None or self._message_drain.done():
            self._message_drain = asyncio.create_task(self._drain_messages())
            self._message_drain.add_done_callback(
                lambda done: None if done.cancelled() else done.exception()
            )
        try:
            await pending.done.wait()
            if pending.error is not None:
                raise pending.error
            assert pending.receipt is not None
            return dict(pending.receipt)
        finally:
            # Cancellation withdraws only input the worker has not drained.
            if message_id is not None and any(
                item is pending for item in self._pending_messages
            ):
                self._pending_by_id.pop(message_id, None)
            self._pending_messages = [
                item for item in self._pending_messages if item is not pending
            ]

    async def _drain_messages(self) -> None:
        try:
            while self._pending_messages:
                async with self._lock, self._gate or nullcontext():
                    batch, self._pending_messages = self._pending_messages, []
                    if not batch:
                        continue
                    try:
                        if self._closing or self._run.closed:
                            raise RuntimeError("this interaction is closed")
                        if self._over or not self._run.ok:
                            receipt = {
                                "outcome": "promptRequired",
                                "reason": "interactionStopped",
                            }
                        elif (
                            not self._started
                            and self.trace.task.data.prompt is not None
                        ):
                            receipt = {
                                "outcome": "promptRequired",
                                "reason": "noRunningTurn",
                            }
                        else:

                            def accepted(batch: list[_PendingMessage] = batch) -> None:
                                for item in batch:
                                    item.receipt = {"outcome": "injected"}
                                    if item.message_id is not None:
                                        self._steering_receipts[item.message_id] = (
                                            item.text,
                                            dict(item.receipt),
                                        )
                                        self._pending_by_id.pop(item.message_id, None)
                                    item.done.set()

                            segment = await self._turn(
                                [UserMessage(content=item.text) for item in batch],
                                on_input=accepted,
                            )
                            receipt = (
                                {
                                    "outcome": "promptRequired",
                                    "reason": "interactionStopped",
                                }
                                if segment.terminated
                                else {"outcome": "injected"}
                            )
                        for item in batch:
                            if item.done.is_set():
                                continue
                            item.receipt = receipt
                            if (
                                receipt.get("outcome") == "injected"
                                and item.message_id is not None
                            ):
                                self._steering_receipts[item.message_id] = (
                                    item.text,
                                    dict(receipt),
                                )
                    except BaseException as error:
                        for item in batch:
                            if not item.done.is_set():
                                item.error = error
                        raise
                    finally:
                        for item in batch:
                            if item.message_id is not None:
                                self._pending_by_id.pop(item.message_id, None)
                            item.done.set()
        except BaseException as error:
            self._finish_pending(error)
            raise

    async def turn(self, message: str | Messages | None = None) -> Segment:
        """Send one user turn (a string, or full `Messages` for multimodal /
        multi-message turns); run one segment; return its `Segment`. A
        prompted task speaks FIRST: take its opening reply with a bare `turn()`
        before answering. A `terminated` segment means the run ended instead of
        answering (the message went unconsumed)."""
        async with self._lock, self._gate or nullcontext():
            return await self._turn(message)

    async def _turn(
        self,
        message: str | Messages | None,
        *,
        on_input: Callable[[], None] | None = None,
    ) -> Segment:
        if self._closing or self._run.closed:
            raise RuntimeError("this interaction is closed")
        if self._over:
            raise RuntimeError(
                "the exchange is over (the run ended); read interaction.trace"
            )
        prompted = not self._started and self.trace.task.data.prompt is not None
        if message is None and not prompted:
            raise ValueError(
                "nothing to run a turn on: a bare turn() takes a prompted task's "
                "opening reply; this exchange takes its next user message"
            )
        if message is not None and prompted:
            raise ValueError(
                "the task's prompt opens this exchange: take its first reply with "
                "a bare turn() before answering (or hand the interaction a task "
                "with `prompt=None` to open the conversation yourself)"
            )
        messages: Messages | None = None
        if isinstance(message, str):
            messages = [UserMessage(content=message)]
        elif message is not None:
            # A turn's messages may arrive typed or as wire dicts (env code naturally
            # writes `{"role": "user", ...}`); the trace speaks typed, so normalize.
            messages = [parse_message(m) if isinstance(m, dict) else m for m in message]
        first = not self._started
        self._started = True
        step_messages = messages
        if self._prefix is not None:
            # A reopened trace waiting on its user: its conversation leads this
            # turn's relaunch (`Harness.resume` on the conversation).
            step_messages = [*self._prefix, *(messages or [])]
            self._prefix = None
        retries_left = (
            self._retries
            if self._reopen is not None and self._opening is not None
            else 0
        )
        turn_input = step_messages
        mark = _TurnMark(self.trace, self.trace.num_turns)
        # Only a failure of this turn is retried, never one a past turn left; a
        # replacement that failed to open is this turn's failure.
        was_ok = self._run.ok
        # Where this turn's segment starts: kept across a same-box retry, whose failed
        # attempt's committed messages are the segment's too; reset on a new trace.
        turns_before = self.trace.num_turns
        nodes_before = len(self.trace.nodes)
        while True:
            await self._run.step(step_messages, on_input=on_input)
            if retries_left == 0 or self._run.ok or not was_ok:
                break
            failure = self._run.failure
            if failure is None or not retryable_error(failure):
                break
            if await self._run.recover():
                # The model call failed but the box is fine: continue on it, from
                # the conversation as it stands, instead of restoring a checkpoint.
                retries_left -= 1
                was_ok = True
                logger.warning(
                    "interaction turn failed on a model call (%s: %s); continuing on "
                    "the same box",
                    type(failure).__name__,
                    failure,
                )
                if len(self.trace.nodes) > nodes_before:
                    step_messages = [UserMessage(content=MODEL_RETRY_NOTE)]
                await asyncio.sleep(backoff(self._retries - retries_left - 1))
                continue
            if not self._resumable(first):
                # Earlier turns changed the box and nothing checkpointed it: the
                # opening box no longer matches the conversation, so there is nothing
                # sound to resume.
                break
            retries_left -= 1
            was_ok = True
            step_messages, mark = await self._retry(
                failure,
                first=first,
                turn_input=turn_input,
                mark=mark,
            )
            turns_before = self.trace.num_turns
            nodes_before = len(self.trace.nodes)
        if self.trace.num_turns > turns_before:
            # The segment answered — even if a limit or @stop then ended the
            # exchange, that surfaces as the NEXT turn's terminated segment.
            segment_messages: Messages = []
            saw_assistant = False
            for node in self.trace.nodes[nodes_before:]:
                if node.sampled:
                    segment_messages.append(node.message)
                    saw_assistant = True
                elif saw_assistant and isinstance(node.message, ToolMessage):
                    segment_messages.append(node.message)
            return Segment(
                messages=segment_messages,
                root_reply=self.trace.root_reply,
            )
        self._over = True
        return Segment(messages=[], terminated=True)

    def _reopens_opening(self, first: bool) -> bool:
        """Whether a retry reopens the interaction exactly as it opened: a first
        segment of the original attempt with nothing checkpointed past its opening."""
        if self._box is not None:
            # A borrowed box's opening state is whatever its seats left on it.
            return False
        checkpoint = self.latest_checkpoint
        return first and (
            checkpoint is None or (checkpoint.turns == 0 and not self.failed_attempts)
        )

    def _resumable(self, first: bool) -> bool:
        """Whether a failed turn has a sound state to resume from."""
        if self._box is not None:
            return self.latest_checkpoint is not None and self._box.latest is not None
        return self._reopens_opening(first) or self.latest_checkpoint is not None

    async def _retry(
        self,
        failure: Exception,
        *,
        first: bool,
        turn_input: Messages | None,
        mark: _TurnMark,
    ) -> tuple[Messages | None, _TurnMark]:
        """Replace the failed rollout with one on a fresh box; return the messages the
        retried step takes and where the turn's input now lives. A first segment with
        nothing checkpointed past its opening reopens exactly as the interaction opened
        and runs the same step. Otherwise the replacement starts from the trace's latest
        checkpoint and the conversation cut at it (`Trace.conversation_at`), opened with
        `RETRY_NOTE`, plus this turn's input when the cut predates it. A later turn with
        no checkpoint is not retried: the opening box no longer matches the
        conversation."""
        assert self._reopen is not None and self._opening is not None
        checkpoint = self.latest_checkpoint
        reopens_opening = self._reopens_opening(first)
        logger.warning(
            "interaction turn failed (%s: %s); running it again on a fresh box from %s",
            type(failure).__name__,
            failure,
            f"checkpoint {checkpoint.id} at turn {checkpoint.turns}"
            if checkpoint is not None and not reopens_opening
            else "its opening state",
        )
        if reopens_opening:
            task = self._opening.task
            start = self._opening.checkpoint
            # The turn's own input: a same-box recovery's note means nothing here.
            next_messages = turn_input
            in_prompt = False
        else:
            assert checkpoint is not None
            trace = self.trace
            conversation = trace.conversation_for(checkpoint)
            # The cut holds this turn's input once a call of this turn committed and
            # the cut reaches it.
            held = mark.in_prompt or (
                mark.trace is trace
                and checkpoint.turns >= mark.turns
                and trace.num_turns > mark.turns
            )
            task = self._opening.task.with_data(
                prompt=[
                    *conversation,
                    UserMessage(content=RETRY_NOTE),
                    *([] if held else (turn_input or [])),
                ]
            )
            start = checkpoint.id
            next_messages = None
            in_prompt = True
        steers = list(self.trace.steers)
        failed_run = self._run
        failed = await failed_run.close()
        self.failed_attempts.append(failed)
        self._run = await self._reopen(task, start, failed, failed_run)
        # A steer the replacement's conversation does not hold (never delivered, or
        # delivered after the checkpoint) goes to its inbox so the retry still hears it.
        prompt = task.data.prompt if isinstance(task.data.prompt, list) else []
        held_texts = {m.content for m in prompt if m.role == "user"}
        for record in steers if self._run.harness.SUPPORTS_INBOX else []:
            if record.text not in held_texts:
                self._run.session.queue_steer(
                    record.text, message_id=record.id, sender=record.sender
                )
        return next_messages, _TurnMark(self.trace, 0, in_prompt)

    async def close(self) -> Trace:
        """End the exchange and finish the rollout (idempotent): scoring and hooks
        run, then the finished trace returns (also on `interaction.trace`). After an
        interaction retry, it also carries the replaced attempts' errors first."""
        await self._cancel_deliveries()
        async with self._lock, self._gate or nullcontext():
            if not self._run.closed and self._run.ok:
                self.trace.stop("user_closed")
            trace = await self._run.close()
            history = [e for failed in self.failed_attempts for e in failed.errors]
            if history and not self._history_merged:
                # As `Agent.run` retries do: success is the `ok` stamp, never
                # errors-emptiness.
                self._history_merged = True
                trace.errors = [*history, *trace.errors]
            return trace


@dataclass(frozen=True)
class _Recorded:
    """One recorded turn of an exchange: the caller's input (`conversation[start:body]`;
    None for a prompted task's opening) and the segment it got back
    (`conversation[body:end]`)."""

    input: Messages | None
    start: int
    body: int
    end: int
    segment: Segment


def _recorded_turns(
    conversation: Messages, trace: Trace, task: Task, skip: int = 0
) -> list[_Recorded]:
    """Split a recorded exchange into the caller's turns. A user message is the
    caller's unless it is a note the framework added (a retry or resume note) or a
    steer; a prompted task's opening is its prompt. The first `skip` messages are a
    continuation's origin conversation, which the first turn continues: no turn's."""
    inner = {RETRY_NOTE, MODEL_RETRY_NOTE, *(steer.text for steer in trace.steers)}

    def callers(message) -> bool:
        return message.role == "user" and message.content not in inner

    prompt = task.data.prompt
    if prompt is None:
        opening = 0
    elif isinstance(prompt, str):
        opening = 1
    else:
        opening = sum(1 for m in prompt if m.role != "system")
    turns: list[_Recorded] = []
    index = skip
    while index < len(conversation):
        start = index
        if not turns and prompt is not None:
            index = min(opening, len(conversation))
        else:
            while index < len(conversation) and callers(conversation[index]):
                index += 1
            if index == start:  # a body without a caller turn: part of the last
                index += 1
                continue
        body = index
        while index < len(conversation) and not callers(conversation[index]):
            index += 1
        replies = [
            m for m in conversation[body:index] if m.role in ("assistant", "tool")
        ]
        turns.append(
            _Recorded(
                input=None
                if not turns and prompt is not None
                else list(conversation[start:body]),
                start=start,
                body=body,
                end=index,
                segment=Segment(messages=replies),
            )
        )
    return turns


def _cut_at(
    trace: Trace,
    checkpoint: Checkpoint,
    conversation: Messages,
    earlier: Iterable[tuple[int, Checkpoint]] = (),
) -> Trace:
    """`trace` as a continuation of `conversation` on `checkpoint`'s box: a copy whose
    latest checkpoint is it, at a conversation held in its task's prompt (a trace with
    no nodes yet holds its conversation there, `Trace.conversation_at`), after the
    `earlier` checkpoints inside that conversation at their positions, which the
    continuation carries (`Trace.carried`)."""
    data = trace.task.data.model_copy(update={"prompt": list(conversation)})
    carried = [
        c.model_copy(update={"turns": 0, "position": length})
        for length, c in earlier
        if length <= len(conversation)
    ]
    return trace.model_copy(
        update={
            "nodes": [],
            "checkpoints": [
                *carried,
                checkpoint.model_copy(update={"turns": 0, "position": None}),
            ],
            "task": trace.task.model_copy(update={"data": data}),
        }
    )


def _same_input(recorded: Messages | None, sent: Messages | None) -> bool:
    if recorded is None or sent is None:
        return recorded is None and sent is None
    return [(m.role, m.content) for m in recorded] == [
        (m.role, m.content) for m in sent
    ]


_OWNER_METHODS = frozenset({"start", "stop", "teardown"})


class ReplayBox:
    """`DurableInteraction.runtime` while the exchange replays its recorded turns and no
    box is restored for the turns given so far: a stand-in for that box, restored on
    first use. An awaited runtime method (`await exchange.runtime.read(path)`) restores
    the box from the checkpoint recorded at the end of those turns
    (`DurableInteraction.box`) and runs on it; `agent.run(runtime=...)` and
    `agent.interaction(runtime=...)` take it as that box. Its other attributes are the
    restored box's, so before the first use they raise AttributeError. Starting and
    stopping it is the exchange's."""

    def __init__(self, exchange: "DurableInteraction") -> None:
        self._exchange = exchange

    def __getattr__(self, name: str) -> Any:
        exchange = self._exchange
        if name.startswith("_") or name in _OWNER_METHODS:
            raise AttributeError(name)
        if inspect.iscoroutinefunction(getattr(Runtime, name, None)):

            async def call(*args: Any, **kwargs: Any) -> Any:
                runtime = await exchange.box()
                return await getattr(runtime, name)(*args, **kwargs)

            return call
        if exchange._box is None:
            raise AttributeError(
                f"{name!r}: the replaying exchange's box is restored on first use; "
                "await one of its methods, or `await interaction.box()`, first"
            )
        return getattr(exchange._box, name)


async def _real_runtime(runtime: Runtime | ReplayBox | None) -> Runtime | None:
    """The box a `ReplayBox` stands for, restored; any other runtime as it is."""
    if isinstance(runtime, ReplayBox):
        return await runtime._exchange.box()
    return runtime


class DurableInteraction:
    """A durable `agent.interaction(task)` (a `Durable` installed, `verifiers.v1.durable`):
    an exchange kept by its traces, so code that drives it the same way gets the same
    replies after a crash.

    A finished exchange of the key replays: each `turn(message)` whose message matches
    the recorded one returns the recorded segment without running anything. A cut-off
    exchange replays the turns its latest checkpoint covers, then continues live from
    that checkpoint (the turn in progress at the cut answers on `RETRY_NOTE`, so its
    segment holds only what came after the cut). A caller that sends a different
    message, or more turns than a finished exchange had, continues live from the
    checkpoint taken at the end of the turns it was given (a per-call save of a final
    reply, or `checkpoint()`); with none, the turn raises, unless no turn was replayed
    yet: then the exchange runs fresh. Steering, `checkpoint()` and `latest_checkpoint`
    act on the live exchange only. Without anything recorded, the exchange is live from
    the start, as a plain `interaction()`.

    An exchange that never goes live ran nothing: its trace is `kept`. A cut-off
    exchange whose every recorded turn was given back is kept too (it was cut after
    its last turn, before it closed): when it closes, its trace is a copy of the
    recorded one, closed as the live exchange would close it (`user_closed`, ok), its
    finalize and scoring not run. A failed exchange (closed with an error after its
    turns: a failed finalize, a driver that raised) was not cut off: its last turn runs
    again live, from its latest checkpoint in it, and the live exchange closes as usual.

    The box. An exchange on a borrowed box (`runtime=`) has that box throughout. One
    that owns its box has none while it replays: `runtime` is then a `ReplayBox`, and
    the first use of it (or `await box()`) restores the box from the checkpoint
    recorded at the end of the turns given so far (with per-call saves, every final
    reply has one) and keeps it. A later replayed turn moves the kept box to that
    turn's checkpoint, in place (`RuntimeBackend.adopt`: a reference to it stays valid). When
    the replay hands over to the live exchange, the live exchange continues on the
    kept box (borrowed from the exchange, with what the driver wrote to it since), moved
    first to the checkpoint it continues from when that is another one. A replay that
    never uses the box restores nothing; the live exchange then restores its own box
    from the hand-over checkpoint, as before. A record that continues an earlier one
    carries its checkpoints (`Trace.carried`), so the box restores wherever the earlier
    record's could; before the first turn with no checkpoint there, the box is the one
    the call opened on (`opening`: a continuation's origin checkpoint)."""

    def __init__(
        self,
        task: Task,
        open_live: Callable[..., object],
        stack: AsyncExitStack,
        recorded: Trace | None = None,
        checkpoints: list[Checkpoint] | None = None,
        *,
        skip: int = 0,
        runtime: Runtime | None = None,
        provision: Callable[[Checkpoint], AbstractAsyncContextManager[Runtime]]
        | None = None,
        opening: Checkpoint | None = None,
    ) -> None:
        self._task = task
        self._open_live = open_live
        self._stack = stack
        self._recorded = recorded
        self._borrowed = runtime
        """The box the caller placed the exchange in (`runtime=`), if any."""
        self._provision = provision
        """Provisions a box restored from a checkpoint, for reads during the replay."""
        self._box: Runtime | None = None
        """The box restored for the replay, once something used it."""
        self._box_at: Checkpoint | None = None
        """The checkpoint `_box` was restored from."""
        self._box_lock = asyncio.Lock()
        self._live: Interaction | None = None
        self._over = False
        self._conversation: Messages = []
        self._turns: list[_Recorded] = []
        self._cuts: list[tuple[int, Checkpoint]] = []
        self._next = 0
        self._position = skip
        """Where the turns given so far end in the recorded conversation; a
        continuation's starts after its origin's conversation (`skip`)."""
        self._skip = skip
        self._opening = opening
        """The checkpoint the exchange opened on (a continuation's origin's, `at`):
        the box before any turn when no recorded checkpoint is there."""
        self._done = False
        self._ended = False
        """A cut-off recorded exchange was given back whole and closed: `_recorded` is
        now its closed copy."""
        self._horizon = 0
        self._complete = 0
        if recorded is None:
            return
        self._conversation = recorded.conversation_at(None)
        self._turns = _recorded_turns(self._conversation, recorded, task, skip)
        for checkpoint in checkpoints or []:
            cut = recorded.conversation_for(checkpoint)
            if checkpoint.position is not None and len(cut) < checkpoint.position:
                continue  # the conversation it was carried into is not recorded yet
            if _same_input(cut, self._conversation[: len(cut)]):
                self._cuts.append((len(cut), checkpoint))
        self._done = recorded.is_completed and recorded.ok
        self._horizon = (
            len(self._conversation)
            if self._done
            else max((length for length, _ in self._cuts), default=0)
        )
        self._complete = len(self._turns)
        """The recorded turns that finished: a cut-off exchange's last turn did not
        unless it ends on a reply with no tool calls, and a failed exchange's last turn
        never did (it runs again live)."""
        if not self._done and self._turns:
            final = self._conversation[self._turns[-1].end - 1]
            if (
                recorded.is_completed
                or final.role != "assistant"
                or getattr(final, "tool_calls", None)
            ):
                self._complete -= 1

    @property
    def kept(self) -> bool:
        """Every turn so far came from the recorded exchange, and the exchange either
        finished or was cut off (never closed) and given back every recorded turn:
        nothing ran. A failed exchange is never kept whole: its last turn is not
        complete (`_complete`), so it goes live."""
        if self._live is not None or self._recorded is None:
            return False
        return self._done or (bool(self._turns) and self._next == len(self._turns))

    @property
    def ended(self) -> bool:
        """A cut-off recorded exchange was given back whole and closed: `trace` is a
        closed copy of the record, which the store does not hold yet."""
        return self._ended

    @property
    def live(self) -> Interaction | None:
        """The live exchange, once the replay has handed over to it."""
        return self._live

    @property
    def trace(self) -> Trace:
        if self._live is not None:
            return self._live.trace
        assert self._recorded is not None
        return self._recorded

    @property
    def runtime(self) -> Runtime | ReplayBox | None:
        """The box: the live exchange's; while replaying, the borrowed box
        (`runtime=`), the box restored for the turns given so far, or a `ReplayBox`
        that restores it on first use (see the class docs)."""
        if self._live is not None:
            return self._live.runtime
        if self._borrowed is not None:
            return self._borrowed
        point = self._replay_point()
        at_box = self._box_at
        if (
            self._box is not None
            and at_box is not None
            and point is not None
            and at_box.id == point.id
        ):
            return self._box
        return ReplayBox(self)

    async def box(self) -> Runtime:
        """The box as the turns given so far left it: the live exchange's, the borrowed
        one, or, while replaying, the one restored for those turns, restored now when
        it is not yet (see the class docs). Raises a `checkpoint_missing`
        `SandboxError` when no checkpoint recorded there matches the box."""
        if self._live is not None:
            runtime = self._live.runtime
            assert runtime is not None
            return runtime
        if self._borrowed is not None:
            return self._borrowed
        point = self._replay_point()
        if point is None:
            raise SandboxError(
                f"no checkpoint of the recorded exchange holds its box after turn "
                f"{self._next}: restoring the box during a replay needs a checkpoint at "
                "the end of each turn (save='per_call', or interaction.checkpoint())",
                kind="checkpoint_missing",
            )
        return await self._restore(point)

    async def _restore(self, point: Checkpoint) -> Runtime:
        """The replay's box, at `point`: provisioned from it the first time, moved to
        it in place after (its owner's `Box.replace`)."""
        async with self._box_lock:
            if self._box is None:
                if self._provision is None:
                    raise RuntimeError("this exchange cannot restore its box")
                self._box = await self._stack.enter_async_context(
                    self._provision(point)
                )
            elif self._box_at is None or self._box_at.id != point.id:
                owner = box_of(self._box)
                assert owner is not None
                await owner.replace(owner.generation, point)
            self._box_at = point
            return self._box

    def _settled_cuts(self) -> list[tuple[int, Checkpoint]]:
        """The cuts whose box is as the turns given so far left it: each ends where they
        end, or holds no more than the next recorded input (which never touched the
        box)."""
        recorded = self._turns[self._next] if self._next < len(self._turns) else None
        last = recorded.body if recorded is not None else self._position
        return [(n, c) for n, c in self._cuts if self._position <= n <= last]

    def _replay_point(self) -> Checkpoint | None:
        """The first of the `_settled_cuts`' checkpoints; before any turn with none,
        the one the exchange opened on."""
        found = self._settled_cuts()
        if found:
            return min(found, key=lambda item: item[0])[1]
        if self._next == 0 and self._position == self._skip:
            return self._opening
        return None

    @property
    def task(self) -> Task:
        return self._live.task if self._live is not None else self._task

    @property
    def failed_attempts(self) -> list[Trace]:
        return self._live.failed_attempts if self._live is not None else []

    @property
    def latest_checkpoint(self) -> Checkpoint | None:
        return self._live.latest_checkpoint if self._live is not None else None

    def _require_live(self, what: str) -> Interaction:
        if self._live is None:
            raise RuntimeError(
                f"{what} needs the live exchange; this one is replaying its recorded "
                "turns (take the next turn first)"
            )
        return self._live

    async def checkpoint(self) -> Checkpoint:
        return await self._require_live("checkpoint()").checkpoint()

    async def send(self, message: str, **kwargs) -> dict:
        return await self._require_live("send()").send(message, **kwargs)

    async def steer(self, message: str, **kwargs) -> dict:
        return await self._require_live("steer()").steer(message, **kwargs)

    async def go_live(
        self,
        resume: Trace | None = None,
        at: str | None = None,
        note: str | None = None,
        *,
        start: Checkpoint | None = None,
    ) -> Interaction:
        """Open the live exchange: fresh (the call's own opening), or continuing
        `resume` at `at`, whose box is `start`. A box restored for the replay is moved
        to `start` and the live exchange continues on it."""
        if self._live is None:
            if self._box is not None and start is not None:
                await self._restore(start)
            self._live = await self._stack.enter_async_context(
                self._open_live(resume, at, note, self._box)
            )
        return self._live

    async def turn(self, message: str | Messages | None = None) -> Segment:
        """The next turn: replayed while it matches the recorded exchange, live
        after (see the class docs)."""
        if self._live is not None:
            return await self._live.turn(message)
        if self._over:
            raise RuntimeError(
                "the exchange is over (the run ended); read interaction.trace"
            )
        sent: Messages | None
        if isinstance(message, str):
            sent = [UserMessage(content=message)]
        elif message is not None:
            sent = [parse_message(m) if isinstance(m, dict) else m for m in message]
        else:
            sent = None
        recorded = self._turns[self._next] if self._next < len(self._turns) else None
        matches = recorded is not None and _same_input(recorded.input, sent)
        if (
            matches
            and recorded is not None
            and self._next < self._complete
            and recorded.end <= self._horizon
        ):
            self._next += 1
            self._position = recorded.end
            return recorded.segment
        trace = self._recorded
        assert trace is not None
        if (
            recorded is None
            and self._done
            and trace.stop_condition not in (None, "user_closed")
        ):
            # The recorded exchange ended itself after the turns it gave.
            self._over = True
            return Segment(messages=[], terminated=True)
        if matches and recorded is not None:
            inside = [
                (length, c)
                for length, c in self._cuts
                if recorded.body <= length
                and (length < recorded.end or self._next >= self._complete)
            ]
            if inside:
                # The cut holds this turn's input: continue it where it was cut (on a
                # note when the turn had begun answering).
                length, checkpoint = max(inside, key=lambda item: item[0])
                note = RETRY_NOTE if length > recorded.body else None
                live = await self.go_live(trace, checkpoint.id, note, start=checkpoint)
                return await live.turn()
        # A checkpoint whose box is as the turns already given left it (the next
        # recorded input, if its cut holds it, is dropped), after an agent reply.
        position = self._position
        boundary = (
            [c for _, c in self._settled_cuts()]
            if position > 0 and self._conversation[position - 1].role == "assistant"
            else []
        )
        if boundary:
            # A box restored for the replay at one of them holds what the driver wrote
            # to it since: the live exchange continues on it as it is.
            at_box = self._box_at
            point = (
                at_box
                if self._box is not None
                and at_box is not None
                and any(c.id == at_box.id for c in boundary)
                else boundary[-1]
            )
            earlier = [(n, c) for n, c in self._cuts if c is not point]
            live = await self.go_live(
                _cut_at(trace, point, self._conversation[:position], earlier),
                None,
                None,
                start=point,
            )
            return await live.turn(message)
        if self._next == 0:
            # Nothing was replayed: the exchange runs fresh, from its own opening.
            live = await self.go_live()
            return await live.turn(message)
        raise RuntimeError(
            f"cannot continue the recorded exchange at turn {self._next + 1}: the "
            "caller's turn differs from the recorded one (or goes past it) and no "
            "checkpoint matches the turns already given; the driver must be "
            "deterministic, or checkpoint at the end of each turn"
        )

    async def close(self) -> Trace:
        if self._live is not None:
            return await self._live.close()
        if self.kept and not self._done:
            # Cut off after its last turn: close the record as the live exchange
            # would have, without running anything.
            assert self._recorded is not None
            closed = self._recorded.model_copy(deep=True)
            closed.stop("user_closed")
            closed.ok = True
            closed._kept = True
            self._recorded = closed
            self._done = self._ended = self._over = True
        return self.trace


def _chain_save(on_save: OnSave | None, store: Durable | None) -> OnSave | None:
    """`on_save`, then the trace written to `store` when it is also the trace store."""
    if store is None or not isinstance(store.lookup, TraceSink):
        return on_save

    def save(trace: Trace, checkpoint: Checkpoint) -> None:
        if on_save is not None:
            on_save(trace, checkpoint)
        store.write(trace)

    return save


def _call_store(found: Durable | None, runtime: Runtime | None) -> Durable | None:
    """Where a durable call writes its trace: `found`, unless the durable box it runs on
    writes its runs to the same store already (`Agent._box_watch`)."""
    box = box_of(runtime)
    if (
        found is not None
        and box is not None
        and box.key is not None
        and box.durable is not None
        and box.durable.lookup is found.lookup
    ):
        return None
    return found


def _call_labels(found: Durable | None, resolved: Resolved) -> dict:
    """The labels a durable call stamps: its scope's and its own (`CALL_LABEL`)."""
    labels = found.scope if found is not None else {}
    return merge_labels(labels, {CALL_LABEL: resolved.label})


def _plan_durable(
    name: str,
    task: Task | None,
    checkpoint: str | None,
    resume: Trace | None,
    at: str | int | None,
    note: str | None,
    key: str | None,
    durable: Durable | bool | None,
) -> tuple[Durable | None, str | None]:
    """`plan_call` for an agent call; `(None, None)` for a plain one. A durable call
    picks its own start: `checkpoint`, and `at`/`note` without `resume`, make a call
    plain without `key` and are refused with it; so does a call with no task."""
    given = {"checkpoint": checkpoint}
    if resume is None:
        given.update(at=at, note=note)
    managed = [arg for arg, value in given.items() if value is not None]
    if key is None and (managed or (task is None and resume is None)):
        durable = False
    # The position is taken here, before anything is awaited.
    found, key = plan_call(name, key, durable)
    if key is not None and managed:
        raise TypeError(f"a durable call (key=) resumes itself; drop {managed}")
    return found, key


def _box_start(runtime: Runtime | None, resumes: Trace | None) -> Checkpoint | None:
    """Where a continuation on a borrowed box starts: the box's latest checkpoint
    (`Box.latest`), not the one it was first restored from, which other runs on it may
    have moved past."""
    box = box_of(runtime)
    return box.latest if box is not None and resumes is not None else None


class Agent:
    """A configured harness + model + runtime policy, runnable on any task.

    Built from an `AgentConfig` alone; `interception=` injects a live resource to
    borrow — its owner keeps the lifecycle. The endpoint stays config: each rollout
    builds and closes its own `Client`, so an agent holds no transport. The config's
    `runtime` is a *policy*: each `run` provisions a fresh box from it, resolved
    per task; `run(runtime=...)` places the run into an existing box instead
    (borrowed boxes are never started or torn down by the run)."""

    def __init__(
        self,
        config: AgentConfig,
        *,
        interception: Interception | None = None,
    ) -> None:
        from verifiers.v1.utils.loaders import harness_config_type, load_harness

        if config.model is None:
            raise ValueError(
                "AgentConfig.model is unset; an Agent needs a pinned model "
                "(inside an env the run's own model fills it in)"
            )
        # Resolve the unpinned identity fields into the config: it is the agent's
        # full identity — what stamps onto every trace (`AgentInfo.config`).
        if config.harness is None:
            config = config.model_copy(
                update={"harness": harness_config_type("bash")(id="bash")}
            )
        config = resolve_agent(config)
        self.config = config
        self.harness = load_harness(config.harness)
        self.ctx = ModelContext(
            model=config.model,
            client=config.client or EvalClientConfig(),
            sampling=config.sampling,
        )
        self._closed = False
        self.runtime_config: RuntimeConfig = config.runtime
        self.interception = interception
        self.limits = RolloutLimits(
            max_turns=config.max_turns,
            max_input_tokens=config.max_input_tokens,
            max_output_tokens=config.max_output_tokens,
            max_total_tokens=config.max_total_tokens,
        )
        self.timeout = config.timeout
        # Env episode agents replace this with the episode's agent semaphore
        # (`--env.max-concurrent-agents`). Interactions acquire it only around active
        # lifecycle work, never while awaiting the caller between segments.
        self._gate: asyncio.Semaphore | None = None
        # Env-owned standing, not config: `Env.setup` marks fixed agents
        # untrainable and traces are stamped from here; inert outside an env.
        self.trainable: bool = True
        self._entered = False
        self._server: InterceptionServer | None = None
        self._warned_resources: set[tuple[str, str]] = set()
        # Env episode agents set this: the episode's seats to their live trace
        # ids, which `message_agents` reaches.
        self._peers: dict[str, str] | None = None

    @property
    def name(self) -> str:
        """The agent's name: its seat in an episode, else "agent". A durable call
        without a key is keyed by it and its position (`<name>/<n>`)."""
        return getattr(self, "_name", "agent")

    async def __aenter__(self) -> Self:
        if self._entered:
            raise RuntimeError("Agent is already entered; enter it once and share it")
        if self._closed:
            raise RuntimeError("Agent is closed; create a new agent")
        self._entered = True
        if self.interception is None:
            # Sized to the runtime policy (remote needs the tunnel); runs the
            # server can't serve fall back per run.
            self._server = InterceptionServer(
                requires_tunnel=not runtime_is_local(self.runtime_config)
            )
            try:
                await self._server.__aenter__()
            except BaseException:
                # A failed __aenter__ gets no __aexit__ from `async with`: unwind
                # here, or the agent stays "already entered" forever.
                self._entered, self._server = False, None
                raise
        return self

    async def __aexit__(self, *exc) -> None:
        self._entered = False
        server, self._server = self._server, None
        if server is not None:
            await server.__aexit__(*exc)

    def _interception_for(
        self,
        run_is_local: bool,
        task: Task,
        shared_tools: Mapping,
        messaging: bool = False,
    ) -> Interception | None:
        """Which interception this run rides: an injected one always (its owner
        sized its reach); the owned server only when provably reachable from all
        the run's consumers — when it tunnels, else for a local run with no tool
        servers in play (such servers may sit in a remote runtime and must reach
        `/state`). Otherwise `None`: a per-run server sized to the task."""
        if self.interception is not None:
            return self.interception
        if self._server is None:
            return None
        if any(tool.state_secret for tool in shared_tools.values()):
            # Shared state credentials are attached per run, after this owned
            # server was created; let the rollout size a scoped server instead.
            return None
        if self._server.tunnel is not None or (
            run_is_local
            and not shared_tools
            and not messaging
            and not task.toolsets(task.config)
        ):
            return self._server
        return None

    def _check_resume_support(self) -> None:
        # Multi-turn capability is derived: a harness needs transcript replay, a
        # native resume implementation, or a rollout-scoped session implementation.
        harness = self.harness
        if (
            type(harness).session is Harness.session
            and type(harness).resume is Harness.resume
            and not harness.SUPPORTS_RESUME
        ):
            raise ValueError(
                f"Harness {harness.config.id!r} cannot host a user: resuming an "
                "exchange takes transcript-backed resume (SUPPORTS_RESUME) for the "
                "default relaunch-on-the-conversation, a native resume() override, "
                "or a rollout-scoped session() override. Use a harness that has "
                "one (e.g. bash or null)."
            )

    def _opening(
        self,
        task: Task | None,
        resume: Trace | None,
        at: str | int | None,
        note: str | None,
        checkpoint: str | None,
        runtime: Runtime | None,
        *,
        interactive: bool,
    ) -> _Opening:
        """Resolve where a run starts. Resuming a trace opens on its conversation cut
        at the checkpoint `at` (default: its latest; `Trace.conversation_at`) as the
        prompt — how `Harness.resume` continues an exchange — continued by `note` as a
        user message, on a box restored from that checkpoint; a conversation that ends
        on the agent's reply with no `note` waits for an interaction's first
        `turn(message)`. Everything that could make the continuation impossible is
        checked here, before anything is provisioned."""
        if resume is None:
            if note is not None or at is not None:
                raise ValueError(
                    "note= and at= continue a trace; pass resume= with them"
                )
            if task is None:
                raise ValueError("pass a task to run (or a trace to resume=)")
            return _Opening(task=task, checkpoint=checkpoint)
        if checkpoint is not None:
            raise ValueError(
                "a resumed trace starts from its own checkpoint; pass resume= or "
                "checkpoint=, not both"
            )
        harness = self.harness
        if not harness.SUPPORTS_RESUME:
            raise ValueError(
                f"Harness {harness.config.id!r} cannot resume a trace: that "
                "relaunches it on the trace's conversation as a Messages prompt "
                "(SUPPORTS_RESUME). Use a harness that has it (e.g. bash or null)."
            )
        start: Checkpoint | None = None
        if resume.checkpoints:
            start = resume.checkpoint_at(at)
        elif at is not None:
            raise ValueError(f"trace {resume.id} has no checkpoints; drop at=")
        elif task is None and runtime is None:
            raise ValueError(
                f"trace {resume.id} has no checkpoint, so its box cannot be restored; "
                "pass task= to resume its conversation on a fresh box (or runtime= "
                "to resume it in a live one)"
            )
        if start is not None and runtime is None:
            try:
                restore_config(self.runtime_config, start.id)
            except TypeError as e:
                raise ValueError(
                    f"this trace's box is checkpoint {start.id!r}, but the agent's "
                    f"{self.runtime_config.type} runtime cannot restore checkpoints: "
                    f"{e}"
                ) from e
        task = task if task is not None else resume.task.rebuild()
        messages = resume.conversation_for(start)
        prefix: Messages | None = None
        prompt: Messages | None
        if note is not None:
            prompt = [*messages, UserMessage(content=note)]
        elif messages and messages[-1].role != "assistant":
            prompt = messages
        elif interactive:
            prompt, prefix = None, messages or None
        else:
            raise ValueError(
                "this trace's conversation is empty or ends on the agent's reply; "
                "pass note= with the next user message"
            )
        task = task.with_data(prompt=prompt)
        try:
            harness.resolve_prompt(
                task.data.model_copy(
                    update={"prompt": prompt or [UserMessage(content="")]}
                )
            )
        except (TypeError, ValueError) as e:
            raise ValueError(
                f"Harness {harness.config.id!r} cannot continue this trace's "
                f"conversation: {e}"
            ) from e
        return _Opening(
            task=task,
            # A borrowed box is as its owner restored it: only the cut is the trace's.
            checkpoint=start.id if start is not None and runtime is None else None,
            prefix=prefix,
            source=resume,
        )

    def _restore_refusal(self) -> str | None:
        """Why the agent's runtime cannot start a box from a checkpoint; None when it
        can."""
        try:
            restore_config(self.runtime_config, "probe")
        except TypeError as e:
            return str(e)
        return None

    def _check_saves(self, runtime: Runtime | None) -> None:
        """`save = "per_call"` on a box the run owns needs a runtime that restores
        checkpoints (a borrowed box is checkpointed through its owner, when it has
        one: `Rollout` decides)."""
        if self.config.save != "per_call" or runtime is not None:
            return
        if (refusal := self._restore_refusal()) is not None:
            raise ValueError(
                f"save='per_call' checkpoints the run's box, but the agent's "
                f"{self.runtime_config.type} runtime has no checkpoints to continue "
                f"from: {refusal}"
            )

    async def run(
        self,
        task: Task | None = None,
        *,
        runtime: Runtime | None = None,
        tools: Mapping[str, SharedToolServer] | None = None,
        on_trace: Callable[[Trace], None] | None = None,
        collect_artifacts: bool = False,
        checkpoint: str | None = None,
        resume: Trace | None = None,
        at: str | int | None = None,
        note: str | None = None,
        on_save: OnSave | None = None,
        key: str | None = None,
        depends: object = None,
        durable: Durable | bool | None = None,
        on_score: Callable[[Trace, Runtime], Awaitable[None]] | None = None,
    ) -> Trace:
        """Run this agent on `task` once and return the trace: one segment — the
        program runs on the task's prompt until it exits (a multi-turn exchange
        is `interaction()`). `runtime` places it into a live borrowed box instead of
        provisioning one; `checkpoint` provisions it from a `Runtime.checkpoint()`
        id instead of the image; `tools` are live servers borrowed from their
        owner, counted in the pairing check; `on_trace` observes the trace the
        moment it's minted, before any I/O. `collect_artifacts` captures the task's
        declared artifacts after its finalizer while its container runtime is still
        alive. Retries whole while the trace ends with a retryable error
        (`config.retries`) — never into a borrowed box; the final trace keeps earlier
        attempts' errors.

        `resume` continues a trace (its record, e.g. loaded from JSON) from its
        checkpoint `at` (an id or index into `resume.checkpoints`; default its latest):
        the task is rebuilt from `resume.task` unless given, the box starts from the
        checkpoint, and the program opens on the trace's conversation cut at it
        (`Trace.conversation_at`) followed by `note` (required when that conversation
        ends on the agent's reply). A trace with no checkpoint resumes only with
        `task=` (on a fresh box) or `runtime=`. With `config.save = "per_call"`, the
        owned box is checkpointed after every model response, each checkpoint recorded
        on the trace and handed to `on_save(trace, checkpoint)`; a box `provision`
        handed out is checkpointed through its owner the same way (`box.Box`).

        With a `Durable` installed (`use_durable`, `Env.durable`), the call is durable
        (`verifiers.v1.durable`), keyed by its position (the agent's n-th call) or by
        `key`: a finished ok trace of the key returns without running (`trace.kept`),
        a cut-off one with checkpoints continues from its latest, and a changed
        fingerprint (task, task code, model, sampling, harness, client, `depends`)
        runs fresh. `durable=False`
        opts out; a `Durable` replaces the installed one for this call (another
        scope). A durable continuation (`resume=` an earlier trace, its origin, with
        `at` and `note`) is durable the same way, the origin's id, `at` and `note` in
        its fingerprint: a finished one is kept, a cut-off one continues from its own
        latest checkpoint, and another origin runs it fresh from the origin. A call
        passing `checkpoint` (or `at`/`note` without `resume`) without `key` is a
        plain one. `on_score(trace, runtime)` runs after task scoring in the
        selected verifier workspace, inside its timeout and retry boundary."""
        if self._closed:
            raise RuntimeError("Agent is closed; create a new agent")
        if durable is not False:
            found, key = _plan_durable(
                self.name, task, checkpoint, resume, at, note, key, durable
            )
            if key is not None:
                return await self._durable_run(
                    task,
                    key,
                    found,
                    runtime=runtime,
                    tools=tools,
                    on_trace=on_trace,
                    collect_artifacts=collect_artifacts,
                    on_save=on_save,
                    on_score=on_score,
                    depends=depends,
                    resume=resume,
                    at=at,
                    note=note,
                )
            return await self.run(
                task,
                runtime=runtime,
                tools=tools,
                on_trace=on_trace,
                collect_artifacts=collect_artifacts,
                checkpoint=checkpoint,
                resume=resume,
                at=at,
                note=note,
                on_save=on_save,
                on_score=on_score,
                durable=False,
            )
        runtime = await _real_runtime(runtime)
        opening = self._opening(
            task, resume, at, note, checkpoint, runtime, interactive=False
        )
        self._check_saves(runtime)
        on_trace, on_save, finish = self._box_watch(runtime, on_trace, on_save)
        retry = self.config.retries
        retry_state = RetryState(retry)
        history: list = []
        for attempt in range(retry_state.max_retries + 1):
            trace = await self._run_once(
                opening.task,
                runtime,
                tools,
                on_trace,
                collect_artifacts,
                opening.checkpoint,
                on_save,
                resumes=opening.source,
                on_score=on_score,
            )
            if attempt == retry_state.max_retries or trace.ok:
                break
            if runtime is not None:
                logger.warning(
                    "not retrying the rollout on a borrowed box (its state is no "
                    "longer the task's start state); the error stands"
                )
                break
            cause = retry_state.next_error(trace.errors)
            if cause is None:
                break
            history.extend(trace.errors)
            delay = backoff(attempt)
            logger.warning(
                "retrying agent rollout (retry %d/%d) in %.1fs after error: %s",
                attempt + 1,
                retry_state.max_retries,
                delay,
                cause.type,
            )
            await asyncio.sleep(delay)
        if history:
            # The full history rides the final trace either way; success is the
            # `ok` stamp, never errors-emptiness.
            trace.errors = history + trace.errors
        finish(trace)
        return trace

    def _box_watch(
        self,
        runtime: Runtime | None,
        on_trace: Callable[[Trace], None] | None,
        on_save: OnSave | None,
    ) -> tuple[Callable[[Trace], None], OnSave | None, Callable[[Trace], None]]:
        """The observers of a plain run: every trace is stamped with the labels of
        the durable box it runs on (`box.Box.labels`) and none it inherited from a
        continued trace; a run on a durable box is written to its store on every save
        and when it finishes. Returns `(on_trace, on_save, finish)`."""
        box = box_of(runtime)
        labels: dict = {}
        store: Durable | None = None
        if box is not None and box.key is not None:
            store = box.durable
            if store is not None:
                labels = merge_labels(store.scope, box.labels)
            else:
                labels = box.labels
        watch = observe(on_trace, labels, drop=(CALL_LABEL, BOX_LABEL))
        if store is None:
            return watch, on_save, lambda trace: None
        return watch, _chain_save(on_save, store), store.write

    def _continuation(
        self, previous: Trace | None, runtime: Runtime | None
    ) -> tuple[Trace | None, str | None]:
        """Where a durable call continues `previous` (its trace and checkpoint), or
        `(None, None)` to run fresh. An owned box restores from the trace's latest
        checkpoint (when the runtime restores checkpoints); a borrowed box must be one
        its owner restored, and the run continues from its latest checkpoint no newer
        than the box's."""
        if previous is None or not self.harness.SUPPORTS_RESUME:
            return None, None
        if runtime is None:
            if self._restore_refusal() is not None:
                return None, None
            return previous, None
        box = box_of(runtime)
        restored = box.restored if box is not None else None
        if restored is None:
            return None, None
        usable = [
            c
            for c in previous.checkpoints
            if c.id == restored.id
            or (c.time is not None and restored.time is not None)
            and c.time <= restored.time
        ]
        return (previous, usable[-1].id) if usable else (None, None)

    def _call_parts(
        self,
        task: Task | None,
        found: Durable | None,
        depends: object,
        resume: Trace | None,
        at: str | int | None,
        note: str | None,
    ) -> dict[str, str]:
        """A durable call's fingerprint parts (`durable.call_parts`); a continuation
        given no task fingerprints the task its origin recorded."""
        if task is None:
            if resume is None:
                raise ValueError("a durable call (key=) needs its task")
            task = resume.task.rebuild()
        return call_parts(task, self, found, depends, resume, at, note)

    async def _durable_run(
        self,
        task: Task | None,
        key: str,
        found: Durable | None,
        *,
        runtime: Runtime | None,
        tools: Mapping[str, SharedToolServer] | None,
        on_trace: Callable[[Trace], None] | None,
        collect_artifacts: bool,
        on_save: OnSave | None,
        depends: object,
        on_score: Callable[[Trace, Runtime], Awaitable[None]] | None = None,
        resume: Trace | None = None,
        at: str | int | None = None,
        note: str | None = None,
    ) -> Trace:
        """A durable `run`: resolve the call by its traces, then return the kept one,
        continue it, or run fresh (`verifiers.v1.durable`). A continuation of `resume`
        runs fresh as that continuation; a cut-off one continues its own trace."""
        runtime = await _real_runtime(runtime)
        resolved = resolve_call(
            found,
            key,
            self._call_parts(task, found, depends, resume, at, note),
        )
        if resolved.kept is not None:
            return resolved.kept
        again, again_at = self._continuation(resolved.resume, runtime)
        if resolved.resume is not None and again is None:
            logger.info("call %r runs fresh: its box cannot be restored", key)
        if again is not None:
            resume, at, note = again, again_at, RETRY_NOTE
        store = _call_store(found, runtime)
        trace = await self.run(
            task,
            runtime=runtime,
            tools=tools,
            on_trace=observe(on_trace, _call_labels(found, resolved)),
            collect_artifacts=collect_artifacts,
            resume=resume,
            at=at,
            note=note,
            on_save=_chain_save(on_save, store),
            on_score=on_score,
            durable=False,
        )
        if store is not None:
            store.write(trace)
        return trace

    async def _run_once(
        self,
        task: Task,
        runtime: Runtime | None,
        shared_tools: Mapping[str, SharedToolServer] | None,
        on_trace: Callable[[Trace], None] | None,
        collect_artifacts: bool,
        checkpoint: str | None,
        on_save: OnSave | None = None,
        *,
        resumes: Trace | None = None,
        on_score: Callable[[Trace, Runtime], Awaitable[None]] | None = None,
    ) -> Trace:
        params = self._rollout_params(
            task, runtime, dict(shared_tools or {}), checkpoint
        )
        if collect_artifacts and isinstance(params["runtime_config"], SubprocessConfig):
            raise TypeError(
                "artifact collection requires a container runtime; subprocess "
                "artifacts live in a host-only temporary working directory"
            )
        run = Rollout(
            task=task,
            on_trace=on_trace,
            collect_artifacts=collect_artifacts,
            on_save=on_save,
            resumes=resumes,
            restored=_box_start(runtime, resumes),
            on_score=on_score,
            **params,
        )
        try:
            if await run.open():
                await run.step()
                if run.ok:
                    run.trace.stop("agent_completed")
            trace = await run.close()
        except BaseException:
            # Finish cleanup even if another cancellation arrives during abort().
            await run_shielded(run.abort())
            raise
        if trace.agent.runtime is not None:
            trace.agent.runtime.borrowed = runtime is not None
        return trace

    @asynccontextmanager
    async def interaction(
        self,
        task: Task | None = None,
        *,
        runtime: Runtime | None = None,
        tools: Mapping[str, SharedToolServer] | None = None,
        on_trace: Callable[[Trace], None] | None = None,
        checkpoint: str | None = None,
        resume: Trace | None = None,
        at: str | int | None = None,
        note: str | None = None,
        on_save: OnSave | None = None,
        key: str | None = None,
        depends: object = None,
        durable: Durable | bool | None = None,
    ) -> AsyncIterator["Interaction | DurableInteraction"]:
        """Interact with this agent turn-by-turn: a full rollout of `task` where
        the CALLER is the run's user — the one exchange surface. Yields an
        `Interaction`; `await interaction.turn("...")` sends one user turn and
        runs one harness segment, returning its `Segment`. Who computes
        the turns is control flow, not framework machinery: an env's rollout loop,
        another agent's interaction, a game engine, a scripted closure, a human.

        The task's shape says who speaks first: a prompt-less task is opened by
        the first `turn(message)`; a prompted task speaks first — take its opening
        reply with a bare `turn()` before answering. A prompt that belongs to the
        USER side (a scenario the caller pursues, not the assistant's seed) is the
        caller's to hide: hand the interaction a task whose `data.prompt` is None
        and keep the scenario on a scoring-side field (the user-sim env's contract).

        `runtime`, `tools` and `checkpoint` work as they do for `run()`; an env
        supplies its taskset's shared tools automatically for tasks loaded from
        that taskset.

        Everything is a real rollout — the trace (live on `interaction.trace`),
        limits, `@stop`s, and scoring all apply; leaving the context ends the
        exchange (`user_closed`) and finishes the rollout, hooks and scoring
        included. A failure while opening the rollout raises before the context
        is entered (the failed trace is still completed and reported through
        `on_trace`). An exchange is caller-driven, so `config.retries` does not
        apply here; `config.interaction_retries` runs a turn that fails on
        infrastructure again: on the same box after a model-side failure, else on a
        fresh box restored from the trace's latest checkpoint (a borrowed box: its
        owner restores it, `box.Box.replace`; one with no owner is not retried).

        `resume`, `at` and `note` continue a trace (see `run`); without `note`, a
        conversation that ends on the agent's reply waits for the first
        `turn(message)`, which continues it. `interaction.checkpoint()` checkpoints
        the box between turns. With `config.save = "per_call"`, the owned box is
        checkpointed after every model response (`on_save(trace, checkpoint)`);
        `interaction.latest_checkpoint` holds the latest.

        With a `Durable` installed the exchange is durable as `run` is
        (`DurableInteraction`): a finished one replays its recorded turns to the same
        caller, and a cut-off one replays them up to its latest checkpoint and
        continues live from there; a continuation (`resume`) is durable too, as for
        `run`. While it replays, `interaction.runtime` restores the box on first use
        (`ReplayBox`, `DurableInteraction.box`). `key`, `depends` and `durable`
        are as for `run`."""
        if self._closed:
            raise RuntimeError("Agent is closed; create a new agent")
        if durable is not False:
            found, key = _plan_durable(
                self.name, task, checkpoint, resume, at, note, key, durable
            )
            if key is not None:
                async with self._durable_interaction(
                    task,
                    key,
                    found,
                    runtime=runtime,
                    tools=tools,
                    on_trace=on_trace,
                    on_save=on_save,
                    depends=depends,
                    resume=resume,
                    at=at,
                    note=note,
                ) as exchange:
                    yield exchange
                return
            async with self.interaction(
                task,
                runtime=runtime,
                tools=tools,
                on_trace=on_trace,
                checkpoint=checkpoint,
                resume=resume,
                at=at,
                note=note,
                on_save=on_save,
                durable=False,
            ) as plain:
                yield plain
            return
        runtime = await _real_runtime(runtime)
        self._check_resume_support()
        opening = self._opening(
            task, resume, at, note, checkpoint, runtime, interactive=True
        )
        self._check_saves(runtime)
        shared_tools = dict(tools or {})
        on_trace, on_save, finish = self._box_watch(runtime, on_trace, on_save)
        box = box_of(runtime)

        def make_run(
            task: Task, checkpoint: str | None, resumes: Trace | None
        ) -> Rollout:
            return Rollout(
                task=task,
                has_user=True,
                on_trace=on_trace,
                on_save=on_save,
                resumes=resumes,
                restored=_box_start(runtime, resumes),
                **self._rollout_params(task, runtime, shared_tools, checkpoint),
            )

        async def reopen(
            task: Task, checkpoint: str | None, failed: Trace, failed_run: Rollout
        ) -> Rollout:
            if box is not None:
                # The owner restores the borrowed box for every seat on it; the run
                # continues on it from its own conversation cut.
                checkpoint = None
                try:
                    await box.replace(failed_run.box_generation)
                except Exception as error:  # noqa: BLE001 - recorded on the replacement
                    # A box that would not start is the replacement's failure, as a
                    # failed open is: its step fails at once.
                    replacement = make_run(task, checkpoint, failed)
                    replacement.fail(error)
                    return replacement
            # A failed open is captured on the replacement: its step fails at once.
            replacement = make_run(task, checkpoint, failed)
            await replacement.open()
            return replacement

        run = make_run(opening.task, opening.checkpoint, opening.source)
        interaction = Interaction(
            run,
            gate=self._gate,
            opening=opening,
            reopen=reopen if runtime is None or box is not None else None,
            retries=self.config.interaction_retries,
            box=box,
        )
        async with self._gate or nullcontext():
            opened = await run.open()
            if not opened and (failure := run.failure) is not None:
                trace = await run.close()
                if trace.agent.runtime is not None:
                    trace.agent.runtime.borrowed = runtime is not None
                raise failure
        try:
            yield interaction
        except Exception as e:
            interaction._run.fail(e)
            raise
        except BaseException:
            await interaction._cancel_deliveries()
            await interaction._run.abort()
            raise
        finally:
            trace = (
                interaction.trace
                if interaction._run.closed
                else await interaction.close()
            )
            if trace.agent.runtime is not None:
                trace.agent.runtime.borrowed = runtime is not None
            finish(trace)

    @asynccontextmanager
    async def _durable_interaction(
        self,
        task: Task | None,
        key: str,
        found: Durable | None,
        *,
        runtime: Runtime | ReplayBox | None,
        tools: Mapping[str, SharedToolServer] | None,
        on_trace: Callable[[Trace], None] | None,
        on_save: OnSave | None,
        depends: object,
        resume: Trace | None = None,
        at: str | int | None = None,
        note: str | None = None,
    ) -> AsyncIterator[DurableInteraction]:
        """A durable `interaction` (`DurableInteraction`); with `resume`, a durable
        continuation of that trace."""
        runtime = await _real_runtime(runtime)
        resolved = resolve_call(
            found,
            key,
            self._call_parts(task, found, depends, resume, at, note),
        )
        # The exchange as it opens: a continuation's on its origin's conversation (a
        # prompt the first bare turn answers, or a prefix the first turn continues).
        opening = (
            self._opening(task, resume, at, note, None, runtime, interactive=True)
            if resume is not None
            else None
        )
        base = opening.task if opening is not None else task
        assert base is not None
        skip = len(opening.prefix or []) if opening is not None else 0
        recorded = resolved.kept
        checkpoints: list[Checkpoint] = []
        if recorded is None and resolved.resume is not None:
            again, again_at = self._continuation(resolved.resume, runtime)
            if again is not None:
                recorded = again
                checkpoints = list(again.checkpoints)
                if runtime is not None:
                    # On a borrowed box, only those no newer than its restore point.
                    ids = [c.id for c in checkpoints]
                    checkpoints = checkpoints[: len(ids) - ids[::-1].index(again_at)]
        elif recorded is not None and self.harness.SUPPORTS_RESUME:
            # A finished exchange goes live again only past its turns; on a borrowed
            # box nothing says the box matches, so it never does.
            if runtime is None and self._restore_refusal() is None:
                checkpoints = list(recorded.checkpoints)
        labels = _call_labels(found, resolved)
        # The live exchange runs on `runtime`, or on a box of its own (never durable).
        store = _call_store(found, runtime)
        save = _chain_save(on_save, store)

        def open_live(
            again: Trace | None,
            again_at: str | None,
            again_note: str | None,
            box: Runtime | None = None,
        ):
            if again is None:
                # Fresh: the call's own opening (a continuation of its origin).
                again, again_at, again_note = resume, at, note
            return self.interaction(
                task,
                runtime=box if box is not None else runtime,
                tools=tools,
                on_trace=observe(on_trace, labels),
                on_save=save,
                resume=again,
                at=again_at,
                note=again_note,
                durable=False,
            )

        def provision(checkpoint: Checkpoint) -> AbstractAsyncContextManager[Runtime]:
            """A box restored from `checkpoint`, for the replay to read."""
            return self.provision(base, checkpoint=checkpoint.id, durable=False)

        exchange: DurableInteraction | None = None
        try:
            async with AsyncExitStack() as stack:
                exchange = DurableInteraction(
                    base,
                    open_live,
                    stack,
                    recorded=recorded,
                    checkpoints=checkpoints,
                    skip=skip,
                    runtime=runtime,
                    provision=provision,
                    opening=(
                        resume.checkpoint_at(at)
                        if resume is not None and resume.checkpoints
                        else None
                    ),
                )
                if recorded is None:
                    await exchange.go_live()
                yield exchange
                if exchange.live is None:
                    await exchange.close()
        finally:
            # Also when the driver raised: its live trace is closed failed (or, when
            # cancelled, cut off), and a rerun must see that, not the last save.
            if found is not None and exchange is not None:
                if exchange.live is not None:
                    if store is not None:
                        store.write(exchange.live.trace)
                elif exchange.ended:
                    # A closed copy of the record: nothing else has written it.
                    found.write(exchange.trace)

    def _rollout_params(
        self,
        task: Task,
        runtime: Runtime | None,
        shared_tools: dict,
        checkpoint: str | None = None,
    ) -> dict:
        """Resolve one run's runtime config, pairing checks, timeouts,
        interception — shared by `run` and `interaction`."""
        harness = self.harness
        skills = [*task.data.skills, *harness.config.skills]
        if skills:
            # Skill installations can hold run-specific state, such as RLM's package environment.
            harness = type(harness)(
                harness.config.model_copy(update={"skills": skills})
            )
        if runtime is not None and checkpoint is not None:
            raise ValueError("a borrowed runtime cannot also start from a checkpoint")
        if runtime is not None:
            _check_borrowed_placement(task, runtime, self.runtime_config)
            runtime_config = runtime.config
            run_is_local = runtime.is_local
        else:
            runtime_config = resolve_runtime_config(
                self.runtime_config, task, self._warned_resources
            )
            if checkpoint is not None:
                runtime_config = restore_config(runtime_config, checkpoint)
            run_is_local = runtime_is_local(runtime_config)
        messaging = (
            [messaging_toolset()]
            if self.config.message_agents and self._peers is not None
            else []
        )
        validate_pairing(
            harness,
            type(task),
            runtime_config,
            tools=[*task.toolsets(task.config), *shared_tools.values(), *messaging],
        )
        timeouts = resolve_rollout_timeouts(self.timeout, task)
        return {
            "agent_config": self.config,
            "harness": harness,
            "ctx": self.ctx,
            "runtime_config": runtime_config,
            "timeouts": replace(
                timeouts,
                agent=cap_remote_agent_timeout(timeouts.agent, runtime_config, task),
            ),
            "limits": self.limits,
            "shared_tools": shared_tools,
            "interception": self._interception_for(
                run_is_local, task, shared_tools, bool(messaging)
            ),
            "runtime": runtime,
            "peers": self._peers,
        }

    @asynccontextmanager
    async def provision(
        self,
        task: Task | None = None,
        *,
        checkpoint: str | None = None,
        key: str | None = None,
        depends: object = None,
        durable: Durable | bool | None = None,
    ) -> AsyncIterator[Runtime]:
        """Provision (and on exit tear down) a box from this agent's runtime
        policy, resolved for `task` when given and started from `checkpoint` when
        given; share it via `run(..., runtime=box)`.

        The box comes with its owner (`box.Box`): runs that borrow it checkpoint it
        through the owner (per call when they or this agent save per call), and an
        interaction retry on it has the owner restore it for every seat. With a
        `Durable` installed the box is durable, keyed as `run` is (its position, or
        `key`): every run on it is labelled with the key, and a later provision under
        the key (a crashed episode run again) starts the box from the latest
        checkpoint those runs recorded, so their durable calls resume against it. Its
        fingerprint is the task, the runtime policy, the durable's parts and
        `depends`; a changed one starts fresh. A box started from `checkpoint` is
        durable only with `key`; the checkpoint is then part of its fingerprint, and the
        box starts from it only until its runs record a later one."""
        config = (
            resolve_runtime_config(self.runtime_config, task, self._warned_resources)
            if task is not None
            else self.runtime_config
        )
        found: Durable | None = None
        box_fingerprint: str | None = None
        ref: str | None = None
        start: Checkpoint | None = None
        if checkpoint is not None:
            # When it was taken is unknown: time 0 orders it before every checkpoint
            # recorded on the box, so a durable call on it continues none of them but
            # this one (`Agent._continuation`).
            start = Checkpoint(id=checkpoint, turns=0, time=0.0)
        if key is None and checkpoint is not None:
            durable = False
        found, key = plan_call(self.name, key, durable)
        if key is not None:
            parts = box_parts(task, self, found, depends)
            if checkpoint is not None:
                parts["checkpoint"] = checkpoint
            box_fingerprint = fingerprint(parts)
            scope = found.scope if found is not None else {}
            ref = box_ref(scope, key, box_fingerprint)
            # The box's runs moved it past where it started: they restore it.
            start = box_checkpoint(found, ref) or start
            if start is not None and start.id != checkpoint:
                logger.info("box %r restores from checkpoint %s", key, start.id)
        if start is not None:
            restore_config(config, start.id)  # raises when the runtime cannot
        async with provision_box(
            config,
            # Keep sandbox startup task-neutral: this box may later host another task.
            env=task.runtime_env() if task is not None else {},
            checkpoint=start,
            save=self.config.save == "per_call",
            key=key,
            fingerprint=box_fingerprint,
            ref=ref,
            durable=found,
        ) as runtime:
            yield runtime


class _EpisodeAgent(Agent):
    """One role's `Agent` for one episode, built fresh each time (a cheap
    bundle of references — expensive resources are env-owned and borrowed, so no
    state spans concurrent episodes): traces get their agent standing the moment
    they're created, finished ones land in `completed` (the episode's traces),
    each run takes one of the episode's agent permits. The taskset's shared tool
    servers ride only its own tasks — on an env-minted task they'd wrongly put MCP
    in play (`tools=` overrides)."""

    def __init__(
        self,
        config: AgentConfig,
        *,
        interception: Interception | None,
        name: str,
        shared_tools: Mapping[str, SharedToolServer],
        task_cls: type[Task],
        gate: asyncio.Semaphore | None,
        completed: list[Trace],
        on_trace: Callable[[Trace], None] | None,
        on_discard: Callable[[Trace], None] | None,
        warned_resources: set,
        peers: dict[str, str] | None = None,
    ) -> None:
        super().__init__(config, interception=interception)
        self._peers = peers
        # Resource warnings dedupe env-wide, not per episode.
        self._warned_resources = warned_resources
        self._name = name
        self._shared_tools = shared_tools
        self._task_cls = task_cls
        self._gate = gate
        self._completed = completed
        self._on_trace = on_trace
        self._on_discard = on_discard

    def _shared_for(
        self, task: Task | None, resume: Trace | None = None
    ) -> Mapping[str, SharedToolServer]:
        if task is None and resume is not None:
            task = resume.task.rebuild()
        return self._shared_tools if isinstance(task, self._task_cls) else {}

    def _watch(
        self, on_trace: Callable[[Trace], None] | None
    ) -> Callable[[Trace], None]:
        last: Trace | None = None

        def watch(trace: Trace) -> None:
            nonlocal last
            if trace.agent is not None:
                trace.agent.name = self._name
                trace.agent.trainable = self.trainable
            if self._peers is not None:
                self._peers[self._name] = trace.id
            # A per-agent retry mints a replacement: the abandoned attempt's trace
            # must leave live views (only the final one joins the episode).
            if last is not None and self._on_discard is not None:
                self._on_discard(last)
            last = trace
            if (runs := _scoring_runs.get()) is not None:
                runs.append((self, trace))
            if self._on_trace is not None:
                self._on_trace(trace)
            if on_trace is not None:
                on_trace(trace)

        return watch

    def _scoring_callback(self, callback):
        if callback is None:
            return None

        async def score(trace, runtime):
            parent_runs = _scoring_runs.get()
            runs = []
            token = _scoring_runs.set(runs)
            # The solver is idle while an Env runs a judge. Release its permit
            # so a one-agent episode can enter the nested judge rollout.
            if self._gate is not None:
                self._gate.release()
            try:
                try:
                    await callback(trace, runtime)
                finally:
                    _scoring_runs.reset(token)
                    if self._gate is not None:
                        await run_shielded(self._gate.acquire())
            except BaseException:
                for agent, child in runs:
                    agent._completed[:] = [
                        t for t in agent._completed if t is not child
                    ]
                    if agent._on_discard is not None:
                        agent._on_discard(child)
                    if child.usage is not None:
                        trace.extra_usage.append(child.usage)
                    trace.extra_usage.extend(child.extra_usage)
                raise
            else:
                if parent_runs is not None:
                    parent_runs.extend(runs)

        return score

    async def run(
        self,
        task: Task | None = None,
        *,
        runtime: Runtime | None = None,
        tools: Mapping[str, SharedToolServer] | None = None,
        on_trace: Callable[[Trace], None] | None = None,
        collect_artifacts: bool = False,
        checkpoint: str | None = None,
        resume: Trace | None = None,
        at: str | int | None = None,
        note: str | None = None,
        on_save: OnSave | None = None,
        key: str | None = None,
        depends: object = None,
        durable: Durable | bool | None = None,
        on_score: Callable[[Trace, Runtime], Awaitable[None]] | None = None,
    ) -> Trace:
        if durable is not False:
            # The call resolves as durable or plain first (a durable one may run
            # nothing); the run it makes comes back through here, `durable=False`.
            trace = await super().run(
                task,
                runtime=runtime,
                tools=tools,
                on_trace=on_trace,
                collect_artifacts=collect_artifacts,
                checkpoint=checkpoint,
                resume=resume,
                at=at,
                note=note,
                on_save=on_save,
                key=key,
                depends=depends,
                durable=durable,
                on_score=on_score,
            )
            if trace.kept:
                # A kept trace is part of this episode's result too: the attempt that
                # made it never landed.
                self._completed.append(trace)
            return trace
        async with self._gate or nullcontext():
            trace = await super().run(
                task,
                runtime=runtime,
                tools=tools if tools is not None else self._shared_for(task, resume),
                on_trace=self._watch(on_trace),
                collect_artifacts=collect_artifacts,
                checkpoint=checkpoint,
                resume=resume,
                at=at,
                note=note,
                on_save=on_save,
                on_score=self._scoring_callback(on_score),
                durable=False,
            )
        self._completed.append(trace)
        return trace

    @asynccontextmanager
    async def interaction(
        self,
        task: Task | None = None,
        *,
        runtime: Runtime | None = None,
        tools: Mapping[str, SharedToolServer] | None = None,
        on_trace: Callable[[Trace], None] | None = None,
        checkpoint: str | None = None,
        resume: Trace | None = None,
        at: str | int | None = None,
        note: str | None = None,
        on_save: OnSave | None = None,
        key: str | None = None,
        depends: object = None,
        durable: Durable | bool | None = None,
    ) -> AsyncIterator["Interaction | DurableInteraction"]:
        """The agent's `interaction`, with every trace stamped with its standing
        at mint and captured in `completed` at close — an interaction driven from
        `Env.run` stays crash-safe. Setup, each active segment, and close acquire
        an agent permit independently; the interaction holds no permit while awaiting
        its caller, so peer interactions still interleave where an episode plays one
        agent at a time."""
        if durable is not False:
            # The call resolves as durable or plain first; its live exchange comes
            # back through here, `durable=False`.
            async with super().interaction(
                task,
                runtime=runtime,
                tools=tools,
                on_trace=on_trace,
                checkpoint=checkpoint,
                resume=resume,
                at=at,
                note=note,
                on_save=on_save,
                key=key,
                depends=depends,
                durable=durable,
            ) as exchange:
                yield exchange
            if isinstance(exchange, DurableInteraction) and exchange.kept:
                self._completed.append(exchange.trace)
            return
        trace: Trace | None = None

        def remember(current: Trace) -> None:
            nonlocal trace
            trace = current
            if on_trace is not None:
                on_trace(current)

        try:
            async with super().interaction(
                task,
                runtime=runtime,
                tools=tools if tools is not None else self._shared_for(task, resume),
                on_trace=self._watch(remember),
                checkpoint=checkpoint,
                resume=resume,
                at=at,
                note=note,
                on_save=on_save,
                durable=False,
            ) as interaction:
                yield interaction
        finally:
            # `Agent.interaction()` may fail before yielding (e.g. task/harness setup).
            # Its trace is minted first, so retain that failed rollout in the episode.
            if trace is not None and trace.is_completed:
                self._completed.append(trace)


def make_agent(
    config: AgentConfig,
    *,
    interception: Interception | None = None,
) -> Agent:
    """The agent for a config; `interception` injects a live resource to borrow,
    everything else comes from the config."""
    return Agent(config, interception=interception)


AgentT = TypeVar("AgentT", bound=Agent, default=Agent)
MakeAgent = Callable[[str, AgentConfig], AgentT]
"""An agent factory keyed by name — what `Agents` calls per scraped config field."""


class Agents(Generic[AgentT]):
    """A config's agents, addressed by attribute: every top-level `AgentConfig`
    field becomes an `Agent` under the field's name (`agents.solver`)."""

    def __init__(self, config, make: MakeAgent[AgentT] | None = None) -> None:
        self._agents: dict[str, AgentT] = {
            name: cast(AgentT, make_agent(value)) if make is None else make(name, value)
            for name, value in agent_config_fields(config).items()
        }

    def __getattr__(self, name: str) -> AgentT:
        # self.__dict__ directly: attribute lookup re-entering __getattr__ before
        # __init__ ran (copy/unpickle) must raise, not recurse.
        agents = self.__dict__.get("_agents")
        if agents is None or name not in agents:
            raise AttributeError(
                f"no agent {name!r}; this config declares "
                f"{sorted(agents) if agents else []}"
            )
        return agents[name]
