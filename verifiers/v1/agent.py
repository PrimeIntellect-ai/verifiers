"""The Agent: a reusable (harness x model x runtime) value with one executable
arrow — `agent.run(task) -> Trace`; `runtime=` borrows a live box,
`provision(task)` hands you one. `agent.interaction(task)` holds the rollout open
turn-by-turn, with the caller as the run's user — one `turn()` per harness segment;
who computes the turns is control flow, not a framework concept.
Inject a live `Interception` to share servers across agents (a pool belongs to
what spans agents, never to one agent); an entered agent (`async with`) owns one
server; un-entered, each run brings its own."""

import asyncio
import logging
from collections.abc import AsyncIterator, Awaitable, Callable, Mapping
from contextlib import asynccontextmanager, nullcontext
from dataclasses import dataclass, field, replace
from typing import Generic, Literal, Self, cast
from weakref import WeakValueDictionary

from typing_extensions import TypeVar

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
from verifiers.v1.harness import Harness
from verifiers.v1.interception import Interception, InterceptionServer
from verifiers.v1.mcp import SharedToolServer
from verifiers.v1.rollout import OnSave, Rollout, RolloutTimeouts
from verifiers.v1.runtimes import (
    Runtime,
    RuntimeConfig,
    SubprocessConfig,
    provision_runtime,
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


Reopen = Callable[[Task, str | None, Trace], Awaitable["Rollout"]]
"""Build and open a replacement rollout: `(task, checkpoint, the trace it replaces)`."""


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
    `interaction.trace` becomes the replacement's."""

    def __init__(
        self,
        run: "Rollout",
        gate: asyncio.Semaphore | None = None,
        *,
        opening: _Opening | None = None,
        reopen: Reopen | None = None,
        retries: int = 0,
    ) -> None:
        self._run = run
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
        """Checkpoint the box this interaction owns, between turns, and record it on
        the trace (`trace.checkpoints`), matching the conversation so far: what
        `agent.run(resume=trace)` / `agent.interaction(resume=trace)` continue from.
        Waits for a running turn to finish. A borrowed box raises (its owner
        checkpoints it)."""
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
        Unsupported steering raises; it never silently changes to queue mode.
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
        """
        if mode not in ("steer", "queue"):
            raise ValueError("message mode must be 'steer' or 'queue'")
        return await self._send(message, message_id=message_id, mode=mode, wake=True)

    async def steer(
        self, message: str, *, message_id: str | None = None, wake: bool = False
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
        while True:
            turns_before = self.trace.num_turns
            nodes_before = len(self.trace.nodes)
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
        checkpoint = self.latest_checkpoint
        return first and (
            checkpoint is None or (checkpoint.turns == 0 and not self.failed_attempts)
        )

    def _resumable(self, first: bool) -> bool:
        """Whether a failed turn has a sound state to resume from."""
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
            conversation = trace.conversation_at(checkpoint.turns)
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
        failed = await self._run.close()
        self.failed_attempts.append(failed)
        self._run = await self._reopen(task, start, failed)
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
        self, run_is_local: bool, task: Task, shared_tools: Mapping
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
            run_is_local and not shared_tools and not task.toolsets(task.config)
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
        if start is not None and runtime is not None:
            raise ValueError(
                "this trace restores its box from a checkpoint, so it cannot also "
                "run on a borrowed runtime"
            )
        if start is not None:
            try:
                restore_config(self.runtime_config, start.id)
            except TypeError as e:
                raise ValueError(
                    f"this trace's box is checkpoint {start.id!r}, but the agent's "
                    f"{self.runtime_config.type} runtime cannot restore checkpoints: "
                    f"{e}"
                ) from e
        task = task if task is not None else resume.task.rebuild()
        messages = resume.conversation_at(start.turns if start is not None else None)
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
            checkpoint=start.id if start is not None else None,
            prefix=prefix,
            source=resume,
        )

    def _check_saves(self, runtime: Runtime | None) -> None:
        """`save = "per_call"` on a box the run owns needs a runtime that restores
        checkpoints (a borrowed box is never checkpointed)."""
        if self.config.save != "per_call" or runtime is not None:
            return
        try:
            restore_config(self.runtime_config, "probe")
        except TypeError as e:
            raise ValueError(
                f"save='per_call' checkpoints the run's box, but the agent's "
                f"{self.runtime_config.type} runtime has no checkpoints to continue "
                f"from: {e}"
            ) from e

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
        on the trace and handed to `on_save(trace, checkpoint)`."""
        if self._closed:
            raise RuntimeError("Agent is closed; create a new agent")
        opening = self._opening(
            task, resume, at, note, checkpoint, runtime, interactive=False
        )
        self._check_saves(runtime)
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
    ) -> AsyncIterator[Interaction]:
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
        infrastructure again on a fresh box (never into a borrowed one).

        `resume`, `at` and `note` continue a trace (see `run`); without `note`, a
        conversation that ends on the agent's reply waits for the first
        `turn(message)`, which continues it. `interaction.checkpoint()` checkpoints
        the box between turns. With `config.save = "per_call"`, the owned box is
        checkpointed after every model response (`on_save(trace, checkpoint)`);
        `interaction.latest_checkpoint` holds the latest."""
        if self._closed:
            raise RuntimeError("Agent is closed; create a new agent")
        self._check_resume_support()
        opening = self._opening(
            task, resume, at, note, checkpoint, runtime, interactive=True
        )
        self._check_saves(runtime)
        shared_tools = dict(tools or {})

        def make_run(
            task: Task, checkpoint: str | None, resumes: Trace | None
        ) -> Rollout:
            return Rollout(
                task=task,
                has_user=True,
                on_trace=on_trace,
                on_save=on_save,
                resumes=resumes,
                **self._rollout_params(task, runtime, shared_tools, checkpoint),
            )

        async def reopen(task: Task, checkpoint: str | None, failed: Trace) -> Rollout:
            # A failed open is captured on the replacement: its step fails at once.
            replacement = make_run(task, checkpoint, failed)
            await replacement.open()
            return replacement

        run = make_run(opening.task, opening.checkpoint, opening.source)
        interaction = Interaction(
            run,
            gate=self._gate,
            opening=opening,
            reopen=reopen if runtime is None else None,
            retries=self.config.interaction_retries,
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
        validate_pairing(
            harness,
            type(task),
            runtime_config,
            tools=[*task.toolsets(task.config), *shared_tools.values()],
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
            "interception": self._interception_for(run_is_local, task, shared_tools),
            "runtime": runtime,
        }

    @asynccontextmanager
    async def provision(
        self, task: Task | None = None, *, checkpoint: str | None = None
    ) -> AsyncIterator[Runtime]:
        """Provision (and on exit tear down) a box from this agent's runtime
        policy, resolved for `task` when given and started from `checkpoint` when
        given; share it via `run(..., runtime=box)`."""
        config = (
            resolve_runtime_config(self.runtime_config, task, self._warned_resources)
            if task is not None
            else self.runtime_config
        )
        if checkpoint is not None:
            config = restore_config(config, checkpoint)
        async with provision_runtime(config) as runtime:
            # Keep sandbox startup task-neutral: this box may later host another task.
            runtime.env = dict(task.runtime_env()) if task is not None else {}
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
    ) -> None:
        super().__init__(config, interception=interception)
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
            # A per-agent retry mints a replacement: the abandoned attempt's trace
            # must leave live views (only the final one joins the episode).
            if last is not None and self._on_discard is not None:
                self._on_discard(last)
            last = trace
            if self._on_trace is not None:
                self._on_trace(trace)
            if on_trace is not None:
                on_trace(trace)

        return watch

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
    ) -> Trace:
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
    ) -> AsyncIterator[Interaction]:
        """The agent's `interaction`, with every trace stamped with its standing
        at mint and captured in `completed` at close — an interaction driven from
        `Env.run` stays crash-safe. Setup, each active segment, and close acquire
        an agent permit independently; the interaction holds no permit while awaiting
        its caller, so peer interactions still interleave where an episode plays one
        agent at a time."""
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
