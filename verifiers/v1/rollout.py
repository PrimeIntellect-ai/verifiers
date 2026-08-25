"""Lifecycle of one agent rollout."""

import asyncio
import contextlib
import logging
import time
from collections.abc import Callable
from contextlib import AsyncExitStack
from dataclasses import dataclass

from verifiers.v1.clients import ModelContext
from verifiers.v1.configs.agent import AgentConfig
from verifiers.v1.configs.runtime import NetworkPolicyConfig
from verifiers.v1.errors import (
    HarnessError,
    HarnessFinalizationError,
    RolloutError,
    TaskError,
    ToolsetError,
    boundary,
)
from verifiers.v1.harness import Harness, HarnessSession
from verifiers.v1.interception import Interception, serve_interception
from verifiers.v1.mcp import SharedToolServer, serve_tools
from verifiers.v1.runtimes import (
    ModalConfig,
    Runtime,
    RuntimeConfig,
    make_runtime,
)
from verifiers.v1.session import RolloutLimits, RolloutSession, hook_boundary
from verifiers.v1.state import state_cls
from verifiers.v1.task import Task
from verifiers.v1.trace import AgentInfo, Trace, TraceTask
from verifiers.v1.types import Messages, Request, Response, SystemMessage, UserMessage
from verifiers.v1.utils.decorators import discover_decorated, invoke

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class RolloutTimeouts:
    """Per-stage rollout timeouts, each bounding one rollout stage."""

    setup: float | None = None
    """Timeout (in seconds) for the task + harness setup hooks."""
    agent: float | None = None
    """Timeout (in seconds) for the agent's solve attempt."""
    finalize: float | None = None
    """Timeout (in seconds) for the task + harness finalize hooks."""
    scoring: float | None = None
    """Timeout (in seconds) for the task + harness metrics + scoring hooks."""


class Rollout:
    """Manages one rollout's lifecycle (open, step, close)."""

    def __init__(
        self,
        *,
        task: Task,
        agent_config: AgentConfig,
        harness: Harness,
        ctx: ModelContext,
        runtime_config: RuntimeConfig,
        has_user: bool = False,
        timeouts: RolloutTimeouts,
        limits: RolloutLimits,
        shared_tools: dict[str, SharedToolServer] | None = None,
        interception: Interception | None = None,
        runtime: Runtime | None = None,
        on_trace: Callable[[Trace], None] | None = None,
    ) -> None:
        self.task = task
        self.harness = harness
        self.ctx = ctx
        self.runtime_config = runtime_config
        self._has_user = has_user
        self._timeouts = timeouts
        self._agent_time_remaining = self._timeouts.agent
        self._shared_tools = shared_tools or {}
        self._interception = interception
        self.runtime = runtime
        self._owns_runtime = runtime is None
        self._previous_runtime_env: dict[str, str] | None = None
        self.trace: Trace = Trace(
            task=TraceTask(
                type=type(task).__name__,
                data=task.data,
                key=task.key,
                hash=task.hash,
            ),
            state=state_cls(type(task))(),
            # The seat's resolved config, role overrides included — the agent
            # this trace can be reproduced with.
            agent=AgentInfo(config=agent_config),
        )
        if on_trace is not None:
            on_trace(self.trace)
        interceptors = [
            (hook_boundary(fn, allow_trace=False), fn)
            for fn in discover_decorated(task, "intercept")
        ]
        stops = [(hook_boundary(fn, allow_trace=True), fn) for fn in task.hooks("stop")]
        self._session = RolloutSession(
            ctx=ctx,
            trace=self.trace,
            network_policy=(
                runtime_config
                if isinstance(runtime_config, NetworkPolicyConfig)
                else NetworkPolicyConfig(
                    allow=[]
                    if isinstance(runtime_config, ModalConfig)
                    and not runtime_config.network_access
                    else ["*"]
                )
            ),
            trace_stops=[fn for boundary, fn in stops if boundary is Trace],
            limits=limits,
            request_interceptors=[
                fn for boundary, fn in interceptors if boundary is Request
            ],
            response_interceptors=[
                fn for boundary, fn in interceptors if boundary is Response
            ],
            request_stops=[fn for boundary, fn in stops if boundary is Request],
            response_stops=[fn for boundary, fn in stops if boundary is Response],
        )
        self._stack = AsyncExitStack()
        self._failed = False
        self._failure: Exception | None = None
        self._opened = False
        self._closed = False
        self._endpoint: str | None = None
        self._urls: dict[str, str] = {}
        self._harness_session: HarnessSession | None = None
        self.deadline_at: float | None = None
        """The active harness segment's absolute deadline (event-loop clock), or
        None between segments / when unbounded. An interaction spends one cumulative
        `timeouts.agent` budget only while its own segments run, so time awaiting
        the caller (including another interleaved agent) cannot starve it."""

    @property
    def ok(self) -> bool:
        """Whether the exchange can continue: nothing failed, nothing stopped it."""
        return not self._failed and self.trace.stop_condition is None

    @property
    def closed(self) -> bool:
        """Whether `close()` (or `abort()`) already ran — no further segments."""
        return self._closed

    @property
    def failure(self) -> Exception | None:
        """The original exception most recently captured onto the trace."""
        return self._failure

    def fail(self, error: Exception) -> None:
        """Record `error` as this rollout's outcome (captured onto the trace, the
        remaining stages skipped) — the run's owner reporting a failure the run
        itself couldn't see, e.g. its user raising between segments."""
        if not self._owns_runtime and self.runtime is not None and self.runtime.stopped:
            # The owner tore the borrowed box down mid-run — a lifetime bug in the
            # borrowing program: raise to the caller instead of capturing a
            # misattributed error onto the trace.
            raise ValueError(
                f"borrowed runtime {self.runtime.name!r} was torn down by its owner "
                "mid-run; keep the provisioning context open until every run "
                "placed into the box has completed"
            ) from error
        if not isinstance(error, RolloutError):
            logger.exception("unexpected error in rollout %s", self.trace.id)
        self._failed = True
        self._failure = error
        self.trace.record_error(error)

    def _release_borrowed_runtime(self) -> None:
        if self._previous_runtime_env is not None and self.runtime is not None:
            self.runtime.env = self._previous_runtime_env
            self._previous_runtime_env = None
            self.runtime.borrow_lock.release()

    async def open(self) -> bool:
        """Boot the rollout's world up to the point where segments can run: start
        (or borrow) the runtime, run task + harness setup, bring up the
        interception slot and tool servers. Returns whether the exchange can
        proceed; a setup failure is captured onto the trace."""
        self._opened = True
        self.trace.timing.boot.start = time.time()
        if self._owns_runtime:
            self.runtime = make_runtime(self.runtime_config, name=self.trace.id)
        elif self.runtime.stopped:
            # A lifetime bug in the borrowing program: raise to the caller instead
            # of capturing onto the trace.
            raise ValueError(
                f"borrowed runtime {self.runtime.name!r} was already torn "
                "down by its owner; keep the provisioning context open for every run "
                "placed into the box"
            )
        runtime = self.runtime
        assert self.trace.agent is not None  # minted with the trace
        self.trace.agent.runtime = runtime.info
        logger.info(
            "rollout start: id=%s task=%s harness=%s runtime=%s",
            self.trace.id,
            self.task.data.idx,
            self.harness.config.name,
            self.runtime_config.type,
        )
        try:
            runtime_env = dict(self.task.runtime_env())
            if self._owns_runtime:
                runtime.env = runtime_env
            else:
                await runtime.borrow_lock.acquire()
                self._previous_runtime_env = runtime.env
                # The owner's defaults return after the borrow; they do not belong to
                # this task and must not flow into its processes.
                runtime.env = runtime_env
            if self.task.data.prompt is None and not self._has_user:
                raise TaskError(
                    "task has no prompt and no user to open the conversation; set "
                    "task.prompt, or drive the run through agent.interaction() and open "
                    "it with the first turn(message)"
                )
            if self._owns_runtime:
                await runtime.start()
            await runtime.prepare_setup()
            now = time.time()
            self.trace.timing.boot.end = now
            self.trace.timing.setup.start = now
            # Task setup and harness provisioning share one setup-stage deadline.
            setup_deadline = (
                None
                if self._timeouts.setup is None
                else asyncio.get_running_loop().time() + self._timeouts.setup
            )
            async with (
                boundary(TaskError, "task setup"),
                asyncio.timeout_at(setup_deadline),
            ):
                await invoke(self.task.setup, {"trace": self.trace, "runtime": runtime})
            async with (
                boundary(HarnessError, "harness setup"),
                asyncio.timeout_at(setup_deadline),
            ):
                await self.harness.setup(runtime)
            async with boundary(ToolsetError, "building tool servers"):
                toolsets = self.task.toolsets(self.task.config)
            # `base_url` is the interception server's reachable URL for this rollout.
            # The harness reaches the model at `{base_url}/v1`; tool servers reach this
            # rollout's `/state` + `/task` at `base_url` — it's universally reachable
            # (the interception is exposed whenever any consumer is remote).
            (
                base_url,
                model_secret,
                state_secret,
            ) = await self._stack.enter_async_context(
                serve_interception(
                    self._interception,
                    runtime,
                    self._session,
                    toolsets,
                    self._shared_tools,
                )
            )
            self._endpoint = f"{runtime.host_url(base_url)}/v1"
            self._secret = model_secret
            self._urls = await self._stack.enter_async_context(
                serve_tools(
                    toolsets,
                    runtime,
                    shared=self._shared_tools,
                    state_secret=state_secret,
                    state_route=self.trace.id,
                    state_base=base_url,
                )
            )
            # Setup and service provisioning are complete. Apply the runtime's
            # execution policy while preserving the framework routes the agent uses.
            await runtime.prepare_execution([self._endpoint, *self._urls.values()])
            async with boundary(HarnessError, "opening harness session"):
                harness_data = self.trace.task.data
                if (
                    self._session.request_interceptors
                    and harness_data.prompt is not None
                ):
                    prompt = harness_data.prompt
                    system_prompt = harness_data.system_prompt
                    if isinstance(prompt, str):
                        system_prompt, prompt = self.harness.resolve_prompt(
                            harness_data
                        )
                    messages = (
                        [UserMessage(content=prompt)]
                        if isinstance(prompt, str)
                        else list(prompt)
                    )
                    has_system = system_prompt is not None
                    if has_system:
                        messages.insert(0, SystemMessage(content=system_prompt))
                    prepared, rewrites = await self._session.prepare_users(
                        Request(messages=messages)
                    )
                    messages = prepared.messages
                    self.trace.request_rewrites.extend(rewrites)
                    if has_system:
                        messages = messages[1:]
                    rewritten_prompt = (
                        messages[0].content
                        if isinstance(prompt, str)
                        and len(messages) == 1
                        and isinstance(messages[0], UserMessage)
                        and isinstance(messages[0].content, str)
                        else messages
                    )
                    harness_data = harness_data.model_copy(
                        update={
                            "prompt": rewritten_prompt,
                            "system_prompt": system_prompt,
                        }
                    )
                if not self._session.stopped:
                    session_kwargs = (
                        {"tool_interception_url": f"{runtime.host_url(base_url)}/tool"}
                        if self.harness.SUPPORTS_TOOL_INTERCEPTION
                        and (
                            self._session.request_interceptors
                            or self._session.request_stops
                        )
                        else {}
                    )
                    self._harness_session = await self.harness.session(
                        self.ctx,
                        self.trace,
                        runtime,
                        self._endpoint,
                        self._secret,
                        self._urls,
                        harness_data,
                        **session_kwargs,
                    )
        except Exception as e:  # noqa: BLE001 - setup boundary records every rollout failure
            self.fail(e)
            return False
        except BaseException:
            # A cancellation mid-setup kills the driver's await with it, so no
            # caller reaches close() — free the started runtime and entered
            # servers here rather than relying on the driver's own guard.
            await self.abort()
            raise
        now = time.time()
        self.trace.timing.setup.end = now
        self.trace.timing.agent.start = now
        return not self._session.stopped

    async def step(self, messages: Messages | None = None) -> bool:
        """Run ONE segment: the harness program to its exit. With `messages`, the
        segment resumes the exchange with the user's turn(s) (`Harness.resume` —
        for an exchange the user opens, this is also the first segment, on an
        empty conversation); without, it launches on the task's own prompt.
        Returns whether the exchange can continue — a refused turn (limit, @stop),
        a failure (an expired agent timeout included), or a segment that made no
        progress all end it."""
        if not self._opened or self._closed or not self.ok:
            return False
        trace = self.trace
        turns_before = trace.num_turns
        primary_reply_before = trace.primary_reply
        trace.primary_reply = None
        loop = asyncio.get_running_loop()
        segment_start = loop.time()
        self.deadline_at = (
            None
            if self._agent_time_remaining is None
            else segment_start + max(0.0, self._agent_time_remaining)
        )
        # Prefer an intercepted model/tool error to the harness exit it caused.
        try:
            async with asyncio.timeout_at(self.deadline_at):
                assert self._harness_session is not None
                if messages is not None and self._session.request_interceptors:
                    prepared, rewrites = await self._session.prepare_users(
                        Request(messages=messages)
                    )
                    messages = prepared.messages
                    self.trace.request_rewrites.extend(rewrites)
                    if self._session.stopped:
                        return False
                await self._harness_session.turn(messages)
        except TimeoutError as e:
            # An expired rollout deadline is the agent breaking its time budget —
            # an agent failure, never a clean stop. A TimeoutError from the
            # harness's own I/O with no expired deadline stays the raw failure.
            if self.deadline_at is not None and (loop.time() >= self.deadline_at):
                self.fail(
                    HarnessError(
                        f"agent timeout: rollout exceeded its "
                        f"{self._timeouts.agent:g}s budget"
                    )
                )
            else:
                self.fail(e)
            return False
        except Exception as e:  # noqa: BLE001 - harness boundary records every rollout failure
            if self._session.stopped:
                return False
            real = self._session.error
            if real is not None and isinstance(e, RolloutError):
                real.__cause__ = e
                self.fail(real)
            else:
                self.fail(e)
            return False
        finally:
            if trace.num_turns == turns_before:
                trace.primary_reply = primary_reply_before
            if self._agent_time_remaining is not None:
                self._agent_time_remaining = max(
                    0.0, self._agent_time_remaining - (loop.time() - segment_start)
                )
            self.deadline_at = None
        if self._session.error is not None:
            self.fail(self._session.error)
            return False
        # A segment that committed nothing can't be waiting on the user; treating
        # it as continuable would consult the user against a conversation that
        # never moved, forever.
        return self.ok and trace.num_turns > turns_before

    async def abort(self) -> None:
        """Free everything this run holds — the entered servers and an owned
        runtime — without finalizing or scoring: the escape path when an exception
        (a cancellation mid-setup, a lifetime bug raised to the caller) means the
        driver will never reach `close()`. Safe after a partial `close()`."""
        self._closed = True
        if self._harness_session is not None:
            with contextlib.suppress(Exception):
                await self._harness_session.close()
        with contextlib.suppress(Exception):
            await self._stack.aclose()
        if self.runtime is not None:
            with contextlib.suppress(Exception):
                await self.harness.cleanup(self.trace, self.runtime)
        self._release_borrowed_runtime()
        if self._owns_runtime and self.runtime is not None:
            with contextlib.suppress(Exception):
                await self.runtime.stop()

    async def close(self) -> Trace:
        """Finish the rollout: tool servers and interception down, task `finalize`
        and per-rollout scoring (skipped when the run already failed — but a stopped
        run is complete and scores its partial trajectory), then runtime teardown.
        Idempotent; always returns the trace."""
        if self._closed:
            return self.trace
        self._closed = True
        trace = self.trace
        runtime = self.runtime
        try:
            if self._harness_session is not None:
                try:
                    await self._harness_session.close()
                except HarnessFinalizationError:
                    raise
                except Exception:
                    # Generation already completed. A transport teardown failure
                    # must not discard its otherwise scoreable trajectory.
                    logger.warning(
                        "harness session close failed (rollout %s)",
                        trace.id,
                        exc_info=True,
                    )
            try:
                await self._stack.aclose()
            finally:
                if trace.timing.agent.start and not trace.timing.agent.end:
                    trace.timing.agent.end = time.time()
            if not self._failed and self._opened:
                trace.timing.finalize.start = time.time()
                async with boundary(TaskError, "task finalize"):
                    await asyncio.wait_for(
                        invoke(
                            self.task.finalize, {"trace": trace, "runtime": runtime}
                        ),
                        self._timeouts.finalize,
                    )
                now = time.time()
                trace.timing.finalize.end = now
                trace.timing.scoring.start = now
                async with boundary(TaskError, "scoring"):
                    # Cross-trace judgement runs later, after the runtime is gone.
                    await asyncio.wait_for(
                        asyncio.gather(
                            self.task.score(trace, runtime),
                            self.harness.score(trace, runtime),
                        ),
                        self._timeouts.scoring,
                    )
                trace.timing.scoring.end = time.time()
        except Exception as e:  # noqa: BLE001 - finalize boundary records every rollout failure
            self.fail(e)
        finally:
            if self._harness_session is not None:
                with contextlib.suppress(Exception):
                    await self._harness_session.close()
            with contextlib.suppress(Exception):
                await self._stack.aclose()
            trace.is_completed = True
            trace.ok = not self._failed
            now = time.time()
            for span in (
                trace.timing.boot,
                trace.timing.setup,
                trace.timing.agent,
                trace.timing.finalize,
                trace.timing.scoring,
            ):
                if span.start and not span.end:
                    span.end = now
            trace.split_agent_time()
            if runtime is not None:
                try:
                    await self.harness.cleanup(trace, runtime)
                except Exception:
                    logger.warning(
                        "harness cleanup failed (rollout %s)", trace.id, exc_info=True
                    )
            self._release_borrowed_runtime()
            # Tear down here — the env's `score()` (later) needs only the traces,
            # not a live runtime. A borrowed runtime is its creator's to tear down,
            # not this rollout's.
            if self._owns_runtime and runtime is not None:
                try:
                    await runtime.stop()
                except Exception:
                    logger.warning(
                        "runtime teardown failed (rollout %s)", trace.id, exc_info=True
                    )
        logger.info(
            "rollout done: id=%s task=%s reward=%.3f turns=%d stop=%s",
            trace.id,
            self.task.data.idx,
            trace.reward,
            trace.num_turns,
            trace.last_error.type if trace.last_error else trace.stop_condition,
        )
        return trace
