"""A rollout: one trajectory — drive a harness segment by segment and score its trace.

A rollout's exchange is a sequence of SEGMENTS: the harness program runs until it
yields (= exits), the run's user answers its final message, and the next segment
resumes the exchange with that answer (`Harness.resume` — a relaunch on the accreted
conversation by default, a native continuation for harnesses with their own session
state). The user loop lives between segments, at the exchange's natural turn
granularity — never inside the model boundary, so a harness's own tool loop can
never race or amputate it.

`RolloutRun` is the engine, a staged lifecycle: `open()` boots the world, each
`step()` runs one segment, `close()` finalizes, scores, and tears the world down —
each stage under its own timeout. `Agent` is its only driver: `Agent.run` is the
one-call single-segment form, `Agent.interaction` holds the run open and lets the
caller supply each user turn, one `turn()` per segment — who answers the program
(an env's control flow, a simulator agent, a game engine, a human) is the caller's
business, never this module's.
"""

import asyncio
import contextlib
import logging
import time
from collections.abc import AsyncIterator, Callable
from contextlib import AsyncExitStack, asynccontextmanager

from verifiers import __version__
from verifiers.v1.harness import Harness
from verifiers.v1.clients import ModelContext
from verifiers.v1.decorators import discover_decorated, invoke
from verifiers.v1.dialects import parse_message
from verifiers.v1.errors import (
    HarnessError,
    RolloutError,
    TaskError,
    ToolsetError,
    boundary,
)
from verifiers.v1.interception import (
    Interception,
    InterceptionServer,
    Slot,
    requires_tunnel,
)
from verifiers.v1.session import RolloutLimits, RolloutSession
from verifiers.v1.runtimes import (
    Runtime,
    RuntimeConfig,
    make_runtime,
)
from verifiers.v1.mcp import SharedToolServer, serve_tools
from verifiers.v1.state import state_cls
from verifiers.v1.task import Task, TaskData
from verifiers.v1.trace import AgentInfo, Trace, TraceTask, VersionInfo
from verifiers.v1.types import Messages
from verifiers.v1.utils.version import verifiers_commit

logger = logging.getLogger(__name__)


def _as_messages(raw: Messages) -> Messages:
    """A turn's messages may arrive typed or as wire dicts (env code naturally
    writes `{"role": "user", ...}`); the trace speaks typed, so normalize here."""
    return [parse_message(m) if isinstance(m, dict) else m for m in raw]


@asynccontextmanager
async def _serve_interception(
    interception: Interception | None,
    runtime: Runtime,
    session: RolloutSession,
    servers: list,
    shared_tools: dict[str, SharedToolServer],
) -> AsyncIterator[Slot]:
    """A slot on the shared interception when one was injected (its owner keeps the
    lifecycle), else on a per-rollout `InterceptionServer` owned — brought up and torn
    down — by this rollout."""
    if interception is not None:
        async with interception.acquire(session) as slot:
            yield slot
        return
    tunneled = requires_tunnel(
        runtime.is_local,
        [server.config for server in servers],
        shared_tools.values(),
    )
    server = InterceptionServer(requires_tunnel=tunneled)
    async with server:
        async with server.acquire(session) as slot:
            yield slot


class RolloutRun:
    """One rollout held open segment by segment.

    `open()` boots the world (runtime, setup, interception, tool servers); each
    `step()` runs ONE harness segment — a program run to its exit — resuming the
    exchange with the user turn(s) it's given; `close()` finalizes, scores, and
    tears the world down, returning the finished trace. Expected `RolloutError`s
    are captured onto the trace (a bad rollout is data, not a crash): `open` and
    `step` report continuability as a bool, and `close` always returns the trace.

    `wire_data` is the run's recorded view of the task — what `trace.task.data`
    says the harness saw (`Agent.interaction(mask_prompt=True)` masks the prompt here
    while the `task` object keeps the full row for its hooks and judges).
    `runtime` is a live box to run in instead of provisioning one; a borrowed
    runtime is neither started nor stopped here. `on_trace` observes the run's
    trace the moment it's minted, before any I/O."""

    def __init__(
        self,
        *,
        task: Task,
        harness: Harness,
        ctx: ModelContext,
        runtime_config: RuntimeConfig,
        wire_data: TaskData | None = None,
        has_user: bool = False,
        setup_timeout: float | None = None,
        harness_timeout: float | None = None,
        finalize_timeout: float | None = None,
        scoring_timeout: float | None = None,
        limits: RolloutLimits | None = None,
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
        self._setup_timeout = setup_timeout
        self._harness_time_remaining = harness_timeout
        self._finalize_timeout = finalize_timeout
        self._scoring_timeout = scoring_timeout
        self._shared_tools = shared_tools or {}
        self._interception = interception
        self.runtime = runtime
        self._owns_runtime = runtime is None
        self.trace: Trace = Trace(
            task=TraceTask(
                type=type(task).__name__,
                data=task.data if wire_data is None else wire_data,
            ),
            state=state_cls(type(task))(),
            verifiers=VersionInfo(version=__version__, commit=verifiers_commit()),
            # The seat's resolved identity, role overrides included.
            agent=AgentInfo(
                model=ctx.model,
                sampling=ctx.sampling,
                harness=harness.config,
            ),
        )
        if on_trace is not None:
            on_trace(self.trace)
        self._session = RolloutSession(
            ctx, self.trace, discover_decorated(task, "stop"), limits or RolloutLimits()
        )
        self._stack = AsyncExitStack()
        self._failed = False
        self._failure: Exception | None = None
        self._opened = False
        self._closed = False
        self._endpoint: str | None = None
        self._urls: dict[str, str] = {}
        self.deadline_at: float | None = None
        """The active harness segment's absolute deadline (event-loop clock), or
        None between segments / when unbounded. An interaction spends one cumulative
        `harness_timeout` budget only while its own segments run, so time awaiting
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
        self.trace.capture_error(error)

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
            if self.task.data.prompt is None and not self._has_user:
                raise TaskError(
                    "task has no prompt and no user to open the conversation; set "
                    "task.prompt, or drive the run through agent.interaction() and open "
                    "it with the first turn(message)"
                )
            if self._owns_runtime:
                await runtime.start()
            await runtime.prepare_setup()
            await runtime.prepare_setup()
            now = time.time()
            self.trace.timing.boot.end = now
            self.trace.timing.setup.start = now
            # Task setup and harness provisioning share one setup-stage deadline.
            setup_deadline = (
                None
                if self._setup_timeout is None
                else asyncio.get_running_loop().time() + self._setup_timeout
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
                tool_servers = self.task.tool_servers()
            # `base_url` is the interception server's reachable URL for this rollout.
            # The harness reaches the model at `{base_url}/v1`; tool servers reach this
            # rollout's `/state` + `/task` at `base_url` — it's universally reachable
            # (the interception is exposed whenever any consumer is remote).
            base_url, secret = await self._stack.enter_async_context(
                _serve_interception(
                    self._interception,
                    runtime,
                    self._session,
                    tool_servers,
                    self._shared_tools,
                )
            )
            self._endpoint = f"{runtime.host_url(base_url)}/v1"
            self._secret = secret
            self._urls = await self._stack.enter_async_context(
                serve_tools(
                    tool_servers,
                    runtime,
                    shared=self._shared_tools,
                    state_secret=secret,
                    state_base=base_url,
                )
            )
            # Setup and service provisioning are complete. Apply the runtime's
            # execution policy while preserving the framework routes the agent uses.
            await runtime.prepare_execution([self._endpoint, *self._urls.values()])
        except Exception as e:
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
        self.trace.timing.generation.start = now
        return True

    async def step(self, messages: Messages | None = None) -> bool:
        """Run ONE segment: the harness program to its exit. With `messages`, the
        segment resumes the exchange with the user's turn(s) (`Harness.resume` —
        for an exchange the user opens, this is also the first segment, on an
        empty conversation); without, it launches on the task's own prompt.
        Returns whether the exchange can continue — a refused turn (limit, @stop),
        a timeout, a failure, or a segment that made no progress all end it."""
        if not self._opened or self._closed or not self.ok:
            return False
        trace = self.trace
        turns_before = trace.num_turns
        loop = asyncio.get_running_loop()
        segment_start = loop.time()
        self.deadline_at = (
            None
            if self._harness_time_remaining is None
            else segment_start + max(0.0, self._harness_time_remaining)
        )
        # Prefer an intercepted model/tool error to the harness exit it caused.
        # A timeout still scores the partial trajectory.
        try:
            async with asyncio.timeout_at(self.deadline_at):
                await self.harness.run(
                    self.ctx,
                    trace,
                    self.runtime,
                    self._endpoint,
                    self._secret,
                    self._urls,
                    trace.task.data,
                    messages,
                )
        except TimeoutError as e:
            # Only the rollout deadline reads as a clean truncation; a TimeoutError
            # from the harness's own I/O with no expired deadline is a failure —
            # recording it as a stop would score a broken run as a partial success.
            if self.deadline_at is not None and (loop.time() >= self.deadline_at):
                trace.stop("harness_timeout")
            else:
                self.fail(e)
            return False
        except Exception as e:
            real = self._session.error
            if real is not None and isinstance(e, RolloutError):
                real.__cause__ = e
                self.fail(real)
            else:
                self.fail(e)
            return False
        finally:
            if self._harness_time_remaining is not None:
                self._harness_time_remaining = max(
                    0.0, self._harness_time_remaining - (loop.time() - segment_start)
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
        with contextlib.suppress(Exception):
            await self._stack.aclose()
        if self.runtime is not None:
            with contextlib.suppress(Exception):
                await self.harness.cleanup(self.trace, self.runtime)
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
            try:
                await self._stack.aclose()
            finally:
                if trace.timing.generation.start and not trace.timing.generation.end:
                    trace.timing.generation.end = time.time()
            if not self._failed and self._opened:
                trace.timing.finalize.start = time.time()
                async with boundary(TaskError, "task finalize"):
                    await asyncio.wait_for(
                        invoke(
                            self.task.finalize, {"trace": trace, "runtime": runtime}
                        ),
                        self._finalize_timeout,
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
                        self._scoring_timeout,
                    )
                trace.timing.scoring.end = time.time()
        except Exception as e:
            self.fail(e)
        finally:
            trace.is_completed = True
            trace.ok = not self._failed
            now = time.time()
            for span in (
                trace.timing.boot,
                trace.timing.setup,
                trace.timing.generation,
                trace.timing.finalize,
                trace.timing.scoring,
            ):
                if span.start and not span.end:
                    span.end = now
            trace.split_generation()
            if runtime is not None:
                try:
                    await self.harness.cleanup(trace, runtime)
                except Exception:
                    logger.warning(
                        "harness cleanup failed (rollout %s)", trace.id, exc_info=True
                    )
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
            trace.error.type if trace.error else trace.stop_condition,
        )
        return trace
