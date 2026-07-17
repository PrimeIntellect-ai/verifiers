"""A rollout: one trajectory — run a harness program on a task and score its trace.

`RolloutRun` is the engine, a staged lifecycle held open across its stages: `open()`
boots the world (runtime, task + harness setup, interception slot, tool and user
servers), `step()` runs the harness program to its exit, and `close()` finalizes,
scores, and tears the world down — each stage under its own timeout
(`setup_timeout`/`harness_timeout`/`finalize_timeout`/`scoring_timeout`). It makes
and starts the runtime (unless handed a live one to borrow — then its creator owns
start and teardown). Cross-trace judgement runs afterwards (`Environment.score`,
over the finished sibling traces alone), so a runtime is never kept up past its own
rollout.

`Agent.run` is its only driver: an agent decides what goes into a rollout, this
module runs it. A task-declared user simulator (`Task.user`) rides the session
inside the model boundary — the interception server injects its replies between
model turns (see `verifiers.v1.mcp.user`).
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
from verifiers.v1.mcp import SharedToolServer, serve_tools, serve_user
from verifiers.v1.state import state_cls
from verifiers.v1.task import Task
from verifiers.v1.trace import AgentInfo, Trace, TraceTask, VersionInfo
from verifiers.v1.utils.version import verifiers_commit

logger = logging.getLogger(__name__)


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
    """One rollout held open across its stages.

    `open()` boots the world (runtime, setup, interception, tool and user servers);
    `step()` runs the harness program to its exit; `close()` finalizes, scores, and
    tears the world down, returning the finished trace. Expected `RolloutError`s
    are captured onto the trace (a bad rollout is data, not a crash): `open` and
    `step` report continuability as a bool, and `close` always returns the trace.

    `runtime` is a
    live box to run in instead of provisioning one; its creator owns teardown: a
    borrowed runtime is neither started nor stopped here. `on_trace` observes the
    run's trace the moment it's minted (before any I/O) — how a caller watches the
    run live (stage from the trace's timing spans, tokens and turns as the session
    records them; see `env.RunSlot`)."""

    def __init__(
        self,
        *,
        task: Task,
        harness: Harness,
        ctx: ModelContext,
        runtime_config: RuntimeConfig,
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
        self._setup_timeout = setup_timeout
        self._harness_timeout = harness_timeout
        self._finalize_timeout = finalize_timeout
        self._scoring_timeout = scoring_timeout
        self._shared_tools = shared_tools or {}
        self._interception = interception
        self.runtime = runtime
        self._owns_runtime = runtime is None
        # The trace carries the DATA (the wire half); behavior stays on the task.
        self.trace: Trace = Trace(
            task=TraceTask(type=type(task).__name__, data=task.data),
            state=state_cls(type(task))(),
            verifiers=VersionInfo(version=__version__, commit=verifiers_commit()),
            # The seat's resolved identity (role overrides included) — policy
            # metadata a training consumer reads off the trace.
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
        self._opened = False
        self._closed = False
        self._endpoint: str | None = None
        self._urls: dict[str, str] = {}
        self.deadline_at: float | None = None
        """The run's absolute deadline (event-loop clock) from `harness_timeout`,
        fixed when generation starts; None = unbounded."""

    @property
    def ok(self) -> bool:
        """Whether the run can continue: nothing failed, nothing stopped it."""
        return not self._failed and self.trace.stop_condition is None

    @property
    def closed(self) -> bool:
        return self._closed

    def fail(self, error: Exception) -> None:
        """Record `error` as this rollout's outcome (captured onto the trace, the
        remaining stages skipped)."""
        if not self._owns_runtime and self.runtime is not None and self.runtime.stopped:
            # The owner tore the borrowed box down mid-run — a lifetime bug in the
            # borrowing program, not a property of this rollout's world: raise to
            # the caller instead of capturing a misattributed error onto the trace.
            raise ValueError(
                f"borrowed runtime {self.runtime.name!r} was torn down by its owner "
                "mid-run; keep the provisioning context open until every run "
                "placed into the box has completed"
            ) from error
        if not isinstance(error, RolloutError):
            logger.exception("unexpected error in rollout %s", self.trace.id)
        self._failed = True
        self.trace.capture_error(error)

    async def open(self) -> bool:
        """Boot the rollout's world up to the point where the program can run:
        start (or borrow) the runtime, run task + harness setup, bring up the
        interception slot and the tool/user servers. Returns whether the run can
        proceed; a setup failure is captured onto the trace."""
        self._opened = True
        self.trace.timing.boot.start = time.time()
        if self._owns_runtime:
            # Named after the rollout for traceability.
            self.runtime = make_runtime(self.runtime_config, name=self.trace.id)
        elif self.runtime.stopped:
            # A lifetime bug in the borrowing program, not a property of this
            # rollout's world: raise to the caller instead of capturing onto the trace.
            raise ValueError(
                f"borrowed runtime {self.runtime.name!r} was already torn "
                "down by its owner; keep the provisioning context open for every run "
                "placed into the box"
            )
        runtime = self.runtime
        self.trace.runtime = runtime.info
        logger.info(
            "rollout start: id=%s task=%s harness=%s runtime=%s",
            self.trace.id,
            self.task.data.idx,
            self.harness.config.name,
            self.runtime_config.type,
        )
        try:
            if self._owns_runtime:
                await runtime.start()
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
            user = self.task.user_server()
            # `base_url` is the interception server's reachable URL for this rollout. The
            # harness reaches the model at `{base_url}/v1`; tool/user servers reach this
            # rollout's `/state` + `/task` at `base_url` — it's universally reachable
            # (the interception is exposed whenever any consumer is remote).
            base_url, secret = await self._stack.enter_async_context(
                _serve_interception(
                    self._interception,
                    runtime,
                    self._session,
                    [*tool_servers, *([user] if user else [])],
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
            self._session.user = await self._stack.enter_async_context(
                serve_user(
                    user,
                    harness_runtime=runtime,
                    state_secret=secret,
                    state_base=base_url,
                )
            )
            if self.task.data.prompt is None and self._session.user is None:
                raise TaskError(
                    "task has no prompt and no user simulator to open the "
                    "conversation; set task.prompt or declare a simulator "
                    "class on Task.user"
                )
        except Exception as e:
            self.fail(e)
            return False
        now = time.time()
        self.trace.timing.setup.end = now
        self.trace.timing.generation.start = now
        if self._harness_timeout is not None:
            self.deadline_at = asyncio.get_running_loop().time() + self._harness_timeout
        return True

    async def step(self) -> bool:
        """Run the harness program to completion — one launch on the task's prompt
        (a task-declared user simulator extends the run turn-by-turn inside the
        model boundary, transparently to the harness). Returns whether the run is
        still continuable — a stop (limit, `@stop`, or the program finishing
        cleanly), a timeout, or a failure all end it."""
        if not self._opened or self._closed or not self.ok:
            return False
        trace = self.trace
        # Prefer an intercepted model/tool/user error to the harness exit it caused.
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
                )
        except TimeoutError:
            trace.stop("harness_timeout")
            return False
        except Exception as e:
            real = self._session.error
            if real is not None and isinstance(e, RolloutError):
                real.__cause__ = e
                self.fail(real)
            else:
                self.fail(e)
            return False
        if self._session.error is not None:
            self.fail(self._session.error)
            return False
        return self.ok

    async def abort(self) -> None:
        """Free everything this run holds — the entered servers and an owned
        runtime — without finalizing or scoring: the escape path when an exception
        (a cancellation mid-setup, a lifetime bug raised to the caller) means the
        driver will never reach `close()`. Safe after a partial `close()`."""
        self._closed = True
        with contextlib.suppress(Exception):
            await self._stack.aclose()
        if self._owns_runtime and self.runtime is not None:
            with contextlib.suppress(Exception):
                await self.runtime.stop()

    async def close(self) -> Trace:
        """Finish the rollout: tool servers and interception down, task `finalize`
        and per-rollout scoring (skipped when the run already failed — but a
        stopped run is a complete one and scores its partial trajectory), then
        runtime teardown. Idempotent; always returns the trace."""
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
            now = time.time()
            for span in (
                trace.timing.boot,
                trace.timing.setup,
                trace.timing.generation,
                trace.timing.finalize,
            ):
                if span.start and not span.end:
                    span.end = now
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
