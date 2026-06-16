"""A rollout: one trajectory — drive an harness in a runtime and score its trace.

A Rollout owns a single trajectory end-to-end, including its runtime's lifecycle. It
makes and starts the runtime, gets an interception endpoint (a slot on the shared pool if
one is given, else a per-rollout server exposed via its own runtime), then drives the
staged lifecycle while the runtime is live — the taskset's `setup`, the harness, the
taskset's `finalize`, and the per-rollout `@reward`/`@metric` scoring — each under its own
stage timeout (`setup_timeout`/`harness_timeout`/`finalize_timeout`/`scoring_timeout`),
then tears the runtime down in a `finally`. Cross-rollout `@group_reward`s run afterwards (in the Episode) over the traces
alone — they never need a live runtime — so a runtime is never kept up past its own
rollout. The runtime ref is set the instant it's created, so it's always tearable-down
even if `run()` crashes.
"""

import asyncio
import logging
import time
from contextlib import asynccontextmanager
from dataclasses import replace
from enum import StrEnum

from verifiers.v1.harness import Harness
from verifiers.v1.clients import RetryingClient, RolloutContext
from verifiers.v1.decorators import discover_decorated
from verifiers.v1.errors import ProgramError, RolloutError
from verifiers.v1.interception import (
    InterceptionPool,
    InterceptionServer,
    RolloutLimits,
    RolloutSession,
)
from verifiers.v1.runtimes import (
    HOST,
    RetryingRuntime,
    Runtime,
    RuntimeConfig,
    make_runtime,
    reachable_url,
)
from verifiers.v1.mcp import serve_tools, serve_user
from verifiers.v1.state import state_cls
from verifiers.v1.task import Task
from verifiers.v1.taskset import Taskset
from verifiers.v1.trace import Trace

logger = logging.getLogger(__name__)


class Phase(StrEnum):
    """A rollout's lifecycle phase (for display): provisioning, the harness driving,
    post-run finalize, per-rollout + group scoring, then fully scored."""

    SETUP = "setup"
    RUNNING = "running"
    FINALIZE = "finalize"
    SCORING = "scoring"
    DONE = "done"


class Rollout:
    def __init__(
        self,
        task: Task,
        taskset: Taskset,
        harness: Harness,
        ctx: RolloutContext,
        runtime_config: RuntimeConfig,
        setup_timeout: float | None = None,
        harness_timeout: float | None = None,
        finalize_timeout: float | None = None,
        scoring_timeout: float | None = None,
        limits: RolloutLimits | None = None,
        model_retries: int = 0,
        runtime_retries: int = 0,
        shared_urls: dict[str, str] | None = None,
        interception: InterceptionPool | None = None,
    ) -> None:
        self.task = task
        self.taskset = taskset
        self.harness = harness
        self.ctx = ctx
        self.runtime_config = runtime_config
        self.setup_timeout = setup_timeout
        self.harness_timeout = harness_timeout
        self.finalize_timeout = finalize_timeout
        self.scoring_timeout = scoring_timeout
        self.limits = limits or RolloutLimits()
        self.model_retries = model_retries
        self.runtime_retries = runtime_retries
        self.shared_urls = shared_urls or {}
        """Eval-level shared tool servers ({name: url}) to reuse instead of starting per rollout;
        the eval-level interception pool. Both injected by `Environment.episode` from the active
        `Environment.serving` context — so a rollout always has them and no runner has to thread
        them in."""
        self.interception = interception
        self.phase = Phase.SETUP
        """Lifecycle phase for display (see `Phase`); advanced through the rollout, and
        set to DONE by the Episode once group scoring has run."""
        self.runtime: Runtime | None = None
        """The runtime, set the moment `run()` creates it (so it's always tearable-down
        even if setup crashes) and torn down in `run()`'s `finally`; the --rich dashboard
        reads it for the runtime descriptor."""
        self.trace: Trace | None = None
        """The live trace, set the moment `run()` creates it; a --rich dashboard reads
        it to show in-flight progress (None until the rollout starts)."""

    @asynccontextmanager
    async def _serve_interception(
        self,
        pool: InterceptionPool | None,
        runtime: Runtime,
        session: RolloutSession,
    ):
        """Yield `(endpoint, secret, state_port)` for the harness — a slot on the shared `pool` if one
        is given, else a per-rollout server exposed via this rollout's own runtime. `state_port` is the
        interception server's host port, for the rollout's servers to reach its shared-state channel."""
        if pool is not None:
            async with pool.acquire(session) as (endpoint, secret, state_port):
                yield endpoint, secret, state_port
        else:
            async with InterceptionServer() as server:
                secret = server.register(session)
                # a HOST service the harness (in `runtime`) reaches: localhost or a tunnel
                async with reachable_url(HOST, server.port, consumer=runtime) as url:
                    yield f"{url}/v1", secret, server.port

    async def run(self) -> Trace:
        """Run the rollout and return its trace. Captures expected `RolloutError`s onto
        the trace (a bad rollout is data, not a crash), runs per-rollout scoring while
        the runtime is live, then tears the runtime down in a `finally`. Reuses the
        eval-level shared tool servers / interception pool injected at construction (see
        `self.shared_urls` / `self.interception`)."""
        trace: Trace = Trace(task=self.task, state=state_cls(type(self.taskset))())
        self.trace = trace  # expose for the --rich dashboard
        trace.timing.setup.start = time.time()
        self.runtime = make_runtime(
            self.runtime_config, name=trace.id
        )  # ref set first → always tearable-down; named after the rollout for traceability
        if self.runtime_retries > 0:
            self.runtime = RetryingRuntime(self.runtime, self.runtime_retries)
        runtime = self.runtime
        ctx = self.ctx
        if self.model_retries > 0:
            ctx = replace(ctx, client=RetryingClient(ctx.client, self.model_retries))
        stops = discover_decorated(self.taskset, "stop")
        logger.info(
            "rollout start: id=%s task=%s harness=%s runtime=%s",
            trace.id,
            self.task.idx,
            self.harness.config.name,
            self.runtime_config.type,
        )
        try:
            session = RolloutSession(ctx, trace, stops, self.limits)
            await runtime.start()
            try:
                await asyncio.wait_for(
                    self.taskset.setup(self.task, runtime), self.setup_timeout
                )
            except TimeoutError:
                raise ProgramError(
                    f"setup exceeded setup_timeout of {self.setup_timeout}s"
                ) from None
            async with self._serve_interception(
                self.interception, runtime, session
            ) as (
                endpoint,
                secret,
                state_port,
            ):
                tool_servers = self.taskset.tools(self.task)
                async with (
                    serve_tools(
                        tool_servers,
                        runtime,
                        self.task,
                        shared_urls=self.shared_urls,
                        state_port=state_port,
                        state_secret=secret,
                    ) as urls,
                    serve_user(
                        self.taskset.user(self.task),
                        self.task,
                        agent_runtime=runtime,
                        state_port=state_port,
                        state_secret=secret,
                    ) as session.user,
                ):
                    if self.task.instruction is None and session.user is None:
                        raise ProgramError(
                            "task has no instruction and no user simulator to open the "
                            "conversation; set task.instruction or have Taskset.user return "
                            "a simulator"
                        )
                    # setup done — the harness is now driving
                    now = time.time()
                    trace.timing.setup.end = now
                    trace.timing.generation.start = now
                    self.phase = Phase.RUNNING
                    try:
                        await asyncio.wait_for(
                            self.harness.run(
                                ctx, trace, runtime, endpoint, secret, urls
                            ),
                            self.harness_timeout,
                        )
                    except TimeoutError:
                        # A timeout is a budget limit, not a crash — score whatever the
                        # harness produced (like max_turns), don't error out. `is_truncated`
                        # is computed from this stop condition.
                        trace.stop("harness_timeout")
            now = time.time()
            trace.timing.generation.end = now
            trace.timing.finalize.start = now
            self.phase = Phase.FINALIZE  # post-run taskset work, before scoring
            try:
                await asyncio.wait_for(
                    self.taskset.finalize(self.task, trace, runtime),
                    self.finalize_timeout,
                )
            except TimeoutError:
                raise ProgramError(
                    f"finalize exceeded finalize_timeout of {self.finalize_timeout}s"
                ) from None
            now = time.time()
            trace.timing.finalize.end = now
            self.phase = (
                Phase.SCORING
            )  # per-rollout scoring; the Episode marks DONE after group scoring
            trace.timing.scoring.start = now
            try:
                # Per-rollout scoring: taskset + harness, concurrently, both with the live
                # runtime. (Cross-rollout `@group_reward`s run later, in the Episode.)
                await asyncio.wait_for(
                    asyncio.gather(
                        self.taskset.score(trace, runtime),
                        self.harness.score(trace, runtime),
                    ),
                    self.scoring_timeout,
                )
            except TimeoutError:
                raise ProgramError(
                    f"scoring exceeded scoring_timeout of {self.scoring_timeout}s"
                ) from None
            trace.timing.scoring.end = time.time()
        except RolloutError as e:
            trace.capture_error(e)
        finally:
            trace.is_completed = True
            now = time.time()
            if not trace.timing.setup.end:  # error during setup: close the setup span
                trace.timing.setup.end = now
            if trace.timing.generation.start and not trace.timing.generation.end:
                trace.timing.generation.end = now  # error mid-run: close generation
            if trace.timing.finalize.start and not trace.timing.finalize.end:
                trace.timing.finalize.end = now  # error mid-finalize: close finalize
            # Tear down here — group rewards (later) need only the trace, not a live
            # runtime. `runtime` is always set: make_runtime() ran before the `try`.
            try:
                await runtime.stop()
            except Exception:
                logger.warning(
                    "runtime teardown failed (rollout %s)", trace.id, exc_info=True
                )
        logger.info(
            "rollout done: id=%s task=%s reward=%.3f turns=%d stop=%s",
            trace.id,
            self.task.idx,
            trace.reward,
            trace.num_turns,
            trace.error.type if trace.error else trace.stop_condition,
        )
        return trace
