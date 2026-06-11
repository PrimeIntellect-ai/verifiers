"""A rollout: one trajectory — drive an harness in a runtime and score its trace.

A Rollout owns a single trajectory end-to-end, including its runtime's lifecycle. It
makes and starts the runtime, gets an interception endpoint (a slot on the shared pool if
one is given, else a per-rollout server exposed via its own runtime), runs the harness
against it under `harness_timeout`, runs the per-rollout `@reward`/`@metric` signals under
`scoring_timeout` while the runtime is still live, then tears the runtime down in a
`finally`. Cross-rollout `@group_reward`s run afterwards (in the Episode) over the traces
alone — they never need a live runtime — so a runtime is never kept up past its own
rollout. The runtime ref is set the instant it's created, so it's always tearable-down
even if `run()` crashes.
"""

import asyncio
import logging
import time
from contextlib import asynccontextmanager
from enum import StrEnum
from typing import TYPE_CHECKING

from verifiers.v1.harness import Harness
from verifiers.v1.clients import RolloutContext
from verifiers.v1.decorators import discover_decorated
from verifiers.v1.errors import ProgramError, RolloutError
from verifiers.v1.interception import InterceptionServer, RolloutLimits, RolloutSession
from verifiers.v1.runtimes import Runtime, RuntimeConfig, make_runtime
from verifiers.v1.task import Task
from verifiers.v1.taskset import Taskset
from verifiers.v1.tools import serve_tools
from verifiers.v1.trace import Trace
from verifiers.v1.user import serve_user

if TYPE_CHECKING:
    from verifiers.v1.interception import InterceptionPool

logger = logging.getLogger(__name__)


class Phase(StrEnum):
    """A rollout's lifecycle phase (for display): provisioning, the harness driving,
    per-rollout + group scoring, then fully scored."""

    SETUP = "setup"
    RUNNING = "running"
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
        harness_timeout: float | None = None,
        scoring_timeout: float | None = None,
        limits: RolloutLimits | None = None,
    ) -> None:
        self.task = task
        self.taskset = taskset
        self.harness = harness
        self.ctx = ctx
        self.runtime_config = runtime_config
        self.harness_timeout = harness_timeout
        self.scoring_timeout = scoring_timeout
        self.limits = limits or RolloutLimits()
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
        pool: "InterceptionPool | None",
        runtime: Runtime,
        session: RolloutSession,
    ):
        """Yield `(endpoint, secret)` for the harness — a slot on the shared `pool` if one
        is given, else a per-rollout server exposed via this rollout's own runtime."""
        if pool is not None:
            async with pool.acquire(session) as (endpoint, secret):
                yield endpoint, secret
        else:
            async with InterceptionServer() as server:
                secret = server.register(session)
                endpoint = f"{await runtime.expose(server.port)}/v1"
                yield endpoint, secret

    async def run(
        self,
        shared_urls: dict[str, str] | None = None,
        interception: "InterceptionPool | None" = None,
    ) -> Trace:
        """Run the rollout and return its trace. Captures expected `RolloutError`s onto
        the trace (a bad rollout is data, not a crash), runs per-rollout scoring while
        the runtime is live, then tears the runtime down in a `finally`. `shared_urls`
        are eval-level shared tool servers to reuse instead of starting; `interception`
        is the eval-level shared interception pool (None = a server per rollout)."""
        trace: Trace = Trace(task=self.task)
        self.trace = trace  # expose for the --rich dashboard
        trace.timing.generation.start = time.time()
        self.runtime = make_runtime(
            self.runtime_config, name=trace.id
        )  # ref set first → always tearable-down; named after the rollout for traceability
        runtime = self.runtime
        stops = discover_decorated(self.taskset, "stop")
        logger.info(
            "rollout start: id=%s task=%s harness=%s runtime=%s",
            trace.id,
            self.task.idx,
            self.harness.config.name,
            self.runtime_config.type,
        )
        try:
            session = RolloutSession(self.ctx, trace, stops, self.limits)
            await runtime.start()
            await self.taskset.setup(self.task, runtime)
            async with self._serve_interception(interception, runtime, session) as (
                endpoint,
                secret,
            ):
                tool_servers = self.taskset.tools(self.task)
                tools = self.taskset.config.tools
                async with (
                    serve_tools(
                        tool_servers,
                        runtime,
                        colocated=tools.colocated,
                        tool_runtime_config=tools.runtime,
                        shared_urls=shared_urls,
                    ) as urls,
                    serve_user(
                        self.taskset.user(self.task), self.taskset.config.user.runtime
                    ) as session.user,
                ):
                    self.phase = (
                        Phase.RUNNING
                    )  # setup done — the harness is now driving
                    try:
                        await asyncio.wait_for(
                            self.harness.run(
                                self.ctx, trace, runtime, endpoint, secret, urls
                            ),
                            self.harness_timeout,
                        )
                    except TimeoutError:
                        # A timeout is a budget limit, not a crash — score whatever the
                        # harness produced (like max_turns), don't error out. `is_truncated`
                        # is computed from this stop condition.
                        trace.stop("harness_timeout")
            trace.timing.generation.end = time.time()
            self.phase = (
                Phase.SCORING
            )  # per-rollout scoring; the Episode marks DONE after group scoring
            trace.timing.scoring.start = time.time()
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
            if not trace.timing.generation.end:  # error path: harness didn't finish
                trace.timing.generation.end = time.time()
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
