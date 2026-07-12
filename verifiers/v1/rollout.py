import asyncio
import logging
import time
from contextlib import asynccontextmanager
from enum import StrEnum

from verifiers.v1.clients import ModelContext
from verifiers.v1.decorators import discover_decorated, invoke
from verifiers.v1.errors import (
    HarnessError,
    RolloutError,
    TaskError,
    ToolsetError,
    boundary,
)
from verifiers.v1.harness import Harness
from verifiers.v1.interception import (
    InterceptionPool,
    InterceptionServer,
    RolloutLimits,
    RolloutSession,
)
from verifiers.v1.mcp import SharedToolServer, serve_tools, serve_user
from verifiers.v1.runtimes import (
    HOST,
    Runtime,
    RuntimeConfig,
    make_runtime,
    reachable_url,
)
from verifiers.v1.state import state_cls
from verifiers.v1.task import Task
from verifiers.v1.trace import Trace, TraceTask
from verifiers.v1.ttt import TTTConfig, TTTRolloutHook

logger = logging.getLogger(__name__)


class Phase(StrEnum):
    PENDING = "pending"
    SETUP = "setup"
    RUNNING = "running"
    FINALIZE = "finalize"
    SCORING = "scoring"
    DONE = "done"


class Rollout:
    def __init__(
        self,
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
        interception: InterceptionPool | None = None,
        ttt: TTTConfig | None = None,
    ) -> None:
        self.task = task
        self.harness = harness
        self.ctx = ctx
        self.runtime_config = runtime_config
        self.setup_timeout = setup_timeout
        self.harness_timeout = harness_timeout
        self.finalize_timeout = finalize_timeout
        self.scoring_timeout = scoring_timeout
        self.limits = limits or RolloutLimits()
        self.shared_tools = shared_tools or {}
        self.ttt = ttt
        self.interception = interception
        self.phase = Phase.PENDING
        self.runtime: Runtime | None = None
        self.trace: Trace | None = None

    @asynccontextmanager
    async def _serve_interception(
        self,
        pool: InterceptionPool | None,
        runtime: Runtime,
        session: RolloutSession,
    ):
        if pool is not None:
            async with pool.acquire(session) as (
                endpoint,
                secret,
                state_port,
                state_base,
            ):
                yield endpoint, secret, state_port, state_base
        else:
            async with InterceptionServer() as server:
                secret = server.register(session)
                # The runtime reaches this host service through localhost or a tunnel.
                async with reachable_url(HOST, server.port, consumer=runtime) as url:
                    yield f"{url}/v1", secret, server.port, url

    async def run(self) -> Trace:
        """Run the rollout and return its trace. Captures expected `RolloutError`s onto
        the trace (a bad rollout is data, not a crash), runs per-rollout scoring while
        the runtime is live, then tears the runtime down in a `finally`. Reuses the
        eval-level shared tool servers / interception pool injected at construction (see
        `self.shared_tools` / `self.interception`)."""
        # The trace carries the DATA (the wire half); behavior stays on `self.task`.
        trace: Trace = Trace(
            task=TraceTask(type=type(self.task).__name__, data=self.task.data),
            state=state_cls(type(self.task))(),
        )
        self.trace = trace  # expose for the --rich dashboard
        self.phase = Phase.SETUP  # leaving the queue: provisioning starts now
        trace.timing.setup.start = time.time()
        self.runtime = make_runtime(self.runtime_config, name=trace.id)
        runtime = self.runtime
        trace.runtime = runtime.info
        ctx = self.ctx
        stops = discover_decorated(self.task, "stop")
        logger.info(
            "rollout start: id=%s task=%s harness=%s runtime=%s",
            trace.id,
            self.task.data.idx,
            self.harness.config.name,
            self.runtime_config.type,
        )
        ttt_hook: TTTRolloutHook | None = None
        try:
            session = RolloutSession(ctx, trace, stops, self.limits)
            if self.ttt is not None and self.ttt.enabled:
                ttt_hook = TTTRolloutHook(self.ttt, trace, ctx=ctx)
                session.ttt = ttt_hook
            await runtime.start()
            # Task setup and harness provisioning share one setup-stage deadline.
            setup_deadline = (
                None if self.setup_timeout is None else asyncio.get_running_loop().time() + self.setup_timeout
            )
            async with (
                boundary(TaskError, "task setup"),
                asyncio.timeout_at(setup_deadline),
            ):
                await invoke(self.task.setup, {"trace": trace, "runtime": runtime})
            async with (
                boundary(HarnessError, "harness setup"),
                asyncio.timeout_at(setup_deadline),
            ):
                await self.harness.setup(runtime)
            async with self._serve_interception(self.interception, runtime, session) as (
                endpoint,
                secret,
                state_port,
                state_base,
            ):
                async with boundary(ToolsetError, "building tool servers"):
                    tool_servers = self.task.tool_servers()
                async with (
                    serve_tools(
                        tool_servers,
                        runtime,
                        shared=self.shared_tools,
                        state_port=state_port,
                        state_secret=secret,
                        state_base=state_base,
                    ) as urls,
                    serve_user(
                        self.task.user_server(),
                        harness_runtime=runtime,
                        state_port=state_port,
                        state_secret=secret,
                        state_base=state_base,
                    ) as session.user,
                ):
                    if self.task.data.prompt is None and session.user is None:
                        raise TaskError(
                            "task has no prompt and no user simulator to open the "
                            "conversation; set task.prompt or declare a simulator "
                            "class on Task.user"
                        )
                    now = time.time()
                    trace.timing.setup.end = now
                    trace.timing.generation.start = now
                    self.phase = Phase.RUNNING
                    # Prefer an intercepted model/tool/user error to the harness exit it caused.
                    # A timeout still scores the partial trajectory.
                    try:
                        await asyncio.wait_for(
                            self.harness.run(ctx, trace, runtime, endpoint, secret, urls),
                            self.harness_timeout,
                        )
                    except TimeoutError:
                        trace.stop("harness_timeout")
                    except RolloutError as e:
                        if session.error is not None:
                            raise session.error from e
                        raise
                    else:
                        if session.error is not None:
                            raise session.error
            if ttt_hook is not None:
                # Optional final-branch update (config); still inside the trace's generation
                # span, before taskset finalize. A failure is a TTTError → captured below.
                await ttt_hook.finalize_rollout()
            now = time.time()
            trace.timing.generation.end = now
            trace.timing.finalize.start = now
            self.phase = Phase.FINALIZE
            async with boundary(TaskError, "task finalize"):
                await asyncio.wait_for(
                    invoke(self.task.finalize, {"trace": trace, "runtime": runtime}),
                    self.finalize_timeout,
                )
            now = time.time()
            trace.timing.finalize.end = now
            self.phase = Phase.SCORING
            trace.timing.scoring.start = now
            async with boundary(TaskError, "scoring"):
                # Group rewards run later, after the runtime is gone.
                await asyncio.wait_for(
                    asyncio.gather(
                        self.task.score(trace, runtime),
                        self.harness.score(trace, runtime),
                    ),
                    self.scoring_timeout,
                )
            trace.timing.scoring.end = time.time()
        except RolloutError as e:
            trace.capture_error(e)
        except Exception as e:
            logger.exception("unexpected error in rollout %s", trace.id)
            trace.capture_error(e)
        finally:
            trace.is_completed = True
            now = time.time()
            for span in (
                trace.timing.setup,
                trace.timing.generation,
                trace.timing.finalize,
            ):
                if span.start and not span.end:
                    span.end = now
            # Release the rollout's TTT adapter (engine slot + service optimizer state);
            # its checkpoints stay on disk for RL replay. Never raises.
            if ttt_hook is not None:
                await ttt_hook.aclose()
            try:
                await runtime.stop()
            except Exception:
                logger.warning("runtime teardown failed (rollout %s)", trace.id, exc_info=True)
        logger.info(
            "rollout done: id=%s task=%s reward=%.3f turns=%d stop=%s",
            trace.id,
            self.task.data.idx,
            trace.reward,
            trace.num_turns,
            trace.error.type if trace.error else trace.stop_condition,
        )
        return trace
