import asyncio
import logging
import time
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from enum import StrEnum

from verifiers.v1.harness import Harness
from verifiers.v1.clients import ModelContext
from verifiers.v1.decorators import discover_decorated, invoke
from verifiers.v1.errors import (
    HarnessError,
    RolloutError,
    TaskError,
    ToolsetError,
    UserError,
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
from verifiers.v1.trace import TraceTask, Trace

logger = logging.getLogger(__name__)


class Phase(StrEnum):
    PENDING = "pending"
    BOOT = "boot"
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
        interception: Interception | None = None,
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
        self.interception = interception
        self.phase = Phase.PENDING
        self.runtime: Runtime | None = None
        self.trace: Trace | None = None

    @asynccontextmanager
    async def _serve_interception(
        self,
        runtime: Runtime,
        session: RolloutSession,
        servers: list,
    ) -> AsyncIterator[Slot]:
        """A slot on the shared interception when one was injected (its owner keeps the
        lifecycle), else on a per-rollout `InterceptionServer` owned — brought up and torn
        down — by this rollout."""
        if self.interception is not None:
            async with self.interception.acquire(session) as slot:
                yield slot
            return
        tunneled = requires_tunnel(
            runtime.reaches_host_locally,
            [server.config for server in servers],
            self.shared_tools.values(),
        )
        server = InterceptionServer(requires_tunnel=tunneled)
        async with server:
            async with server.acquire(session) as slot:
                yield slot

    async def run(self) -> Trace:
        """Run the rollout and return its trace. Captures expected `RolloutError`s onto
        the trace (a bad rollout is data, not a crash), runs per-rollout scoring while
        the runtime is live, then tears the runtime down in a `finally`. Reuses the
        eval-level shared tool servers / interception injected at construction (see
        `self.shared_tools` / `self.interception`)."""
        # The trace carries the DATA (the wire half); behavior stays on `self.task`.
        trace: Trace = Trace(
            task=TraceTask(type=type(self.task).__name__, data=self.task.data),
            state=state_cls(type(self.task))(),
        )
        self.trace = trace  # expose for the --rich dashboard
        self.phase = Phase.BOOT  # leaving the queue: the runtime boots now
        trace.timing.boot.start = time.time()
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
        try:
            session = RolloutSession(ctx, trace, stops, self.limits)
            await runtime.start()
            now = time.time()
            trace.timing.boot.end = now
            trace.timing.setup.start = now
            self.phase = Phase.SETUP
            # Task setup and harness provisioning share one setup-stage deadline.
            setup_deadline = (
                None
                if self.setup_timeout is None
                else asyncio.get_running_loop().time() + self.setup_timeout
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
            async with boundary(ToolsetError, "building tool servers"):
                tool_servers = self.task.tool_servers()
                if not runtime.supports_colocated_tools and any(
                    server.config.colocated and not server.config.url
                    for server in tool_servers
                ):
                    raise ToolsetError(
                        "this harness network policy does not support colocated MCP servers"
                    )
            user = self.task.user_server()
            if (
                not runtime.supports_colocated_user
                and user is not None
                and user.config.colocated
            ):
                raise UserError(
                    "this harness network policy does not support a colocated user server"
                )
            # `base_url` is the interception server's reachable URL for this rollout. The
            # harness reaches the model at `{base_url}/v1`; tool/user servers reach this
            # rollout's `/state` + `/task` at `base_url` — it's universally reachable (the
            # interception is exposed whenever any consumer is remote).
            async with self._serve_interception(
                runtime, session, [*tool_servers, *([user] if user else [])]
            ) as (base_url, secret):
                endpoint = f"{base_url}/v1"
                async with (
                    serve_tools(
                        tool_servers,
                        runtime,
                        shared=self.shared_tools,
                        state_secret=secret,
                        state_base=base_url,
                    ) as urls,
                    serve_user(
                        user,
                        harness_runtime=runtime,
                        state_secret=secret,
                        state_base=base_url,
                    ) as session.user,
                ):
                    if self.task.data.prompt is None and session.user is None:
                        raise TaskError(
                            "task has no prompt and no user simulator to open the "
                            "conversation; set task.prompt or declare a simulator "
                            "class on Task.user"
                        )
                    routes = await runtime.apply_network_policy(
                        {
                            "model": endpoint,
                            **{f"mcp:{name}": url for name, url in urls.items()},
                        }
                    )
                    endpoint = routes["model"]
                    urls = {name: routes[f"mcp:{name}"] for name in urls}
                    now = time.time()
                    trace.timing.setup.end = now
                    trace.timing.generation.start = now
                    self.phase = Phase.RUNNING
                    # Prefer an intercepted model/tool/user error to the harness exit it caused.
                    # A timeout still scores the partial trajectory.
                    try:
                        await asyncio.wait_for(
                            self.harness.run(
                                ctx, trace, runtime, endpoint, secret, urls
                            ),
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
                trace.timing.boot,
                trace.timing.setup,
                trace.timing.generation,
                trace.timing.finalize,
            ):
                if span.start and not span.end:
                    span.end = now
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
