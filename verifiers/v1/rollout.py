"""A rollout: one trajectory — drive a harness in a runtime and score its trace.

`run_rollout` owns a single trajectory end-to-end, including its runtime's lifecycle. It
makes and starts the runtime (unless handed a live one to borrow — then its creator owns
start and teardown), gets an interception endpoint (a slot on the shared interception if
one is given, else a per-rollout server exposed via its own runtime), then drives the
staged lifecycle while the runtime is live — task + harness setup, the harness run,
task `finalize`, and per-rollout `@reward`/`@metric` scoring — each under its own stage
timeout (`setup_timeout`/`harness_timeout`/`finalize_timeout`/`scoring_timeout`),
then tears the runtime down in a `finally`. Cross-trace judgement runs afterwards
(`Environment.score`, over the finished sibling traces alone — it never needs a live
runtime), so a runtime is never kept up past its own rollout. `Agent.run` is the only
caller: an agent decides what goes into a rollout, this module runs it.
"""

import asyncio
import logging
import time
from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager

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
from verifiers.v1.session import Respond, RolloutLimits, RolloutSession
from verifiers.v1.runtimes import (
    Runtime,
    RuntimeConfig,
    make_runtime,
)
from verifiers.v1.mcp import SharedToolServer, serve_tools
from verifiers.v1.state import state_cls
from verifiers.v1.task import Task
from verifiers.v1.trace import TraceTask, Trace

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


async def run_rollout(
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
    user: Respond | None = None,
    on_trace: Callable[[Trace], None] | None = None,
) -> Trace:
    """Run one rollout and return its trace. Captures expected `RolloutError`s onto
    the trace (a bad rollout is data, not a crash), runs per-rollout scoring while
    the runtime is live, then tears the runtime down in a `finally`.

    `runtime` is a live box to run in instead of provisioning one (an agent program
    placing this run into an existing world — see `verifiers.v1.agent`); its creator
    owns teardown: a borrowed runtime is neither started nor stopped here. `user` is
    the run's user half (see `session.Respond`): the interception injects its replies
    as user turns after each tool-less model turn, and — for a task with no prompt —
    asks it to open the conversation. `on_trace` observes the run's trace the moment
    it's minted (before any I/O) — how a caller watches the run live (stage from the
    trace's timing spans, tokens and turns as the session records them; see
    `env.RunSlot`)."""
    limits = limits or RolloutLimits()
    shared_tools = shared_tools or {}
    # The trace carries the DATA (the wire half); behavior stays on the task.
    trace: Trace = Trace(
        task=TraceTask(type=type(task).__name__, data=task.data),
        state=state_cls(type(task))(),
        # The resolved sampling this run's calls are made with (role overrides
        # included) — policy metadata a training consumer reads off the trace.
        sampling=ctx.sampling,
    )
    if on_trace is not None:
        on_trace(trace)
    trace.timing.boot.start = time.time()
    owns_runtime = runtime is None
    if owns_runtime:
        # Named after the rollout for traceability.
        runtime = make_runtime(runtime_config, name=trace.id)
    elif runtime.stopped:
        # A lifetime bug in the borrowing program, not a property of this rollout's
        # world: raise to the caller instead of capturing onto the trace.
        raise ValueError(
            f"borrowed runtime {runtime.name!r} was already torn "
            "down by its owner; keep the provisioning context open for every run "
            "placed into the box"
        )
    trace.runtime = runtime.info
    stops = discover_decorated(task, "stop")
    logger.info(
        "rollout start: id=%s task=%s harness=%s runtime=%s",
        trace.id,
        task.data.idx,
        harness.config.name,
        runtime_config.type,
    )
    try:
        session = RolloutSession(ctx, trace, stops, limits, user)
        if task.data.prompt is None and user is None:
            raise TaskError(
                "task has no prompt and no user to open the conversation; set "
                "task.prompt, or pass user= to Agent.run (or drive the run through "
                "agent.chat())"
            )
        if owns_runtime:
            await runtime.start()
        now = time.time()
        trace.timing.boot.end = now
        trace.timing.setup.start = now
        # Task setup and harness provisioning share one setup-stage deadline.
        setup_deadline = (
            None
            if setup_timeout is None
            else asyncio.get_running_loop().time() + setup_timeout
        )
        async with (
            boundary(TaskError, "task setup"),
            asyncio.timeout_at(setup_deadline),
        ):
            await invoke(task.setup, {"trace": trace, "runtime": runtime})
        async with (
            boundary(HarnessError, "harness setup"),
            asyncio.timeout_at(setup_deadline),
        ):
            await harness.setup(runtime)
        async with boundary(ToolsetError, "building tool servers"):
            tool_servers = task.tool_servers()
        # `base_url` is the interception server's reachable URL for this rollout. The
        # harness reaches the model at `{base_url}/v1`; tool servers reach this
        # rollout's `/state` + `/task` at `base_url` — it's universally reachable (the
        # interception is exposed whenever any consumer is remote).
        async with _serve_interception(
            interception,
            runtime,
            session,
            tool_servers,
            shared_tools,
        ) as (base_url, secret):
            endpoint = f"{runtime.host_url(base_url)}/v1"
            async with serve_tools(
                tool_servers,
                runtime,
                shared=shared_tools,
                state_secret=secret,
                state_base=base_url,
            ) as urls:
                now = time.time()
                trace.timing.setup.end = now
                trace.timing.generation.start = now
                # Prefer an intercepted model/tool/user error to the harness exit it caused.
                # A timeout still scores the partial trajectory.
                try:
                    await asyncio.wait_for(
                        harness.run(ctx, trace, runtime, endpoint, secret, urls),
                        harness_timeout,
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
        async with boundary(TaskError, "task finalize"):
            await asyncio.wait_for(
                invoke(task.finalize, {"trace": trace, "runtime": runtime}),
                finalize_timeout,
            )
        now = time.time()
        trace.timing.finalize.end = now
        trace.timing.scoring.start = now
        async with boundary(TaskError, "scoring"):
            # Group rewards run later, after the runtime is gone.
            await asyncio.wait_for(
                asyncio.gather(
                    task.score(trace, runtime),
                    harness.score(trace, runtime),
                ),
                scoring_timeout,
            )
        trace.timing.scoring.end = time.time()
    except Exception as e:
        if not owns_runtime and runtime.stopped:
            # The owner tore the borrowed box down mid-run — the same lifetime bug
            # as borrowing a stopped runtime, surfaced through the same channel:
            # raise to the caller (raw failure chained) instead of capturing a
            # misattributed world error onto the trace.
            raise ValueError(
                f"borrowed runtime {runtime.name!r} was torn down by its owner "
                "mid-run; keep the provisioning context open until every run "
                "placed into the box has completed"
            ) from e
        if not isinstance(e, RolloutError):
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
        # Tear down here — group rewards (later) need only the trace, not a live
        # runtime. `runtime` is always set: make_runtime() ran before the `try`.
        # A borrowed runtime is its creator's to tear down, not this rollout's.
        if owns_runtime:
            try:
                await runtime.stop()
            except Exception:
                logger.warning(
                    "runtime teardown failed (rollout %s)", trace.id, exc_info=True
                )
    logger.info(
        "rollout done: id=%s task=%s reward=%.3f turns=%d stop=%s",
        trace.id,
        task.data.idx,
        trace.reward,
        trace.num_turns,
        trace.error.type if trace.error else trace.stop_condition,
    )
    return trace
