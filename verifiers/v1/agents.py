"""The Agent: a configured (harness x model x runtime policy) with one executable
arrow — `agent.run(task) -> Trace`. Built from its `AgentConfig` alone
(`make_agent(config)`); live resources (`client`, `interception`) are injected and
borrowed, never owned."""

import asyncio
import contextlib
import logging
import time
from collections.abc import AsyncIterator, Callable, Iterator, Mapping
from contextlib import AsyncExitStack, asynccontextmanager

from pydantic import BaseModel, SerializeAsAny, model_validator
from pydantic_config import BaseConfig

from verifiers import __version__
from verifiers.v1.clients import (
    Client,
    ClientConfig,
    EvalClientConfig,
    ModelContext,
    resolve_client,
)
from verifiers.v1.decorators import discover_decorated, invoke
from verifiers.v1.errors import (
    HarnessError,
    RolloutError,
    TaskError,
    ToolsetError,
    boundary,
)
from verifiers.v1.harness import HarnessConfig
from verifiers.v1.interception import (
    Interception,
    InterceptionServer,
    Slot,
    requires_tunnel,
)
from verifiers.v1.mcp import SharedToolServer, serve_tools, serve_user
from verifiers.v1.retries import RetryConfig, retryable
from verifiers.v1.runtimes import (
    Runtime,
    RuntimeConfig,
    SubprocessConfig,
    make_runtime,
)
from verifiers.v1.session import RolloutLimits, RolloutSession
from verifiers.v1.state import state_cls
from verifiers.v1.task import Task
from verifiers.v1.trace import AgentInfo, Trace, TraceTask, VersionInfo
from verifiers.v1.types import Sampling
from verifiers.v1.utils.compile import (
    cap_remote_harness_timeout,
    resolve_runtime_config,
    validate_pairing,
)
from verifiers.v1.utils.version import verifiers_commit

logger = logging.getLogger(__name__)


class TimeoutConfig(BaseConfig):
    """Per-run wall-clock timeouts by stage, in seconds (None = no limit); each
    stage falls back to the task's own `TaskTimeout` when unset."""

    setup: float | None = None
    rollout: float | None = None
    finalize: float | None = None
    scoring: float | None = None


class AgentConfig(BaseConfig):
    """One agent: who plays, and its per-run caps. As an env config field it pins
    only what makes it a different actor; everything unpinned falls back — the
    model context to the run's own, the harness to the taskset's default."""

    harness: SerializeAsAny[HarnessConfig] | None = None
    """The agent's program + runtime policy (None = the taskset's default harness,
    the built-in `bash` outside an env)."""
    model: str | None = None
    """Model id (None = the run's model, i.e. the policy under evaluation/training)."""
    client: ClientConfig | None = None
    """Endpoint override (None = the run's client)."""
    sampling: Sampling | None = None
    """Sampling override (None = the run's sampling)."""
    timeout: TimeoutConfig = TimeoutConfig()
    retries: RetryConfig = RetryConfig()
    """Whole-run retries: rerun this agent's rollout while its trace ends with a
    retryable error."""
    max_turns: int | None = None
    """Max model turns per run (None = no limit). Framework-enforced (the
    interception server refuses turns past it), so it applies to any harness."""
    max_input_tokens: int | None = None
    max_output_tokens: int | None = None
    max_total_tokens: int | None = None
    """Token caps per run (None = no limit); framework-enforced between turns."""

    @model_validator(mode="before")
    @classmethod
    def _resolve_harness(cls, data):
        # Narrow a pinned `harness` to its concrete config type by `id`. Lazy
        # import: class-body `AgentConfig()` defaults construct while this module
        # is still initializing.
        if isinstance(data, dict) and data.get("harness") is not None:
            from verifiers.v1.loaders import harness_config_type, narrow_plugin_field

            narrow_plugin_field(data, "harness", harness_config_type, "bash")
        return data


def _check_borrowed_placement(task: Task, runtime: Runtime) -> None:
    """A borrowed box is never re-provisioned, so a task's placement fields can't be
    honored. A task `image` on a subprocess box raises (a wiring bug in the
    borrowing program — it goes to the caller, not the trace); a container box whose
    image differs only warns, since placing a run into an existing world is the
    point of borrowing."""
    if task.data.image is None:
        return
    if isinstance(runtime.config, SubprocessConfig):
        raise ValueError(
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


class Agent:
    """A configured harness + model + runtime policy, runnable on any task.

    Built from an `AgentConfig` alone; `client=`/`interception=` inject live
    resources to borrow — agents on the same endpoint should share one `Client`
    (one connection pool), and without an interception each run brings up its own
    one-off server. The harness config's `runtime` is a *policy*: each `run`
    provisions a fresh box from it, resolved per task; `run(runtime=...)` places
    the run into an existing box instead (borrowed boxes are never started or torn
    down by the run)."""

    def __init__(
        self,
        config: AgentConfig,
        *,
        client: Client | None = None,
        interception: Interception | None = None,
    ) -> None:
        from verifiers.v1.loaders import harness_config_type, load_harness

        if config.model is None:
            raise ValueError(
                "AgentConfig.model is unset; an Agent needs a pinned model "
                "(inside an env the run's own model fills it in)"
            )
        harness_config = config.harness
        if harness_config is None:
            harness_config = harness_config_type("bash")(id="bash")
        self.config = config
        self.harness = load_harness(harness_config)
        if client is None:
            client = resolve_client(config.client or EvalClientConfig())
        self.ctx = ModelContext(
            model=config.model,
            client=client,
            sampling=config.sampling if config.sampling is not None else Sampling(),
        )
        self.runtime_config: RuntimeConfig = self.harness.config.runtime
        self.interception = interception
        self.limits = RolloutLimits(
            max_turns=config.max_turns,
            max_input_tokens=config.max_input_tokens,
            max_output_tokens=config.max_output_tokens,
            max_total_tokens=config.max_total_tokens,
        )
        self.timeout = config.timeout
        self.retries = config.retries
        # Env-owned standing, not config: `Env.setup` marks fixed agents
        # (a frozen grader) untrainable and their traces are stamped from here.
        # Inert in bespoke scripts — nothing outside an env reads it.
        self.trainable: bool = True
        self._warned_resources: set[tuple[str, str]] = set()

    async def run(
        self,
        task: Task,
        *,
        runtime: Runtime | None = None,
        shared_tools: Mapping[str, SharedToolServer] | None = None,
        on_trace: Callable[[Trace], None] | None = None,
    ) -> Trace:
        """Run this agent on `task` once and return the trace. The task carries
        its own judgement (a plain base `Task` makes the run unscored); `runtime`
        places the run into a live borrowed box; `shared_tools` are live servers
        borrowed from their owner; `on_trace` observes the trace the moment it's
        minted. The run retries whole while its trace ends with a retryable error
        (`config.retries`); when the final attempt fails too, the earlier
        attempts' errors are kept on its trace."""
        shared = dict(shared_tools or {})
        if runtime is not None:
            _check_borrowed_placement(task, runtime)
            runtime_config = runtime.config
        else:
            runtime_config = resolve_runtime_config(
                self.runtime_config, task, self._warned_resources
            )
        validate_pairing(self.harness, type(task), runtime_config, shared_tools=shared)
        # Timeout precedence: agent-level wins, else the task's, else no limit.
        timeouts = dict(
            setup=self.timeout.setup
            if self.timeout.setup is not None
            else task.data.timeout.setup,
            harness=cap_remote_harness_timeout(
                self.timeout.rollout
                if self.timeout.rollout is not None
                else task.data.timeout.harness,
                runtime_config,
                task,
            ),
            finalize=self.timeout.finalize
            if self.timeout.finalize is not None
            else task.data.timeout.finalize,
            scoring=self.timeout.scoring
            if self.timeout.scoring is not None
            else task.data.timeout.scoring,
        )
        history: list = []
        for attempt in range(self.retries.max_retries + 1):
            trace = await self._execute(
                task, runtime, runtime_config, shared, on_trace, timeouts
            )
            if attempt >= self.retries.max_retries or not any(
                retryable(e, self.retries) for e in trace.errors
            ):
                break
            logger.warning(
                "retrying rollout %s (retry %d/%d) after error: %s",
                trace.id,
                attempt + 1,
                self.retries.max_retries,
                trace.error.type if trace.error else "?",
            )
            history.extend(trace.errors)
        if history and trace.errors:
            # The final attempt failed too: keep the full history, oldest first.
            trace.errors = history + trace.errors
        return trace

    async def _execute(
        self,
        task: Task,
        borrowed: Runtime | None,
        runtime_config: RuntimeConfig,
        shared: dict[str, SharedToolServer],
        on_trace: Callable[[Trace], None] | None,
        timeouts: dict,
    ) -> Trace:
        """One rollout, staged: boot the world (runtime, setup, interception, tool
        and user servers), run the harness program to its exit, then finalize,
        score, and tear the world down. Expected `RolloutError`s are captured onto
        the trace (a bad rollout is data, not a crash); a cancellation or a
        lifetime bug frees the run's servers and owned runtime before re-raising."""
        trace: Trace = Trace(
            task=TraceTask(type=type(task).__name__, data=task.data),
            state=state_cls(type(task))(),
            verifiers=VersionInfo(version=__version__, commit=verifiers_commit()),
            agent=AgentInfo(
                model=self.ctx.model,
                sampling=self.ctx.sampling,
                harness=self.harness.config,
            ),
        )
        if on_trace is not None:
            on_trace(trace)
        session = RolloutSession(
            self.ctx, trace, discover_decorated(task, "stop"), self.limits
        )
        owns_runtime = borrowed is None
        if borrowed is not None and borrowed.stopped:
            # A lifetime bug in the borrowing program: raise to the caller instead
            # of capturing onto the trace.
            raise ValueError(
                f"borrowed runtime {borrowed.name!r} was already torn down by its "
                "owner; keep the provisioning context open for every run placed "
                "into the box"
            )
        runtime = (
            borrowed
            if borrowed is not None
            else make_runtime(runtime_config, name=trace.id)
        )
        if owns_runtime:
            # The shared object: start() below fills the provisioned id in place.
            trace.runtime = runtime.info
        else:
            # A borrowed box is already provisioned and its info object is shared
            # across the runs placed in it — stamp `borrowed` on a copy.
            trace.runtime = runtime.info.model_copy(update={"borrowed": True})
        stack = AsyncExitStack()
        failed = False

        def fail(error: Exception) -> None:
            """Record `error` as this rollout's outcome; the remaining stages skip."""
            nonlocal failed
            if not owns_runtime and runtime.stopped:
                # The owner tore the borrowed box down mid-run — a lifetime bug in
                # the borrowing program: raise to the caller instead of capturing a
                # misattributed error onto the trace.
                raise ValueError(
                    f"borrowed runtime {runtime.name!r} was torn down by its owner "
                    "mid-run; keep the provisioning context open until every run "
                    "placed into the box has completed"
                ) from error
            if not isinstance(error, RolloutError):
                logger.exception("unexpected error in rollout %s", trace.id)
            failed = True
            trace.capture_error(error)

        logger.info(
            "rollout start: id=%s task=%s harness=%s runtime=%s",
            trace.id,
            task.data.idx,
            self.harness.config.name,
            runtime_config.type,
        )
        try:
            # --- boot the world -------------------------------------------------
            deadline_at: float | None = None
            try:
                trace.timing.boot.start = time.time()
                if owns_runtime:
                    await runtime.start()
                now = time.time()
                trace.timing.boot.end = now
                trace.timing.setup.start = now
                # Task setup and harness provisioning share one setup-stage deadline.
                setup_deadline = (
                    None
                    if timeouts["setup"] is None
                    else asyncio.get_running_loop().time() + timeouts["setup"]
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
                    await self.harness.setup(runtime)
                async with boundary(ToolsetError, "building tool servers"):
                    tool_servers = task.tool_servers()
                user = task.user_server()
                # `base_url` is the interception server's URL for this rollout: the
                # harness reaches the model at `{base_url}/v1`, tool/user servers
                # reach `/state` + `/task` there.
                base_url, secret = await stack.enter_async_context(
                    _serve_interception(
                        self.interception,
                        runtime,
                        session,
                        [*tool_servers, *([user] if user else [])],
                        shared,
                    )
                )
                endpoint = f"{runtime.host_url(base_url)}/v1"
                urls = await stack.enter_async_context(
                    serve_tools(
                        tool_servers,
                        runtime,
                        shared=shared,
                        state_secret=secret,
                        state_base=base_url,
                    )
                )
                session.user = await stack.enter_async_context(
                    serve_user(
                        user,
                        harness_runtime=runtime,
                        state_secret=secret,
                        state_base=base_url,
                    )
                )
                if task.data.prompt is None and session.user is None:
                    raise TaskError(
                        "task has no prompt and no user simulator to open the "
                        "conversation; set task.prompt or declare a simulator "
                        "class on Task.user"
                    )
            except Exception as e:
                fail(e)
            else:
                # --- run the program to its exit --------------------------------
                now = time.time()
                trace.timing.setup.end = now
                trace.timing.generation.start = now
                if timeouts["harness"] is not None:
                    deadline_at = (
                        asyncio.get_running_loop().time() + timeouts["harness"]
                    )
                if trace.stop_condition is None:
                    # Prefer an intercepted model/tool/user error to the harness
                    # exit it caused; a timeout still scores the partial trajectory.
                    try:
                        async with asyncio.timeout_at(deadline_at):
                            await self.harness.run(
                                self.ctx, trace, runtime, endpoint, secret, urls
                            )
                    except TimeoutError:
                        trace.stop("harness_timeout")
                    except Exception as e:
                        real = session.error
                        if real is not None and isinstance(e, RolloutError):
                            real.__cause__ = e
                            fail(real)
                        else:
                            fail(e)
                    else:
                        if session.error is not None:
                            fail(session.error)
            # --- finalize, score ------------------------------------------------
            try:
                try:
                    await stack.aclose()
                finally:
                    if (
                        trace.timing.generation.start
                        and not trace.timing.generation.end
                    ):
                        trace.timing.generation.end = time.time()
                if not failed:
                    trace.timing.finalize.start = time.time()
                    async with boundary(TaskError, "task finalize"):
                        await asyncio.wait_for(
                            invoke(task.finalize, {"trace": trace, "runtime": runtime}),
                            timeouts["finalize"],
                        )
                    now = time.time()
                    trace.timing.finalize.end = now
                    trace.timing.scoring.start = now
                    async with boundary(TaskError, "scoring"):
                        # Cross-trace judgement (the env's finalize) runs later,
                        # after the runtime is gone.
                        await asyncio.wait_for(
                            asyncio.gather(
                                task.score(trace, runtime),
                                self.harness.score(trace, runtime),
                            ),
                            timeouts["scoring"],
                        )
                    trace.timing.scoring.end = time.time()
            except Exception as e:
                fail(e)
        except BaseException:
            # A cancellation (or a lifetime bug raised to the caller) means the
            # completion stamping below never runs — free the entered servers and
            # an owned runtime first.
            with contextlib.suppress(Exception):
                await stack.aclose()
            if owns_runtime:
                with contextlib.suppress(Exception):
                    await runtime.stop()
            raise
        trace.is_completed = True
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
        # Tear down here — the env's `finalize()` (later) needs only the traces,
        # not a live runtime. A borrowed runtime is its creator's to tear down,
        # not this rollout's.
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
            trace.stop_condition,
        )
        return trace

    @asynccontextmanager
    async def provision(self, task: Task | None = None) -> AsyncIterator[Runtime]:
        """Provision (and on exit tear down) a box from this agent's runtime policy,
        resolved for `task` when given. Place runs into it via `run(..., runtime=box)`:
        the provisioning program owns the box, so several runs can share one world."""
        config = (
            resolve_runtime_config(self.runtime_config, task, self._warned_resources)
            if task is not None
            else self.runtime_config
        )
        runtime = make_runtime(config)
        try:
            # start() inside the try: a failed start may already hold a remote
            # sandbox, so it must reach stop() (safe on a partially-started runtime).
            await runtime.start()
            yield runtime
        finally:
            await runtime.stop()


def make_agent(
    config: AgentConfig,
    *,
    client: Client | None = None,
    interception: Interception | None = None,
) -> Agent:
    """The agent for a config (the counterpart to `make_runtime`/`make_interception`).
    `client`/`interception` inject live resources to borrow; everything else — the
    harness, the model, the caps — comes from the config."""
    return Agent(config, client=client, interception=interception)


MakeAgent = Callable[[str, AgentConfig], Agent]
"""An agent factory keyed by name — what `Agents` calls per scraped config field."""


def agent_config_fields(config: BaseModel) -> dict[str, AgentConfig]:
    """The top-level `AgentConfig` fields declared on a config, in declaration
    order — the env's agents, keyed by field name (the only naming site)."""
    return {name: value for name, value in config if isinstance(value, AgentConfig)}


class Agents:
    """A config's agents, addressed by attribute: every top-level `AgentConfig`
    field becomes an `Agent` under the field's name (`agents.solver`)."""

    def __init__(self, config: BaseModel, make: MakeAgent = make_agent) -> None:
        self._agents: dict[str, Agent] = {
            name: make(name, value)
            for name, value in agent_config_fields(config).items()
        }

    def __getattr__(self, name: str) -> Agent:
        try:
            return self._agents[name]
        except KeyError:
            raise AttributeError(
                f"no agent {name!r}; this config declares {sorted(self._agents)}"
            ) from None

    def __iter__(self) -> Iterator[Agent]:
        return iter(self._agents.values())

    def __len__(self) -> int:
        return len(self._agents)
