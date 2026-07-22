"""The Agent: a reusable (harness x model x runtime) value with one executable
arrow — `agent.run(task) -> Trace`; `runtime=` borrows a live box,
`provision(task)` hands you one. Inject a live `Interception` to share servers
across agents (a pool belongs to what spans agents, never to one agent); an
entered agent (`async with`) owns one server; un-entered, each run brings its own."""

import asyncio
import logging
from collections.abc import Callable, Iterator, Mapping
from contextlib import asynccontextmanager, nullcontext
from typing import AsyncIterator

from pydantic import SerializeAsAny, model_validator
from pydantic_config import BaseConfig

from verifiers.v1.clients import (
    Client,
    ClientConfig,
    EvalClientConfig,
    ModelContext,
    resolve_client,
)
from verifiers.v1.harness import HarnessConfig
from verifiers.v1.interception import Interception, InterceptionServer
from verifiers.v1.mcp import SharedToolServer
from verifiers.v1.retries import RetryConfig, backoff, trace_should_retry
from verifiers.v1.rollout import RolloutRun
from verifiers.v1.runtimes import (
    DockerConfig,
    Runtime,
    RuntimeConfig,
    SubprocessConfig,
    make_runtime,
    runtime_is_local,
)
from verifiers.v1.session import RolloutLimits
from verifiers.v1.task import Task
from verifiers.v1.trace import Trace
from verifiers.v1.types import Sampling, SamplingConfig
from verifiers.v1.utils.compile import (
    cap_remote_harness_timeout,
    resolve_runtime_config,
    validate_pairing,
)

logger = logging.getLogger(__name__)


class TimeoutConfig(BaseConfig):
    """Per-agent wall-clock timeouts per rollout stage, in seconds (None = no
    limit); each stage falls back to the task's own `TaskTimeout` when unset."""

    setup: float | None = None  # one shared budget: task setup + provisioning
    rollout: float | None = None
    finalize: float | None = None
    scoring: float | None = None


class AgentConfig(BaseConfig):
    """One env agent: who plays it, and its per-run caps. It pins only what
    makes it a different actor; everything unpinned falls back — the model context
    to the run's own, the harness to the taskset's default."""

    harness: SerializeAsAny[HarnessConfig] | None = None
    """The agent's program + runtime policy (None = the taskset's default harness)."""
    model: str | None = None
    """Model id (None = the run's model, i.e. the policy under evaluation/training)."""
    client: ClientConfig | None = None
    """Endpoint override (None = the run's client)."""
    sampling: SamplingConfig | None = None
    """Sampling override (None = the run's sampling)."""
    timeout: TimeoutConfig = TimeoutConfig()
    retries: RetryConfig = RetryConfig()
    """Whole-run retries: rerun this agent's rollout while its trace ends with a
    retryable error (never into a borrowed box)."""
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
        """Narrow a pinned `harness` to its concrete config type by `id` (absent
        stays None = the taskset's default). The lazy import keeps class-body
        `AgentConfig()` defaults constructible while this module initializes."""
        if isinstance(data, dict) and data.get("harness") is not None:
            from verifiers.v1.loaders import harness_config_type, narrow_plugin_field

            narrow_plugin_field(data, "harness", harness_config_type, "bash")
        return data


def _check_borrowed_placement(
    task: Task, runtime: Runtime, base_config: RuntimeConfig
) -> None:
    """A borrowed box is never re-provisioned, so a task's placement fields can't
    be honored. Reject requirements that cannot be applied to the running box; an
    image mismatch on a container only warns, since sharing its world is the point."""
    task_policy = "*" not in task.data.network_allow or bool(task.data.network_block)
    base_docker = base_config if isinstance(base_config, DockerConfig) else None
    if task_policy or (base_docker is not None and base_docker.network_isolated):
        config = runtime.config
        if not isinstance(config, DockerConfig):
            raise ValueError(
                f"task {task.data.idx!r} requires a Docker URL network policy, but "
                f"borrowed runtime {runtime.name!r} is not Docker-backed; use "
                "agent.provision(task)"
            )
        policy_base = (
            base_docker if base_docker is not None else DockerConfig(allow=["*"])
        )
        expected = resolve_runtime_config(policy_base, task)
        assert isinstance(expected, DockerConfig)
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


class Agent:
    """A configured harness + model + runtime policy, runnable on any task.

    Built from an `AgentConfig` alone; `client=`/`interception=` inject live
    resources to borrow — agents on one endpoint should share one `Client`, and a
    live `Interception`'s owner keeps its lifecycle. The harness config's
    `runtime` is a *policy*: each `run` provisions a fresh box from it, resolved
    per task; `run(runtime=...)` places the run into an existing box instead
    (borrowed boxes are never started or torn down by the run)."""

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
        self._owns_client = client is None
        if self._owns_client:
            client = resolve_client(config.client or EvalClientConfig())
        self.ctx = ModelContext(
            model=config.model,
            client=client,
            sampling=config.sampling if config.sampling is not None else Sampling(),
        )
        self._closed = False
        self.runtime_config: RuntimeConfig = self.harness.config.runtime
        self.interception = interception
        self.limits = RolloutLimits(
            max_turns=config.max_turns,
            max_input_tokens=config.max_input_tokens,
            max_output_tokens=config.max_output_tokens,
            max_total_tokens=config.max_total_tokens,
        )
        self.timeout = config.timeout
        # Env-owned standing, not config: `Env.setup` marks fixed agents
        # untrainable and traces are stamped from here; inert outside an env.
        self.trainable: bool = True
        self._entered = False
        self._server: InterceptionServer | None = None
        self._warned_resources: set[tuple[str, str]] = set()

    async def __aenter__(self) -> "Agent":
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
                if self._owns_client:
                    self._closed = True
                    await self.ctx.client.close()
                raise
        return self

    async def __aexit__(self, *exc) -> None:
        self._entered = False
        server, self._server = self._server, None
        try:
            if server is not None:
                await server.__aexit__(*exc)
        finally:
            if self._owns_client:
                self._closed = True
                await self.ctx.client.close()

    def _interception_for(
        self, run_is_local: bool, task: Task, shared_tools: Mapping
    ) -> Interception | None:
        """Which interception this run rides: an injected one always (its owner
        sized its reach); the owned server only when provably reachable from all
        the run's consumers — when it tunnels, else for a local run with no tool
        or user servers in play (such servers may sit in a remote runtime and must
        reach `/state`). Otherwise `None`: a per-run server sized to the task."""
        if self.interception is not None:
            return self.interception
        if self._server is None:
            return None
        if self._server.tunnel is not None or (
            run_is_local
            and not shared_tools
            and not type(task).tools
            and type(task).user is None
        ):
            return self._server
        return None

    async def run(
        self,
        task: Task,
        *,
        runtime: Runtime | None = None,
        tools: Mapping[str, SharedToolServer] | None = None,
        on_trace: Callable[[Trace], None] | None = None,
    ) -> Trace:
        """Run this agent on `task` once and return the trace: `runtime` places it
        into a live borrowed box instead of provisioning one; `tools` are live
        servers borrowed from their owner, counted in the pairing check;
        `on_trace` observes the trace the moment it's minted, before any I/O.
        Retries whole while the trace ends with a retryable error (`config.retries`)
        — never into a borrowed box; the final trace keeps earlier attempts' errors."""
        if self._closed:
            raise RuntimeError("Agent is closed; create a new agent")
        retry = self.config.retries
        history: list = []
        for attempt in range(retry.max_retries + 1):
            trace = await self._run_once(task, runtime, tools, on_trace)
            if attempt == retry.max_retries or not trace_should_retry(trace, retry):
                break
            if runtime is not None:
                logger.warning(
                    "not retrying the rollout on a borrowed box (its state is no "
                    "longer the task's start state); the error stands"
                )
                break
            history.extend(trace.errors)
            delay = backoff(attempt)
            logger.warning(
                "retrying agent rollout (retry %d/%d) in %.1fs after error: %s",
                attempt + 1,
                retry.max_retries,
                delay,
                trace.error.type if trace.error else "?",
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
    ) -> Trace:
        params = self._rollout_params(task, runtime, dict(shared_tools or {}))
        run = RolloutRun(task=task, on_trace=on_trace, **params)
        try:
            if await run.open():
                await run.step()
            trace = await run.close()
        except BaseException:
            # A cancellation mid-run (or a lifetime bug raised to the caller) means
            # close() never runs — free the run's servers and owned runtime first.
            await run.abort()
            raise
        if trace.runtime is not None:
            trace.runtime.borrowed = runtime is not None
        return trace

    def _rollout_params(
        self, task: Task, runtime: Runtime | None, shared_tools: dict
    ) -> dict:
        """Resolve one run's runtime config, pairing checks, timeouts, interception."""
        if runtime is not None:
            _check_borrowed_placement(task, runtime, self.runtime_config)
            runtime_config = runtime.config
            run_is_local = runtime.is_local
        else:
            runtime_config = resolve_runtime_config(
                self.runtime_config, task, self._warned_resources
            )
            run_is_local = runtime_is_local(runtime_config)
        validate_pairing(
            self.harness, type(task), runtime_config, shared_tools=shared_tools
        )
        # Timeout precedence: agent-level wins, else the task's, else no limit.
        harness_timeout = (
            self.timeout.rollout
            if self.timeout.rollout is not None
            else task.data.timeout.harness
        )
        return dict(
            harness=self.harness,
            ctx=self.ctx,
            runtime_config=runtime_config,
            setup_timeout=(
                self.timeout.setup
                if self.timeout.setup is not None
                else task.data.timeout.setup
            ),
            harness_timeout=cap_remote_harness_timeout(
                harness_timeout, runtime_config, task
            ),
            finalize_timeout=(
                self.timeout.finalize
                if self.timeout.finalize is not None
                else task.data.timeout.finalize
            ),
            scoring_timeout=(
                self.timeout.scoring
                if self.timeout.scoring is not None
                else task.data.timeout.scoring
            ),
            limits=self.limits,
            shared_tools=shared_tools,
            interception=self._interception_for(run_is_local, task, shared_tools),
            runtime=runtime,
        )

    @asynccontextmanager
    async def provision(self, task: Task | None = None) -> AsyncIterator[Runtime]:
        """Provision (and on exit tear down) a box from this agent's runtime
        policy, resolved for `task` when given; share it via `run(..., runtime=box)`."""
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


class _EpisodeAgent(Agent):
    """One role's `Agent` for one episode, built fresh each time (a cheap
    bundle of references — expensive resources are env-owned and borrowed, so no
    state spans concurrent episodes): traces get their agent standing the moment
    they're created, finished ones land in `completed` (the episode's traces),
    each run takes the eval's gate. The taskset's shared tool servers ride only
    its own tasks — on an env-minted task they'd wrongly put MCP in play
    (`tools=` overrides)."""

    def __init__(
        self,
        config: AgentConfig,
        *,
        client: Client,
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
        super().__init__(config, client=client, interception=interception)
        # Resource warnings dedupe env-wide, not per episode.
        self._warned_resources = warned_resources
        self._name = name
        self._shared_tools = shared_tools
        self._task_cls = task_cls
        self._gate = gate
        self._completed = completed
        self._on_trace = on_trace
        self._on_discard = on_discard

    def _shared_for(self, task: Task) -> Mapping[str, SharedToolServer]:
        return self._shared_tools if isinstance(task, self._task_cls) else {}

    async def run(
        self,
        task: Task,
        *,
        runtime: Runtime | None = None,
        tools: Mapping[str, SharedToolServer] | None = None,
        on_trace: Callable[[Trace], None] | None = None,
    ) -> Trace:
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

        async with self._gate or nullcontext():
            trace = await super().run(
                task,
                runtime=runtime,
                tools=tools if tools is not None else self._shared_for(task),
                on_trace=watch,
            )
        self._completed.append(trace)
        return trace


def make_agent(
    config: AgentConfig,
    *,
    client: Client | None = None,
    interception: Interception | None = None,
) -> Agent:
    """The agent for a config; `client`/`interception` inject live resources to
    borrow, everything else comes from the config."""
    return Agent(config, client=client, interception=interception)


MakeAgent = Callable[[str, AgentConfig], Agent]
"""An agent factory keyed by name — what `Agents` calls per scraped config field."""


def agent_config_fields(config) -> dict[str, AgentConfig]:
    """The top-level `AgentConfig` fields declared on a config, in declaration
    order — the env's agents, keyed by field name (the only naming site)."""
    return {name: value for name, value in config if isinstance(value, AgentConfig)}


class Agents:
    """A config's agents, addressed by attribute: every top-level `AgentConfig`
    field becomes an `Agent` under the field's name (`agents.solver`)."""

    def __init__(self, config, make: MakeAgent | None = None) -> None:
        if make is None:
            make = lambda _, spec: make_agent(spec)  # noqa: E731
        self._agents: dict[str, Agent] = {
            name: make(name, value)
            for name, value in agent_config_fields(config).items()
        }

    def __getattr__(self, name: str) -> Agent:
        # self.__dict__ directly: attribute lookup re-entering __getattr__ before
        # __init__ ran (copy/unpickle) must raise, not recurse.
        agents = self.__dict__.get("_agents")
        if agents is None or name not in agents:
            raise AttributeError(
                f"no agent {name!r}; this config declares "
                f"{sorted(agents) if agents else []}"
            )
        return agents[name]

    def __iter__(self) -> Iterator[Agent]:
        return iter(self._agents.values())

    def __len__(self) -> int:
        return len(self._agents)
