"""The Agent: a configured (harness x model x runtime policy) with one executable
arrow — `agent.run(task) -> Trace`. Built from its `AgentConfig` alone
(`make_agent(config)`); live resources (`client`, `interception`) are injected and
borrowed, never owned."""

import asyncio
import logging
from collections.abc import Callable, Iterator, Mapping
from contextlib import asynccontextmanager, nullcontext
from typing import AsyncIterator

from pydantic import BaseModel, SerializeAsAny, model_validator
from pydantic_config import BaseConfig

from verifiers.v1.clients import (
    Client,
    ClientConfig,
    EvalClientConfig,
    ModelContext,
    resolve_client,
)
from verifiers.v1.harness import HarnessConfig
from verifiers.v1.interception import Interception
from verifiers.v1.mcp import SharedToolServer
from verifiers.v1.rollout import RolloutRun
from verifiers.v1.runtimes import (
    Runtime,
    RuntimeConfig,
    SubprocessConfig,
    make_runtime,
)
from verifiers.v1.session import RolloutLimits
from verifiers.v1.task import Task
from verifiers.v1.trace import EpisodeInfo, Trace
from verifiers.v1.types import Sampling
from verifiers.v1.utils.compile import (
    cap_remote_harness_timeout,
    resolve_runtime_config,
    validate_pairing,
)

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
        minted."""
        params = self._rollout_params(task, runtime, dict(shared_tools or {}))
        run = RolloutRun(task=task, on_trace=on_trace, **params)
        try:
            if await run.open():
                await run.step()
            return await run.close()
        except BaseException:
            # A cancellation mid-run (or a lifetime bug raised to the caller) means
            # close() never runs — free the run's servers and owned runtime first.
            await run.abort()
            raise

    def _rollout_params(
        self, task: Task, runtime: Runtime | None, shared_tools: dict
    ) -> dict:
        """Resolve one run's execution parameters — runtime config, pairing checks,
        stage timeouts, interception."""
        if runtime is not None:
            _check_borrowed_placement(task, runtime)
            runtime_config = runtime.config
        else:
            runtime_config = resolve_runtime_config(
                self.runtime_config, task, self._warned_resources
            )
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
            interception=self.interception,
            runtime=runtime,
        )

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
    """The `AgentConfig` fields declared on a config, in declaration order — the
    env's agents, keyed by field name (the only naming site)."""
    return {name: value for name, value in config if isinstance(value, AgentConfig)}


def contains_agent_config(value) -> bool:
    """Whether `value` carries an `AgentConfig` anywhere `Agents` would scrape —
    directly, on a nested model, or inside a list/tuple."""
    if isinstance(value, AgentConfig):
        return True
    if isinstance(value, BaseModel):
        return any(contains_agent_config(v) for _, v in value)
    if isinstance(value, (list, tuple)):
        return any(contains_agent_config(v) for v in value)
    return False


class Agents:
    """A config's agents, addressed by attribute: every `AgentConfig` field
    becomes an `Agent` under the field's name (`agents.solver`), a list of
    `AgentConfig`s an index-addressable list, a nested model a nested `Agents`."""

    def __init__(self, config: BaseModel, make: MakeAgent = make_agent) -> None:
        self._agents: dict = {}
        for name, value in config:
            if isinstance(value, AgentConfig):
                self._agents[name] = make(name, value)
            elif isinstance(value, BaseModel) and contains_agent_config(value):
                self._agents[name] = Agents(value, make)
            elif isinstance(value, (list, tuple)) and contains_agent_config(value):
                self._agents[name] = [
                    make(f"{name}[{i}]", item)
                    for i, item in enumerate(value)
                    if isinstance(item, AgentConfig)
                ]

    def __getattr__(self, name: str):
        try:
            return self._agents[name]
        except KeyError:
            raise AttributeError(
                f"no agent {name!r}; this config declares {sorted(self._agents)}"
            ) from None

    def __iter__(self) -> Iterator[Agent]:
        """Every constructed agent, flattened."""
        for value in self._agents.values():
            if isinstance(value, Agent):
                yield value
            elif isinstance(value, Agents):
                yield from value
            else:
                yield from value

    def __len__(self) -> int:
        return sum(1 for _ in self)


class _EpisodeAgent(Agent):
    """One agent for one env-rollout, built fresh per episode: every trace is
    stamped at mint (agent name, trainability, the shared `EpisodeInfo`), finished
    traces land in `completed`, each run takes the eval's concurrency gate, and
    the taskset's shared tool servers ride only the taskset's own tasks (pass
    `shared_tools=` explicitly to override)."""

    def __init__(
        self,
        config: AgentConfig,
        *,
        client: Client,
        interception: Interception | None,
        name: str,
        episode: EpisodeInfo,
        shared_tools: Mapping[str, SharedToolServer],
        task_cls: type[Task],
        gate: asyncio.Semaphore | None,
        completed: list[Trace],
        on_trace: Callable[[Trace], None] | None,
        warned_resources: set,
    ) -> None:
        super().__init__(config, client=client, interception=interception)
        # Resource warnings dedupe env-wide, not per episode.
        self._warned_resources = warned_resources
        self._name = name
        self._episode = episode
        self._shared_tools = shared_tools
        self._task_cls = task_cls
        self._gate = gate
        self._completed = completed
        self._on_trace = on_trace

    def _shared_for(self, task: Task) -> Mapping[str, SharedToolServer]:
        return self._shared_tools if isinstance(task, self._task_cls) else {}

    async def run(
        self,
        task: Task,
        *,
        runtime: Runtime | None = None,
        shared_tools: Mapping[str, SharedToolServer] | None = None,
        on_trace: Callable[[Trace], None] | None = None,
    ) -> Trace:
        def watch(trace: Trace) -> None:
            # The trace's episode standing, self-contained on the trace: the shared
            # EpisodeInfo instance links it to its siblings.
            if trace.agent is not None:
                trace.agent.name = self._name
                trace.agent.trainable = self.trainable
            trace.episode = self._episode
            if self._on_trace is not None:
                self._on_trace(trace)
            if on_trace is not None:
                on_trace(trace)

        async with self._gate or nullcontext():
            trace = await super().run(
                task,
                runtime=runtime,
                shared_tools=shared_tools
                if shared_tools is not None
                else self._shared_for(task),
                on_trace=watch,
            )
        self._completed.append(trace)
        return trace
