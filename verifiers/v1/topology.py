"""The topology: a surface over which agents interact.

A `Topology` composes agent runs — one agent consuming one task and producing one trace —
into a multi-agent interaction: which agents exist, how one agent's trace becomes a
downstream agent's task, and how rewards flow backwards once downstream agents have run.
Each agent run is an ordinary `Rollout` (same lifecycle, same error model, same trace), and
tasks are the whole task-side contract — their classes carry the behavior — so a topology
declares named agent bindings, then executes topology-bound agents against tasks.

The interaction pattern is plain imperative Python in `go` — not a DSL: a loop is rounds,
`asyncio.gather` over `run.agent(...).run(...)` calls is fan-out, and awaiting several traces
before building the next task is fan-in. `go` owns *control flow only*, including the
forward arrow (trace → next task, pure host-side construction); judgement — including the
backward arrow, a reward derived from downstream traces — is declared as
`@vf.reward(agent=...)`/`@vf.metric(agent=...)` methods, run over the finished instance
(see `Topology.score`).

Running one instance produces an `AgentGraph` — the serialized instance artifact: the
global, causally ordered view over its traces, each linked to its parents (`trace.agent` /
`trace.parents`). A `Trace` stays the per-agent view of one run; the graph is the
cross-agent view of the whole interaction. Interleaving two agents' *execution* inside one
run is deliberately out of scope — an agent run completes before its trace
feeds anything downstream.
"""

import asyncio
import contextlib
import logging
import uuid
from collections.abc import Callable, Iterable, Mapping, Sequence
from functools import cached_property
from typing import Generic, TypeVar

from pydantic import Field, SerializeAsAny, model_validator
from pydantic_config import BaseConfig

from verifiers.v1.agent import Agent, Parent, Session
from verifiers.v1.clients import Client, ClientConfig, ModelContext, resolve_client
from verifiers.v1.decorators import discover_decorated, invoke
from verifiers.v1.env import (
    EnvConfig,
    EnvServerConfig,
    taskset_server_configs,
    validate_pairing,
)
from verifiers.v1.errors import TopologyError, boundary
from verifiers.v1.harness import Harness, HarnessConfig
from verifiers.v1.interception import (
    Interception,
    make_interception,
    requires_tunnel,
)
from verifiers.v1.rollout import Rollout
from verifiers.v1.runtimes import Runtime, SubprocessConfig
from verifiers.v1.session import RolloutLimits
from verifiers.v1.task import Task
from verifiers.v1.taskset import Taskset, TasksetConfig
from verifiers.v1.trace import Error, Trace, TraceTask, WireTrace
from verifiers.v1.types import ID, SamplingConfig, StrictBaseModel
from verifiers.v1.utils.install import env_name
from verifiers.v1.utils.memory import trim_memory_periodically
from verifiers.v1.utils.sampling import sample

logger = logging.getLogger(__name__)


def _deep_merge(base: dict, override: dict) -> dict:
    """`override` layered onto `base`, recursing into nested dicts — how a partial
    `--topology.<agent>.harness.*` override tunes a pinned harness instead of replacing it."""
    merged = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def _merge_sampling(
    base: SamplingConfig, override: SamplingConfig | None
) -> SamplingConfig:
    """Agent sampling overrides layer onto the run sampling, field by field."""
    if override is None:
        return base
    return base.model_copy(update=override.model_dump(exclude_none=True))


class AgentConfig(BaseConfig):
    """One agent in a topology: which harness drives its runs (and where —
    `harness.runtime`), and how its model calls are routed. Declared as typed fields of a
    `TopologyConfig` subclass — the field *name* is the agent's name in `go` — so every
    agent is CLI/toml-addressable (`--topology.<agent>.harness.id`,
    `--topology.<agent>.model`, ...). An agent carries nothing task-side: the tasks it
    consumes (each carrying its own behavior) arrive per run, from the topology's seeds
    or constructed in `go`.

    To pin a per-agent default harness, **subclass and set the field default**
    (`harness: SerializeAsAny[HarnessConfig] = HarnessConfig(id="direct")`) — pins must
    live on an `AgentConfig` subclass, never on the outer topology-config field's default
    instance, which a partial override (`--topology.<agent>.model ...`) would silently
    replace (pydantic re-validates the whole field; it never merges into instance
    defaults). A pinned harness survives everything except an explicit
    `--topology.<agent>.harness.id`."""

    harness: SerializeAsAny[HarnessConfig] = HarnessConfig(id="default")
    """The program driving this agent's runs, and where it runs (`harness.runtime`) —
    per agent, so a judge can run the in-process `direct` chat loop while the solver's
    coding agent runs in a container."""
    model: str | None = None
    """Model override for this agent (None = the eval's model) — e.g. a stronger judge."""
    client: ClientConfig | None = None
    """Client override for this agent (None = the eval's client) — how its model calls are
    routed: e.g. a non-trainable judge relayed to a plain API endpoint while the solver
    runs against the train client. Per-agent routing lives here, not in extra interception
    machinery — sessions are already per-rollout."""
    sampling: SamplingConfig | None = None
    """Sampling overrides for this agent, layered over the eval's sampling."""
    trainable: bool = True
    """Whether this agent's traces are training samples — stamped onto `Trace.trainable`
    so a trainer can filter without consulting the topology config."""

    @model_validator(mode="before")
    @classmethod
    def _resolve_plugins(cls, data):
        """Resolve the `harness` field. Three cases, in precedence order:
        an explicit `id` swaps the harness (narrowed to that harness's own config type);
        otherwise a subclass **pin** — detected by value, so any changed default counts,
        including a base-typed `HarnessConfig(id="null")` — absorbs partial overrides by
        deep-merging them over the pinned default (`--...harness.runtime.type docker`
        tunes the pin, never silently replaces it); otherwise the base default narrows to
        the `default` harness's config."""
        if not isinstance(data, dict):
            return data
        from verifiers.v1.loaders import harness_config_type, narrow_plugin_field

        raw = data.get("harness")
        if isinstance(raw, BaseConfig):
            raw = raw.model_dump()
        raw = dict(raw or {})
        default = cls.model_fields["harness"].default
        pinned = (
            isinstance(default, HarnessConfig)
            and default != AgentConfig.model_fields["harness"].default
        )
        if raw.get("id"):  # explicit swap always wins
            narrow_plugin_field(data, "harness", harness_config_type)
        elif pinned:
            data["harness"] = harness_config_type(default.id).model_validate(
                _deep_merge(default.model_dump(), raw)
            )
        else:
            narrow_plugin_field(data, "harness", harness_config_type, "default")
        return data


class DirectAgentConfig(AgentConfig):
    """An agent pinned to the in-process `direct` chat loop — tool-less, a run ≈ one
    API call. The most common pin, shared so topologies don't each redeclare it; per the
    pin contract, partial overrides (`--topology.<agent>.model ...`) tune it and an
    explicit `--topology.<agent>.harness.id` still swaps it."""

    harness: SerializeAsAny[HarnessConfig] = HarnessConfig(id="direct")


class NullAgentConfig(AgentConfig):
    """An agent pinned to the `null` chat loop — a subprocess program WITH the task's MCP
    tools but none of its own. The pin for an agent that must call task tools without
    being a full coding agent."""

    harness: SerializeAsAny[HarnessConfig] = HarnessConfig(id="null")


class TopologyConfig(BaseConfig):
    """Base topology config. Subclass to declare the agents (typed `AgentConfig` fields —
    the field name is the agent's name) plus any interaction knobs (fan-out width, number
    of rounds, reward weights). Mirrors `TasksetConfig`: the concrete subclass is resolved
    by `id`, so its fields surface typed on the CLI/toml."""

    id: ID = ""
    """The topology id, which selects this topology: a built-in (`llm-judge`), a local
    package, or an `org/name[@version]` package installed on demand from the Environments
    Hub (see `ID`). Set via `--topology.id`."""
    taskset: SerializeAsAny[TasksetConfig] = TasksetConfig()
    """The seed source: a taskset (a pure task factory, resolved by id — see
    `verifiers.v1.taskset`) whose tasks seed the instances, one instance per seed. Set via
    `--topology.taskset.id <id>` — the same word, slot, and grammar as the single-agent
    route's `--taskset.id`. Exclusive-or with overriding `load_tasks`: when this slot can
    be set, it IS the seed source, verbatim — a topology that constructs its own seeds
    overrides `load_tasks` and is refused this flag (a custom `load_tasks` wanting a
    config-driven source declares its own factory field instead)."""

    @property
    def name(self) -> str:
        """The topology's package name (the id with any org / version stripped)."""
        return env_name(self.id)

    @model_validator(mode="before")
    @classmethod
    def _resolve_taskset(cls, data):
        """Narrow the seed `taskset` factory to the config type its `id` resolves to, so its
        knobs (`--topology.taskset.split ...`) validate typed."""
        if isinstance(data, dict) and data.get("taskset"):
            from verifiers.v1.loaders import narrow_plugin_field, taskset_config_type

            narrow_plugin_field(data, "taskset", taskset_config_type)
        return data


class SingleAgentTopologyConfig(TopologyConfig):
    """Internal lowering target for the user-facing taskset + harness syntax."""

    id: ID = "single-agent"
    agent: AgentConfig = AgentConfig()


ConfigT = TypeVar("ConfigT", bound=TopologyConfig)


class AgentBinding:
    """A topology-registered agent slot: name + config + loaded harness.

    The value exposed inside `Topology.go` is a topology-bound `TopologyAgent`; this
    binding is the config-side declaration a topology loads and validates before serving.
    """

    def __init__(
        self, name: str, config: AgentConfig, harness: Harness | None = None
    ) -> None:
        from verifiers.v1.loaders import load_harness

        self.name = name
        self.config = config
        self.harness = harness if harness is not None else load_harness(config.harness)


class AgentGraph(StrictBaseModel):
    """One topology instance's artifact: the global, causally ordered view over its traces,
    each linked to the traces it was derived from (`trace.agent` names the producing agent,
    `trace.parents` its upstream trace ids). Traces append in completion order — a parent
    always finishes before a task is derived from it, so the list is topologically sorted.

    The trace's sibling, one level up: what a trace is to an agent run, the graph is to an
    instance. A topology run persists one graph per line (`traces.jsonl`), traces nested —
    and since each trace carries its own links, the graph is also *recoverable* from a flat
    trace dump (one instance = one connected component)."""

    id: str = Field(default_factory=lambda: uuid.uuid4().hex)
    """Unique id for this instance, auto-generated per graph."""
    topology: str = ""
    """The topology id that produced this instance."""
    task: SerializeAsAny[TraceTask]
    """The seed task for this invocation, including its task behavior type."""
    error: Error | None = None
    """A failure in the topology's own code (`go` or instance scoring, a `TopologyError`),
    recorded instead of raised — agent failures live on their traces, this is for the
    composition itself. Traces completed before the failure remain data."""
    traces: list[SerializeAsAny[Trace]] = Field(default_factory=list)
    """Every agent trace, in completion (= topological) order."""

    def add(self, trace: Trace) -> None:
        self.traces.append(trace)

    def roots(self) -> list[Trace]:
        """The traces with no parents — the entry runs (usually one, on the seed task)."""
        return [trace for trace in self.traces if not trace.parents]

    def children(self, trace: Trace, agent: str | None = None) -> list[Trace]:
        """The traces derived from `trace` (its direct downstream runs), optionally
        only those a named agent produced — the navigation cross-agent scoring lives on
        (`graph.children(proposer, agent="solver")`)."""
        return [
            t
            for t in self.traces
            if trace.id in t.parents and (agent is None or t.agent == agent)
        ]

    def by_agent(self, agent: str) -> list[Trace]:
        """The traces a named agent produced, in completion order."""
        return [trace for trace in self.traces if trace.agent == agent]

    def to_record(self) -> dict:
        """A JSON-serializable record of this instance for `traces.jsonl` — each nested
        trace dumped like `Trace.to_record` (per-node training tensors stripped)."""
        from verifiers.v1.trace import _NODE_DUMP_EXCLUDE

        return self.model_dump(
            mode="json", exclude={"traces": {"__all__": _NODE_DUMP_EXCLUDE}}
        )

    @classmethod
    def load(cls, data: dict) -> "AgentGraph":
        """Load a dumped instance record without the originating packages: each trace is
        typed as a `WireTrace` (task-specific fields ride in `task.data.model_extra`)."""
        from verifiers.v1.task import WireTaskData

        task = TraceTask[WireTaskData].model_validate(data["task"])
        graph = cls.model_validate({**data, "task": task, "traces": []})
        graph.traces = [WireTrace.model_validate(t) for t in data.get("traces", [])]
        return graph


def graph_complete(graph: AgentGraph) -> bool:
    """The conservative instance-validity default (`Topology.complete`'s base rule):
    no instance-level error and no errored trace. Module-level so consumers without a
    live topology (the server-mode eval client) apply the same rule."""
    return graph.error is None and not any(t.has_error for t in graph.traces)


class Topology(Generic[ConfigT]):
    """Generic over its config type, so `self.config` is fully typed in subclasses.
    Subclass: declare `AgentConfig` fields on your config, implement `go` (control flow)
    and `@vf.reward(agent=...)`/`@vf.metric(agent=...)` methods (judgement). Seeds come
    from the config's `taskset` slot, or override `load_tasks` to construct them."""

    def __init__(self, config: ConfigT) -> None:
        self.config = config

    @cached_property
    def taskset(self) -> Taskset | None:
        """The configured seed taskset, loaded once, or None for self-seeding topologies."""
        if not self.config.taskset.id:
            return None
        from verifiers.v1.loaders import load_taskset

        return load_taskset(self.config.taskset)

    def load_agents(self) -> dict[str, AgentBinding]:
        """The topology's agents, one per `AgentConfig` field on the config (in declaration
        order). Override only to compose agents programmatically — each is still just
        `AgentBinding(name, config)`."""
        return {
            name: AgentBinding(name, value)
            for name, value in self.config
            if isinstance(value, AgentConfig)
        }

    @cached_property
    def agents(self) -> dict[str, AgentBinding]:
        """The loaded agents, built once via `load_agents`. Loading also validates the
        topology's declared judgement (`@reward`/`@metric` methods) against them, so a
        typo'd or missing agent scope fails at load time, not mid-eval."""
        agents = self.load_agents()
        if not agents:
            raise ValueError(
                f"topology {self.config.id!r} declares no agents: give its config "
                "`AgentConfig` fields (or override `load_agents`)"
            )
        # Scan the class, not `discover_decorated` (whose getmembers would re-enter this
        # very cached_property via the properties it evaluates). Unwrap descriptors so a
        # classmethod/staticmethod-wrapped decorator can't slip past validation only to be
        # found (and silently mis-scoped) by `score`'s getmembers discovery later.
        for klass in type(self).__mro__:
            for fn in vars(klass).values():
                fn = getattr(fn, "__func__", fn)
                kind = next(
                    (k for k in ("reward", "metric", "stop") if getattr(fn, k, False)),
                    None,
                )
                if kind is None:
                    continue
                if kind == "stop":
                    raise ValueError(
                        f"@stop is not supported on a Topology (found {fn.__name__!r}): stops belong to Task classes"
                    )
                scope = getattr(fn, "_vf_agent", None)
                if scope is None:
                    raise ValueError(
                        f"topology @{kind} {fn.__name__!r} declares no agent scope; "
                        f"topology judgement is per-agent — use @vf.{kind}(agent=...)"
                    )
                if scope not in agents:
                    raise ValueError(
                        f"topology @{kind} {fn.__name__!r} scopes to unknown agent "
                        f"{scope!r}; this topology defines {sorted(agents)}"
                    )
        return agents

    def load_tasks(self) -> Iterable[Task]:
        """The seed tasks — one topology instance (`go`) runs per seed task. Defaults to
        the config's `taskset` slot (`--topology.taskset.id <id>`), whose `load` may be
        lazy or infinite (see `Taskset`); override for a topology that constructs its own
        seeds (a finite, materializable iterable)."""
        if not self.config.taskset.id:
            raise ValueError(
                f"topology {self.config.id!r} has no seed tasks: set --topology.taskset.id "
                "<id>, or override `load_tasks` to construct them"
            )
        assert self.taskset is not None
        return self.taskset.load()

    @property
    def infinite(self) -> bool:
        """Whether the seed source never ends (an `INFINITE` taskset in the config slot) —
        such runs must be bounded with `num_tasks`. A custom `load_tasks` override is
        finite by contract."""
        if type(self).load_tasks is not Topology.load_tasks:
            return False
        return self.taskset is not None and type(self.taskset).INFINITE

    def complete(self, graph: AgentGraph) -> bool:
        """Whether a persisted instance counts as a valid result of this topology — the
        verdict consumers read when deciding what to redo or drop (today: `--resume`
        re-runs instances that fail it). The default is conservative — no instance-level
        error and no errored trace — which is exact for the single-agent lowering, where
        the one trace IS the invocation. A topology whose `go` tolerates child failures
        overrides this to match (typically `graph.error is None`), else resume redoes
        instances it already accepted and scored. A read-only verdict over a finished
        graph: what a failed child *means* stays in `go` and the declared rewards."""
        return graph_complete(graph)

    async def score(self, graph: AgentGraph) -> None:
        """Run the topology's declared judgement over one completed instance: every
        `@metric(agent=...)`, then every `@reward(agent=...)`, each invoked once per
        trace the named agent produced — declaring any of `task`/`trace`/`graph` by
        parameter name. Runs after `go` returns and before the instance persists, so
        every trace (across all rounds and fan-outs) is scored, automatically.

        Ordering contract, chosen for predictability over cleverness: methods run
        *sequentially*, metrics before rewards, each phase in (priority, name) order.
        A method may read task-recorded rewards (final since the agent run ended) and,
        in the rewards phase, any metric — but topology rewards must not read each
        other; derive shared inputs from the traces (or a metric) instead. Mirrors
        `Task.score` one level up: trace judgement there, instance judgement here."""
        for kind in ("metric", "reward"):
            for fn in discover_decorated(self, kind):
                weight = getattr(fn, "_vf_weight", 1.0)
                for trace in graph.by_agent(getattr(fn, "_vf_agent", None)):
                    available = {
                        "task": trace.task.data,  # the wire half, as in `Task.score`
                        "trace": trace,
                        "graph": graph,
                    }
                    result = await invoke(fn, available)
                    if kind == "metric":
                        if isinstance(result, Mapping):
                            trace.record_metrics(result)
                        else:
                            trace.record_metric(fn.__name__, result)
                    elif isinstance(result, Mapping):
                        for name, value in result.items():
                            trace.record_reward(name, value, weight)
                    else:
                        trace.record_reward(fn.__name__, result, weight)

    async def go(self, task: Task, run: "TopologyRun") -> None:
        """Run one topology instance from seed `task`: the *control flow only* — which
        agents run, in what order, with what tasks:
        `await run.agent(name).run(task, parents=...)` per agent invocation,
        `asyncio.gather` for fan-out, loops for rounds, and
        `trace.info[...] = ...` to annotate provenance. The forward arrow lives here —
        construct the next agent's typed `Task` from an upstream trace (its typed task,
        `last_reply`, `transcript`, or what its `finalize` peeled into `trace.info`).
        Judgement belongs in the topology's `@reward(agent=...)`/`@metric(agent=...)`
        methods (see `score`); recording a reward imperatively here is the escape hatch,
        not the norm. Agent failures come back as data on their traces (`trace.error`),
        never as exceptions — `go` decides what a failed child means (drop it, count it
        against a pass rate, retry the round, ...)."""
        raise NotImplementedError


class SingleAgentTopology(Topology[SingleAgentTopologyConfig]):
    """The canonical one-agent topology produced from taskset + harness syntax."""

    async def go(self, task: Task, run: "TopologyRun") -> None:
        await run.agent("agent").run(task)


def resolve_topology_runner(config: EnvConfig) -> "TopologyRunner":
    """Resolve explicit topology or taskset + harness syntax to the canonical runner."""
    if config.topology is not None:
        from verifiers.v1.loaders import load_topology

        topology = load_topology(config.topology)
    else:
        topology = SingleAgentTopology(
            SingleAgentTopologyConfig(
                taskset=config.taskset,
                agent=AgentConfig(harness=config.harness),
            )
        )
    return TopologyRunner(topology, config)


class TopologyAgent:
    """A topology-bound view of a registered executable `Agent`.

    It exposes the public agent's `run` / `provision` surface, but routes completed
    traces through the owning `TopologyRun` so the graph, parents, trainability, retries,
    and concurrency limit remain topology-owned.
    """

    def __init__(self, run: "TopologyRun", name: str) -> None:
        self._run = run
        self.name = name

    def _executable(self) -> Agent:
        return self._run.executable_agent(self.name)

    @property
    def config(self) -> AgentConfig:
        return self._run.runner.agent(self.name).config

    @property
    def harness(self) -> Harness:
        return self._executable().harness

    def provision(
        self, task: Task | None = None
    ) -> contextlib.AbstractAsyncContextManager[Runtime]:
        """Provision a runtime from this agent's policy (resolved for `task` when given)
        and tear it down on exit — the box for `run(..., runtime=box)` calls to share,
        by this agent or any other (see `Agent.provision`)."""
        return self._executable().provision(task)

    async def run(
        self,
        task: Task,
        *,
        parents: Sequence[Parent] = (),
        runtime: Runtime | None = None,
    ) -> Trace:
        return await self._run.run_agent(
            self.name, task, parents=parents, runtime=runtime
        )

    def interact(
        self,
        task: Task,
        *,
        parents: Sequence[Parent] = (),
        runtime: Runtime | None = None,
    ) -> contextlib.AbstractAsyncContextManager[Session]:
        """Hold a live agent run open and yield its `Session` — the
        back-and-forth primitive (see `Agent.interact`): `go` converses with the
        suspended run via `session.turn(...)`, N sessions compose into games,
        debates, negotiations. The completed trace is graph-recorded on close."""
        return self._run.interact_agent(
            self.name, task, parents=parents, runtime=runtime
        )


class TopologyRun:
    """One live topology instance: the execution surface `go` programs against. Owns the
    instance's `AgentGraph` and links every agent trace into it; concurrency, retries, and
    limits come from the eval config, identically to a single-agent eval."""

    def __init__(
        self,
        runner: "TopologyRunner",
        task: Task,
        agents: dict[str, Agent],
        semaphore: asyncio.Semaphore | None = None,
    ) -> None:
        self.runner = runner
        self.graph = AgentGraph(
            topology=runner.topology.config.id,
            task=TraceTask(type=type(task).__name__, data=task.data),
        )
        self._agents = agents
        self._semaphore = semaphore

    def agent(self, name: str) -> TopologyAgent:
        """The registered agent named `name`, bound to this graph."""
        self.runner.agent(name)  # validate spelling with the existing actionable error
        return TopologyAgent(self, name)

    def executable_agent(self, name: str) -> Agent:
        try:
            return self._agents[name]
        except KeyError:
            self.runner.agent(name)
            raise RuntimeError(
                f"agent {name!r} is not bound to this topology run"
            ) from None

    async def run_agent(
        self,
        agent: str,
        task: Task,
        *,
        parents: Sequence[Parent] = (),
        runtime: Runtime | None = None,
    ) -> Trace:
        """Run a topology-bound explicit agent and record its trace in this graph."""
        executable = self.executable_agent(agent)
        async with self._semaphore or contextlib.nullcontext():
            trace = await executable.run(
                task,
                parents=parents,
                runtime=runtime,
                retry=self.runner.config.retries.rollout,
            )
        self.graph.add(trace)
        await trim_memory_periodically()
        return trace

    @contextlib.asynccontextmanager
    async def interact_agent(
        self,
        agent: str,
        task: Task,
        *,
        parents: Sequence[Parent] = (),
        runtime: Runtime | None = None,
    ):
        """Hold a live run of a topology-bound agent open (see `Agent.interact`) and
        record its completed trace in this graph on close — even when `go` crashed
        mid-interaction (completed traces remain data; `run_instance` captures the
        crash as the graph's `TopologyError`).

        Sessions deliberately bypass the rollout-concurrency semaphore: a seat spends
        most of an interaction suspended (no model call in flight), and a multi-seat
        game holding N slots for its whole duration could deadlock the cap. Instance
        concurrency (the eval's `-c`) is what bounds session pressure."""
        executable = self.executable_agent(agent)
        session: Session | None = None
        try:
            async with executable.interact(
                task, parents=parents, runtime=runtime
            ) as session:
                yield session
        finally:
            if session is not None:
                try:
                    trace = session.trace
                except RuntimeError:  # the run never started — nothing to record
                    trace = None
                if trace is not None:
                    self.graph.add(trace)
                    await trim_memory_periodically()


class TopologyRunner:
    """Long-lived executor for one topology and its worker-scoped services."""

    def __init__(self, topology: Topology | TopologyConfig, config: EnvConfig) -> None:
        if isinstance(topology, TopologyConfig):
            from verifiers.v1.loaders import load_topology

            topology = load_topology(topology)
        self.config = config
        self.topology = topology
        # The seed contract is exclusive-or: the `taskset` slot IS the seed source, verbatim,
        # or `load_tasks` is overridden — never both. A self-seeding topology accepting a
        # `--topology.taskset.id` it then ignores would silently run a different experiment
        # than the config claims, so refuse it up front.
        if topology.config.taskset.id and (
            type(self.topology).load_tasks is not Topology.load_tasks
        ):
            raise ValueError(
                f"topology {topology.config.id!r} constructs its own seeds (it overrides "
                "`load_tasks`), so `--topology.taskset.id` would be silently ignored; drop "
                "it. A topology wanting a config-driven source *inside* a custom "
                "`load_tasks` declares its own factory field, not the built-in `taskset` slot."
            )
        bindings = self.topology.agents
        if isinstance(topology, SingleAgentTopology):
            assert topology.taskset is not None
            validate_pairing(bindings["agent"].harness, topology.taskset)
        for binding in bindings.values():
            if binding.harness.config.id != "null" and isinstance(
                binding.harness.config.runtime, SubprocessConfig
            ):
                logger.warning(
                    "Harness %r is running in the subprocess runtime on the local system. "
                    "Local files and settings may affect the evaluation; use subprocess only "
                    "for debugging. Use --harness.runtime.type docker or prime for an isolated run.",
                    binding.harness.config.id,
                )
        self._interception: Interception | None = None
        self._shared_tools: dict = {}
        self._override_clients: dict[str, Client] = {}

    @property
    def infinite(self) -> bool:
        """Whether this topology's seed source is infinite (see `Topology.infinite`)."""
        return self.topology.infinite

    def load_tasks(self) -> "Iterable[Task]":
        """The topology's seed tasks, possibly lazy/infinite (see `Topology.load_tasks`)."""
        return self.topology.load_tasks()

    def select_tasks(
        self, num_tasks: int | None = None, shuffle: bool = False
    ) -> list[Task]:
        """Materialize the seeds a run needs. When the config's `taskset` slot is the
        seed source this is `Taskset.select` (lazy/infinite-aware — an infinite taskset
        requires `num_tasks`); a topology's own `load_tasks` override is finite, so it
        materializes and samples the same way (`verifiers.v1.utils.sampling`)."""
        topology = self.topology
        if (
            type(topology).load_tasks is Topology.load_tasks
            and topology.taskset is not None
        ):
            return topology.taskset.select(num_tasks, shuffle)
        return sample(list(topology.load_tasks()), shuffle, num_tasks)

    def agent(self, name: str) -> AgentBinding:
        """The named agent, with an actionable error for a typo in `go`."""
        try:
            return self.topology.agents[name]
        except KeyError:
            raise ValueError(
                f"unknown agent {name!r}: topology {self.topology.config.id!r} defines {sorted(self.topology.agents)}"
            ) from None

    @contextlib.asynccontextmanager
    async def serving(self):
        """Hold worker-scoped services while request contexts remain invocation-local."""
        if self._interception is not None:
            raise RuntimeError("TopologyRunner.serving() is already active")
        from verifiers.v1.mcp import serve_shared
        from verifiers.v1.runtimes import runtime_is_local

        async with contextlib.AsyncExitStack() as stack:
            bindings = self.topology.agents
            # Every agent's harness placement — any remote seat must reach the shared
            # servers and the interception through a tunnel.
            local = all(
                runtime_is_local(b.harness.config.runtime) for b in bindings.values()
            )
            shared_tools: dict = {}
            if self.topology.taskset is not None:
                servers = self.topology.taskset.tool_servers()
                if servers:
                    shared_tools = await stack.enter_async_context(
                        serve_shared(servers, harness_is_local=local)
                    )
            # One interception serves the whole run, tunneled whenever any statically
            # knowable consumer — an agent's harness, a live shared server, or a seed
            # task's tool/user server — is off the host network. Tasks minted in `go`
            # aren't knowable here; like the pairing checks, the seed taskset stands in
            # for them.
            seed_configs = (
                taskset_server_configs(self.topology.taskset)
                if self.topology.taskset is not None
                else []
            )
            tunneled = seed_configs is None or requires_tunnel(
                local, seed_configs, shared_tools.values()
            )
            interception = await stack.enter_async_context(
                make_interception(self.config.interception, requires_tunnel=tunneled)
            )
            override_clients: dict[str, Client] = {}
            for name, binding in bindings.items():
                if binding.config.client is not None:
                    client = resolve_client(binding.config.client)
                    stack.push_async_callback(client.close)
                    override_clients[name] = client
            self._interception = interception
            self._shared_tools = shared_tools
            self._override_clients = override_clients
            try:
                yield
            finally:
                self._interception = None
                self._shared_tools = {}
                self._override_clients = {}

    def _agents_for(
        self,
        ctx: ModelContext,
        on_rollout: Callable[[Rollout], None] | None = None,
    ) -> dict[str, Agent]:
        interception = self._interception
        if interception is None:
            raise RuntimeError(
                "TopologyRunner.run_instance() must be called inside TopologyRunner.serving()"
            )
        limits = RolloutLimits(
            max_turns=self.config.max_turns,
            max_input_tokens=self.config.max_input_tokens,
            max_output_tokens=self.config.max_output_tokens,
            max_total_tokens=self.config.max_total_tokens,
        )
        agents: dict[str, Agent] = {}
        for name, binding in self.topology.agents.items():
            agent_ctx = ModelContext(
                model=binding.config.model or ctx.model,
                client=self._override_clients.get(name, ctx.client),
                sampling=_merge_sampling(ctx.sampling, binding.config.sampling),
            )
            agents[name] = Agent(
                binding.harness,
                agent_ctx,
                binding.harness.config.runtime,
                name=name,
                trainable=binding.config.trainable,
                limits=limits,
                timeout=self.config.timeout,
                interception=interception,
                shared_tools=self._shared_tools,
                on_rollout=on_rollout,
            )
        return agents

    async def run_instance(
        self,
        task: Task,
        ctx: ModelContext,
        semaphore: asyncio.Semaphore | None = None,
        on_rollout: Callable[[Rollout], None] | None = None,
    ) -> AgentGraph:
        """Run one topology instance — a single `go` over one seed task, then the declared
        instance judgement (`Topology.score`) — and return its agent graph. A failure in
        topology-authored code is classified (`TopologyError`) and captured on the graph,
        never raised: completed traces remain data, and sibling instances keep
        running (mirrors `Rollout.run`'s a-bad-rollout-is-data stance one level up)."""
        run = TopologyRun(self, task, self._agents_for(ctx, on_rollout), semaphore)
        try:
            async with boundary(
                TopologyError, f"topology {self.topology.config.id!r} go"
            ):
                await self.topology.go(task, run)
            # Instance judgement: the declared @reward/@metric methods over the finished
            # graph. Skipped when `go` itself failed — a broken instance isn't scored.
            async with boundary(
                TopologyError, f"topology {self.topology.config.id!r} scoring"
            ):
                await self.topology.score(run.graph)
        except TopologyError as e:
            logger.exception("topology instance failed (seed task %s)", task.data.idx)
            run.graph.error = Error(type=type(e).__name__, message=str(e))
        return run.graph


# `EnvConfig.topology` is annotated as a forward reference — env.py sits *below* this
# module and can't import it. Finalize the models here, where `TopologyConfig` exists;
# anything importing `verifiers.v1` (or any submodule — the package init runs first and
# imports this module) sees the completed models.
EnvConfig.model_rebuild()
EnvServerConfig.model_rebuild()
