"""The topology: a surface over which agents interact.

A `Topology` composes agent runs into a multi-agent interaction: which agents exist, how
one agent's trace becomes a downstream agent's task (the forward arrow), and how rewards
flow backwards. It declares its agents as config fields; `run(task, agents)` is the
control flow — plain imperative Python (a loop is rounds, `asyncio.gather` is fan-out,
awaiting several traces before the next task is fan-in), no DSL. Judgement, including the
backward arrow, is declared separately as `@vf.reward(agent=...)`/`@vf.metric(agent=...)`
methods run over the finished instance (see `Topology.score`).

Agents are framework-managed: the runner builds one executable `Agent` per `AgentConfig`
field (a `list[AgentConfig]` field is one *role* with many seats) and hands them to `run`
as an `Agents` namespace mirroring the config. Each completed run records its trace onto
the instance's `AgentGraph` automatically (so a crash in `run` keeps finished work);
lineage is named at the call site, `run(task, parents=[upstream])`.

One instance produces one `AgentGraph`: the causally ordered view over its traces, each
linked to its parents. A `Trace` is the per-agent view of one run; the graph is the
cross-agent view. Interleaving two agents' *execution* is out of scope — a run completes
before its trace feeds anything downstream.
"""

import logging
import uuid
from collections.abc import Iterable, Mapping
from functools import cached_property
from typing import TYPE_CHECKING, Annotated, Generic, TypeVar, get_args, get_origin

from pydantic import Field, SerializeAsAny, model_validator
from pydantic_config import BaseConfig

from verifiers.v1.clients import ClientConfig
from verifiers.v1.decorators import discover_decorated, invoke
from verifiers.v1.harness import Harness, HarnessConfig
from verifiers.v1.task import Task
from verifiers.v1.taskset import Taskset, TasksetConfig
from verifiers.v1.trace import Error, Trace, TraceTask, WireTrace
from verifiers.v1.types import ID, SamplingConfig, StrictBaseModel
from verifiers.v1.utils.install import env_name

if TYPE_CHECKING:
    # Runtime would cycle: agent.py imports env.py, which imports this module.
    from verifiers.v1.agent import Agent

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


class AgentConfig(BaseConfig):
    """One agent in a topology: which harness drives its runs (and where —
    `harness.runtime`) and how its model calls are routed. Declared as a typed field of a
    `TopologyConfig` subclass — the field *name* is the agent's name in `run`, a
    `list[AgentConfig]` field is one *role* with several seats — so every agent is
    CLI/toml-addressable. An agent carries nothing task-side; tasks arrive per run.

    To pin a per-agent default harness, **subclass and set the field default**
    (`harness: SerializeAsAny[HarnessConfig] = HarnessConfig(id="direct")`): the pin must
    live on an `AgentConfig` subclass, not the outer topology-config field default, which
    a partial override would replace wholesale (pydantic re-validates the field, never
    merges into instance defaults). A pin survives everything but an explicit `harness.id`."""

    harness: SerializeAsAny[HarnessConfig] = HarnessConfig(id="default")
    """The program driving this agent's runs, and where (`harness.runtime`) — per agent,
    so a judge can run the in-process `direct` loop while a solver runs in a container."""
    model: str | None = None
    """Model override (None = the eval's model)."""
    client: ClientConfig | None = None
    """Client override (None = the eval's client) — e.g. a non-trainable judge routed to a
    plain API endpoint while the solver runs against the train client."""
    sampling: SamplingConfig | None = None
    """Sampling overrides, layered over the eval's sampling."""
    trainable: bool = True
    """Whether this agent's traces are training samples — stamped onto `Trace.trainable`
    so a trainer can filter without consulting the topology config."""

    @model_validator(mode="before")
    @classmethod
    def _resolve_plugins(cls, data):
        """Resolve `harness`, in precedence order: an explicit `id` swaps it (narrowing to
        that harness's config type); else a subclass **pin** — detected by value, so a
        base-typed `HarnessConfig(id="null")` counts — deep-merges partial overrides over
        the pinned default (`harness.runtime.type docker` tunes it, never resets it); else
        the base default narrows to the `default` harness."""
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
    """Pinned to the in-process `direct` chat loop — tool-less, a run ≈ one API call. The
    shared common pin so topologies don't each redeclare it."""

    harness: SerializeAsAny[HarnessConfig] = HarnessConfig(id="direct")


class NullAgentConfig(AgentConfig):
    """Pinned to the `null` chat loop — a subprocess program with the task's MCP tools but
    none of its own (calls task tools without being a full coding agent)."""

    harness: SerializeAsAny[HarnessConfig] = HarnessConfig(id="null")


def _agent_list_item_type(annotation) -> type[AgentConfig] | None:
    """The `AgentConfig` subclass a `list[...]` annotation holds (unwrapping `Annotated`
    wrappers like `SerializeAsAny` on both levels), else None — how a field is recognized
    as a list *role*."""
    while get_origin(annotation) is Annotated:
        annotation = get_args(annotation)[0]
    if get_origin(annotation) is not list:
        return None
    args = get_args(annotation)
    item = args[0] if args else None
    while get_origin(item) is Annotated:
        item = get_args(item)[0]
    if isinstance(item, type) and issubclass(item, AgentConfig):
        return item
    return None


class TopologyConfig(BaseConfig):
    """Base topology config. Subclass to declare the agents (typed `AgentConfig` fields —
    the field name is the agent's name; a `list[AgentConfig]` field is one role with
    several seats) plus any interaction knobs (fan-out width, number of rounds, reward
    weights). Mirrors `TasksetConfig`: the concrete subclass is resolved by `id`, so its
    fields surface typed on the CLI/toml."""

    id: ID = ""
    """The topology id, which selects this topology: a local package (e.g. an environment
    under `environments/`) or an `org/name[@version]` package installed on demand from the
    Environments Hub (see `ID`). Set via `--topology.id`."""
    taskset: SerializeAsAny[TasksetConfig] | None = None
    """The seed source: a taskset (resolved by id, `--topology.taskset.id <id>`) whose
    tasks seed the instances, one per seed. Exclusive-or with a `load_tasks` override: a
    topology that constructs its own seeds overrides `load_tasks` and is refused this
    flag, rather than silently ignoring it."""

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
    """A topology-declared agent slot: name + config + loaded harness.

    The value exposed inside `Topology.run` is an executable `Agent` built per instance;
    this binding is the config-side declaration the runner loads and validates once,
    before serving."""

    def __init__(
        self, name: str, config: AgentConfig, harness: Harness | None = None
    ) -> None:
        from verifiers.v1.loaders import load_harness

        self.name = name
        self.config = config
        self.harness = harness if harness is not None else load_harness(config.harness)


class Agents:
    """The framework-built executable agents of one topology instance, mirroring the
    config's declaration: an `AgentConfig` field is one `Agent` (`agents.judge`), a
    `list[AgentConfig]` field is a list of them (`agents.editors`, one role, several
    seats). Built fresh per instance by the runner with the run's model context, serving
    resources, budgets, and graph recording already bound — `run` just picks who acts."""

    def __init__(self, agents: "dict[str, Agent | list[Agent]]") -> None:
        self._agents = agents

    def __getattr__(self, name: str) -> "Agent | list[Agent]":
        if name.startswith("_"):
            raise AttributeError(name)
        try:
            return self.__dict__["_agents"][name]
        except KeyError:
            raise AttributeError(
                f"unknown agent {name!r}: this topology defines "
                f"{sorted(self.__dict__['_agents'])}"
            ) from None

    def __getitem__(self, name: str) -> "Agent | list[Agent]":
        try:
            return self._agents[name]
        except KeyError:
            raise KeyError(
                f"unknown agent {name!r}: this topology defines {sorted(self._agents)}"
            ) from None

    def __contains__(self, name: str) -> bool:
        return name in self._agents


class AgentGraph(StrictBaseModel):
    """One topology instance's artifact: the causally ordered view over its traces, each
    linked to its upstream (`trace.agent` / `trace.parents`). Traces append in completion
    order, so the list is topologically sorted. Persisted one graph per `traces.jsonl`
    line (traces nested); since each trace carries its own links, the graph also rebuilds
    from a flat trace dump (one instance = one connected component)."""

    id: str = Field(default_factory=lambda: uuid.uuid4().hex)
    """Unique id for this instance, auto-generated per graph."""
    topology: str = ""
    """The topology id that produced this instance."""
    task: SerializeAsAny[TraceTask]
    """The seed task for this invocation, including its task behavior type."""
    error: Error | None = None
    """A failure in the topology's own code (`run` or instance scoring, a `TopologyError`),
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
        """The traces a named agent (or every seat of a named role) produced, in
        completion order."""
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
    Subclass: declare `AgentConfig` fields on your config (a `list[AgentConfig]` field
    is one role with several seats), implement `run` (control flow) and
    `@vf.reward(agent=...)`/`@vf.metric(agent=...)` methods (judgement). Seeds come
    from the config's `taskset` slot, or override `load_tasks` to construct them."""

    def __init__(self, config: ConfigT) -> None:
        self.config = config

    @cached_property
    def taskset(self) -> Taskset | None:
        """The configured seed taskset, loaded once, or None for self-seeding topologies."""
        if self.config.taskset is None or not self.config.taskset.id:
            return None
        from verifiers.v1.loaders import load_taskset

        return load_taskset(self.config.taskset)

    @cached_property
    def agents(self) -> dict[str, AgentBinding | list[AgentBinding]]:
        """The declared agents, one binding per `AgentConfig` field (a `list[AgentConfig]`
        field yields a list under its role name), in declaration order. Also validates the
        declared `@reward`/`@metric` scopes against them, so a typo'd or missing agent
        scope fails at load time."""
        fields = type(self.config).model_fields
        agents: dict[str, AgentBinding | list[AgentBinding]] = {}
        for name, value in self.config:
            if isinstance(value, AgentConfig):
                agents[name] = AgentBinding(name, value)
            elif isinstance(value, list) and (
                name in fields
                and _agent_list_item_type(fields[name].annotation) is not None
            ):
                agents[name] = [AgentBinding(name, seat) for seat in value]
        if not agents:
            raise ValueError(
                f"topology {self.config.id!r} declares no agents: give its config "
                "`AgentConfig` fields"
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
        """The seed tasks — one instance runs per seed. Defaults to the config's `taskset`
        slot (whose `load` may be lazy or infinite); override to construct seeds (a finite
        iterable). A start-from-nothing topology returns identity-only stubs
        (`Task(TaskData(idx=i))`) — the seed is each instance's identity for `-n`, resume,
        and dispatch, not necessarily content."""
        if self.taskset is None:
            raise ValueError(
                f"topology {self.config.id!r} has no seed tasks: set --topology.taskset.id "
                "<id>, or override `load_tasks` to construct them"
            )
        return self.taskset.load()

    @property
    def infinite(self) -> bool:
        """Whether the seed source never ends (an `INFINITE` taskset in the slot; runs must
        then be bounded with `num_tasks`). A `load_tasks` override is finite by contract."""
        if type(self).load_tasks is not Topology.load_tasks:
            return False
        return self.taskset is not None and type(self.taskset).INFINITE

    async def setup(self) -> None:
        """Author hook: bring up topology-owned shared resources. Called once inside
        `serving()`, after the framework's own services are up. Default no-op."""

    async def teardown(self) -> None:
        """Author hook: tear down what `setup` brought up. Runs on serving exit, even when
        instances failed. Default no-op."""

    def complete(self, graph: AgentGraph) -> bool:
        """Whether a persisted instance is a valid result — the verdict `--resume` reads to
        decide what to redo. The default is conservative (no instance-level error, no
        errored trace); a topology whose `run` tolerates child failures overrides this
        (typically `graph.error is None`), else resume redoes instances it already
        accepted. Read-only: what a failed child *means* stays in `run` and the rewards."""
        return graph_complete(graph)

    async def score(self, graph: AgentGraph) -> None:
        """Run the declared judgement over one finished instance: every `@metric(agent=)`
        then every `@reward(agent=)`, once per trace the named agent produced (declaring
        `task`/`trace`/`graph` by parameter name), recorded on that trace.

        Ordering, for predictability: sequential, metrics before rewards, each phase in
        (priority, name) order. A method may read task-recorded rewards and, in the rewards
        phase, any metric — but rewards must not read each other; derive shared inputs from
        the traces or a metric. Mirrors `Task.score` one level up."""
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

    async def run(self, task: Task, agents: Agents) -> None:
        """Run one instance from seed `task`: *control flow only*.
        `await agents.<name>.run(task, parents=...)` per invocation, `asyncio.gather` for
        fan-out, loops for rounds. The forward arrow lives here — build the next agent's
        typed `Task` from an upstream trace and name the lineage at the call
        (`parents=[upstream]`). Every completed run records its trace automatically —
        nothing to return. Judgement belongs in the `@reward`/`@metric` methods (see
        `score`). Agent failures come back as data on their traces, never as exceptions —
        `run` decides what a failed child means."""
        raise NotImplementedError


class SingleAgentTopology(Topology[SingleAgentTopologyConfig]):
    """The canonical one-agent topology produced from taskset + harness syntax."""

    async def run(self, task: Task, agents: Agents) -> None:
        await agents.agent.run(task)
