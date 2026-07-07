"""The topology: a surface over which agents interact.

A `Topology` composes *episodes* — one agent consuming one task and producing one trace —
into a multi-agent interaction: which agents exist, how one agent's trace becomes a
downstream agent's task, and how rewards flow backwards once downstream agents have run.
Each episode is an ordinary `Rollout` (same lifecycle, same error model, same trace), and
tasks are the whole task-side contract — their classes carry the behavior — so an `Agent`
is nothing but a name + a harness + routing, and any task runs under any agent.

The interaction pattern is plain imperative Python in `go` — not a DSL: a loop is rounds,
`gather` (or `asyncio.gather` over `rollout` calls) is fan-out, and awaiting several traces
before building the next task is fan-in. `go` owns *control flow only*, including the
forward arrow (trace → next task, pure host-side construction); judgement — including the
backward arrow, a reward derived from downstream episodes — is declared as
`@vf.reward(agent=...)`/`@vf.metric(agent=...)` methods, run over the finished instance
(see `Topology.score`).

Running one instance produces an `AgentGraph` — the serialized instance artifact: the
global, causally ordered view over its traces, each linked to its parents (`trace.agent` /
`trace.parents`). A `Trace` stays the per-agent view of one episode; the graph is the
cross-agent view of the whole interaction. Interleaving two agents' *execution* inside one
episode is deliberately out of scope — an episode runs to completion before its trace
feeds anything downstream.
"""

import asyncio
import contextlib
import logging
import uuid
from collections.abc import Mapping, Sequence
from functools import cached_property
from typing import Generic, TypeVar

from pydantic import Field, SerializeAsAny, model_validator
from pydantic_config import BaseConfig

from verifiers.v1.clients import ClientConfig, RolloutContext, resolve_client
from verifiers.v1.decorators import discover_decorated, invoke
from verifiers.v1.env import (
    EnvConfig,
    resolve_runtime_config,
    resolve_stage_timeouts,
    validate_pairing,
)
from verifiers.v1.errors import TopologyError, boundary
from verifiers.v1.harness import Harness, HarnessConfig
from verifiers.v1.interception import InterceptionPool
from verifiers.v1.mcp import SharedServers
from verifiers.v1.retries import run_with_retry
from verifiers.v1.rollout import Rollout
from verifiers.v1.task import Task
from verifiers.v1.taskset import TasksetConfig
from verifiers.v1.trace import Error, Trace, WireTrace
from verifiers.v1.types import ID, SamplingConfig, StrictBaseModel
from verifiers.v1.utils.install import env_name
from verifiers.v1.utils.memory import trim_memory_periodically

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
    """One agent in a topology: which harness drives its episodes (and where —
    `harness.runtime`), and how its model calls are routed. Declared as typed fields of a
    `TopologyConfig` subclass — the field *name* is the agent's name in `go` — so every
    agent is CLI/toml-addressable (`--topology.<agent>.harness.id`,
    `--topology.<agent>.model`, ...). An agent carries nothing task-side: the tasks it
    consumes (each carrying its own behavior) arrive per episode, from the topology's seeds
    or constructed in `go`.

    To pin a per-agent default harness, **subclass and set the field default**
    (`harness: SerializeAsAny[HarnessConfig] = HarnessConfig(id="direct")`) — pins must
    live on an `AgentConfig` subclass, never on the outer topology-config field's default
    instance, which a partial override (`--topology.<agent>.model ...`) would silently
    replace (pydantic re-validates the whole field; it never merges into instance
    defaults). A pinned harness survives everything except an explicit
    `--topology.<agent>.harness.id`."""

    harness: SerializeAsAny[HarnessConfig] = HarnessConfig(id="default")
    """The program driving this agent's episodes, and where it runs (`harness.runtime`) —
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
    """Sampling override for this agent (None = the eval's sampling)."""
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


ConfigT = TypeVar("ConfigT", bound=TopologyConfig)


class Agent:
    """A loaded agent: a name + the harness driving its episodes + routing, nothing more —
    a partially-applied rollout that `TopologyRunner.rollout(agent, task)` completes.
    Tasks (which carry their own behavior) arrive per episode; the task × harness pairing
    is validated once per task class, at the first episode that pairs them."""

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

    The trace's sibling, one level up: what a trace is to an episode, the graph is to an
    instance. A topology run persists one graph per line (`results.jsonl`), traces nested —
    and since each trace carries its own links, the graph is also *recoverable* from a flat
    trace dump (one instance = one connected component)."""

    id: str = Field(default_factory=lambda: uuid.uuid4().hex)
    """Unique id for this instance, auto-generated per graph."""
    topology: str = ""
    """The topology id that produced this instance."""
    error: Error | None = None
    """A failure in the topology's own code (`go` or instance scoring, a `TopologyError`),
    recorded instead of raised — episode failures live on their traces, this is for the
    composition itself. Traces completed before the failure remain data."""
    traces: list[SerializeAsAny[Trace]] = Field(default_factory=list)
    """Every episode's trace, in completion (= topological) order."""

    def add(self, trace: Trace) -> None:
        self.traces.append(trace)

    def roots(self) -> list[Trace]:
        """The traces with no parents — the entry episodes (usually one, on the seed task)."""
        return [trace for trace in self.traces if not trace.parents]

    def children(self, trace: Trace, agent: str | None = None) -> list[Trace]:
        """The traces derived from `trace` (its direct downstream episodes), optionally
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
        """A JSON-serializable record of this instance for `results.jsonl` — each nested
        trace dumped like `Trace.to_record` (per-node training tensors stripped)."""
        from verifiers.v1.trace import _NODE_DUMP_EXCLUDE

        return self.model_dump(
            mode="json", exclude={"traces": {"__all__": _NODE_DUMP_EXCLUDE}}
        )

    @classmethod
    def load(cls, data: dict) -> "AgentGraph":
        """Load a dumped instance record without the originating packages: each trace is
        typed as a `WireTrace` (task-specific fields ride in `task.model_extra`)."""
        graph = cls.model_validate({**data, "traces": []})
        graph.traces = [WireTrace.model_validate(t) for t in data.get("traces", [])]
        return graph


class Topology(Generic[ConfigT]):
    """Generic over its config type, so `self.config` is fully typed in subclasses.
    Subclass: declare `AgentConfig` fields on your config, implement `go` (control flow)
    and `@vf.reward(agent=...)`/`@vf.metric(agent=...)` methods (judgement). Seeds come
    from the config's `tasks` factory, or override `load_tasks` to construct them."""

    def __init__(self, config: ConfigT) -> None:
        self.config = config

    def load_agents(self) -> dict[str, Agent]:
        """The topology's agents, one per `AgentConfig` field on the config (in declaration
        order). Override only to compose agents programmatically — each is still just
        `Agent(name, config)`."""
        return {
            name: Agent(name, value)
            for name, value in self.config
            if isinstance(value, AgentConfig)
        }

    @cached_property
    def agents(self) -> dict[str, Agent]:
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
                    (
                        k
                        for k in ("reward", "metric", "group_reward", "stop")
                        if getattr(fn, k, False)
                    ),
                    None,
                )
                if kind is None:
                    continue
                if kind in ("group_reward", "stop"):
                    raise ValueError(
                        f"@{kind} is not supported on a Topology (found "
                        f"{fn.__name__!r}): stops and group rewards belong to Task "
                        "classes; cross-agent judgement is @vf.reward(agent=...)"
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

    def load_tasks(self) -> list[Task]:
        """The seed tasks — one topology instance (`go`) runs per seed task. Defaults to
        the config's `taskset` slot (`--topology.taskset.id <id>`); override for a
        topology that constructs its own seeds."""
        if not self.config.taskset.id:
            raise ValueError(
                f"topology {self.config.id!r} has no seed tasks: set --topology.taskset.id "
                "<id>, or override `load_tasks` to construct them"
            )
        from verifiers.v1.loaders import load_taskset

        return load_taskset(self.config.taskset).load_tasks()

    async def score(self, graph: AgentGraph) -> None:
        """Run the topology's declared judgement over one completed instance: every
        `@metric(agent=...)`, then every `@reward(agent=...)`, each invoked once per
        trace the named agent produced — declaring any of `task`/`trace`/`graph` by
        parameter name. Runs after `go` returns and before the instance persists, so
        every episode (across all rounds and fan-outs) is scored, automatically.

        Ordering contract, chosen for predictability over cleverness: methods run
        *sequentially*, metrics before rewards, each phase in (priority, name) order.
        A method may read task-recorded rewards (final since the episode ended) and,
        in the rewards phase, any metric — but topology rewards must not read each
        other; derive shared inputs from the traces (or a metric) instead. Mirrors
        `Task.score` one level up: episode judgement there, instance judgement here."""
        for kind in ("metric", "reward"):
            for fn in discover_decorated(self, kind):
                weight = getattr(fn, "_vf_weight", 1.0)
                for trace in graph.by_agent(getattr(fn, "_vf_agent", None)):
                    available = {"task": trace.task, "trace": trace, "graph": graph}
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
        episodes run, in what order, with what tasks: `await run.rollout(agent, task,
        parents=...)` per episode, `run.gather` for fan-out, loops for rounds, and
        `trace.info[...] = ...` to annotate provenance. The forward arrow lives here —
        construct the next agent's typed `Task` from an upstream trace (its typed task,
        `last_reply`, `transcript`, or what its `finalize` peeled into `trace.info`).
        Judgement belongs in the topology's `@reward(agent=...)`/`@metric(agent=...)`
        methods (see `score`); recording a reward imperatively here is the escape hatch,
        not the norm. Episode failures come back as data on their traces (`trace.error`),
        never as exceptions — `go` decides what a failed child means (drop it, count it
        against a pass rate, retry the round, ...)."""
        raise NotImplementedError


class TopologyRun:
    """One live topology instance: the execution surface `go` programs against. Owns the
    instance's `AgentGraph` and links every episode into it; concurrency, retries, and
    limits come from the eval config, identically to a single-agent eval."""

    def __init__(
        self,
        env: "TopologyRunner",
        semaphore: asyncio.Semaphore | None = None,
    ) -> None:
        self.env = env
        self.graph = AgentGraph(topology=env.topology.config.id)
        self._semaphore = semaphore

    async def rollout(
        self, agent: str, task: Task, parents: Sequence[Trace] = ()
    ) -> Trace:
        """Run one episode — `agent` consuming `task` — and return its trace, linked under
        `parents` in the agent graph. Bounded by the eval's rollout concurrency and retried
        per its whole-rollout retry policy. Never raises on a failed episode: the error is
        data on the returned trace (`trace.error`)."""
        binding = self.env.agent(agent)
        rollout = self.env.rollout(binding, task)
        async with self._semaphore or contextlib.nullcontext():
            trace = await run_with_retry(rollout, self.env.config.retries.rollout)
        trace.agent = binding.name
        trace.parents = [parent.id for parent in parents]
        trace.trainable = binding.config.trainable
        self.graph.add(trace)
        # hand freed per-turn request bodies (base64 images) back to the OS
        await trim_memory_periodically()
        return trace

    async def gather(
        self, agent: str, tasks: Sequence[Task], parents: Sequence[Trace] = ()
    ) -> list[Trace]:
        """Fan out: run `agent` over `tasks` concurrently and return the traces aligned to
        `tasks` (`[task] * n` samples one task n times). Fan-in is just Python: await the
        batch, then read what you need off the traces."""
        return list(
            await asyncio.gather(
                *(self.rollout(agent, task, parents) for task in tasks)
            )
        )

    async def score_group(self, traces: list[Trace]) -> None:
        """Run the shared task's `@group_reward`s across `traces` — for a fan-out that is a
        classic group (n rollouts of one derived task, scored comparatively)."""
        if traces:
            await traces[0].task.score_group(traces)


class TopologyRunner:
    """The topology-level composition and resolver — the multi-agent counterpart of
    `Environment`. Loads the topology, holds the eval-level run settings (timeouts,
    limits, retries, multiplex), and turns one seed task into a running instance
    (`run_instance`). Execution of each episode lives in `Rollout`, exactly as in a
    single-agent eval; the topology's `go` decides what runs when."""

    def __init__(self, topology_config: TopologyConfig, config: EnvConfig) -> None:
        from verifiers.v1.loaders import load_topology

        self.config = config
        self.topology = load_topology(topology_config)
        # The seed contract is exclusive-or: the `taskset` slot IS the seed source, verbatim,
        # or `load_tasks` is overridden — never both. A self-seeding topology accepting a
        # `--topology.taskset.id` it then ignores would silently run a different experiment
        # than the config claims, so refuse it up front.
        if topology_config.taskset.id and (
            type(self.topology).load_tasks is not Topology.load_tasks
        ):
            raise ValueError(
                f"topology {topology_config.id!r} constructs its own seeds (it overrides "
                "`load_tasks`), so `--topology.taskset.id` would be silently ignored; drop "
                "it. A topology wanting a config-driven source *inside* a custom "
                "`load_tasks` declares its own factory field, not the built-in `taskset` slot."
            )
        self._warned_resources: set[tuple[str, str]] = set()
        self._validated: set[tuple[type[Task], str]] = set()
        """`(task class, agent name)` pairs already `validate_pairing`-checked — derived
        task classes don't exist at load time, so the check runs (once) at the first
        episode that pairs them with an agent's harness."""
        self._ctxs: dict[str, RolloutContext] = {}
        self._pools: dict[str, InterceptionPool] = {}
        self._shared: SharedServers | None = None
        """Run-level serving resources, live only inside `serving()`: one rollout context
        and one interception pool per agent (an agent's harness may run in its own runtime,
        against its own model/client), plus the lazy shared tool-server registry."""

    def agent(self, name: str) -> Agent:
        """The named agent, with an actionable error for a typo in `go`."""
        try:
            return self.topology.agents[name]
        except KeyError:
            raise ValueError(
                f"unknown agent {name!r}: topology {self.topology.config.id!r} defines "
                f"{sorted(self.topology.agents)}"
            ) from None

    @contextlib.asynccontextmanager
    async def serving(self, ctx: RolloutContext):
        """Hold the run-level serving resources for the duration of a topology eval: per
        agent, a `RolloutContext` (`ctx` with the agent's model/client/sampling overrides
        applied) and an interception pool — pools are deduped by the harness runtime they
        tunnel to, so two agents on identical runtimes share servers + tunnels — plus the
        lazy shared tool-server registry. Clients built for an override are closed on
        exit. Everything is built into locals and published atomically, so a failure
        mid-enter never leaves the runner pointing at torn-down resources. Build
        instances inside this context."""
        async with contextlib.AsyncExitStack() as stack:
            shared = await stack.enter_async_context(SharedServers())
            ctxs: dict[str, RolloutContext] = {}
            pools: dict[str, InterceptionPool] = {}
            pool_by_runtime: dict[str, InterceptionPool] = {}
            for name, agent in self.topology.agents.items():
                client = ctx.client
                if agent.config.client is not None:
                    client = resolve_client(agent.config.client)
                    stack.push_async_callback(client.close)
                ctxs[name] = RolloutContext(
                    model=agent.config.model or ctx.model,
                    client=client,
                    sampling=agent.config.sampling or ctx.sampling,
                )
                runtime_key = agent.harness.config.runtime.model_dump_json()
                if runtime_key not in pool_by_runtime:
                    pool_by_runtime[runtime_key] = await stack.enter_async_context(
                        InterceptionPool(
                            agent.harness.config.runtime, self.config.multiplex
                        )
                    )
                pools[name] = pool_by_runtime[runtime_key]
            self._ctxs, self._pools, self._shared = ctxs, pools, shared
            try:
                yield
            finally:
                self._ctxs, self._pools, self._shared = {}, {}, None

    def rollout(self, agent: Agent, task: Task) -> Rollout:
        """Resolve one episode — `task` run by `agent` — into a `Rollout` wired to the
        agent's runtime, context, and interception pool, under the eval-level timeouts and
        limits (task overrides apply as usual). The task × harness pairing is validated on
        the first episode that pairs them (memoized per class), so a bad pairing fails
        before any runtime spins up."""
        if (type(task), agent.name) not in self._validated:
            validate_pairing(type(task), agent.harness)
            self._validated.add((type(task), agent.name))
        runtime_config = resolve_runtime_config(
            agent.harness.config.runtime, task, self._warned_resources
        )
        timeouts = resolve_stage_timeouts(self.config.timeout, task, runtime_config)
        return Rollout(
            task=task,
            harness=agent.harness,
            ctx=self._ctxs[agent.name],
            runtime_config=runtime_config,
            setup_timeout=timeouts.setup,
            harness_timeout=timeouts.rollout,
            finalize_timeout=timeouts.finalize,
            scoring_timeout=timeouts.scoring,
            limits=self.config.limits,
            shared=self._shared,
            interception=self._pools.get(agent.name),
        )

    async def run_instance(
        self, task: Task, semaphore: asyncio.Semaphore | None = None
    ) -> AgentGraph:
        """Run one topology instance — a single `go` over one seed task, then the declared
        instance judgement (`Topology.score`) — and return its agent graph. A failure in
        topology-authored code is classified (`TopologyError`) and captured on the graph,
        never raised: episodes already completed remain data, and sibling instances keep
        running (mirrors `Rollout.run`'s a-bad-rollout-is-data stance one level up)."""
        run = TopologyRun(self, semaphore)
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
            logger.exception("topology instance failed (seed task %s)", task.idx)
            run.graph.error = Error(type=type(e).__name__, message=str(e))
        return run.graph
