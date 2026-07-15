"""The topology runner: framework glue between a `Topology` and one process's run.

`TopologyRunner` is the long-lived executor for one topology — it loads and validates the
declaration once, holds the worker-scoped services (`serving()`: the run's one
interception, the taskset's shared tool servers, per-agent override clients, the
topology's own `setup`/`teardown` resources), and runs instances (`run_instance`: build
the instance's `AgentGraph` and executable agents, execute `Topology.run`, then the
declared judgement). The authoring surface it executes lives in `verifiers.v1.topology`;
this module is the only half that knows about `EnvConfig` and serving.
"""

import asyncio
import contextlib
import logging
from collections.abc import Callable, Iterable, Iterator

from verifiers.v1.agent import Agent
from verifiers.v1.clients import Client, ModelContext, resolve_client
from verifiers.v1.env import (
    EnvConfig,
    taskset_server_configs,
    validate_pairing,
)
from verifiers.v1.errors import TopologyError, boundary
from verifiers.v1.interception import (
    Interception,
    make_interception,
    requires_tunnel,
)
from verifiers.v1.rollout import Rollout
from verifiers.v1.runtimes import SubprocessConfig
from verifiers.v1.session import RolloutLimits
from verifiers.v1.task import Task
from verifiers.v1.topology import (
    AgentBinding,
    AgentConfig,
    AgentGraph,
    Agents,
    SingleAgentTopology,
    SingleAgentTopologyConfig,
    Topology,
    TopologyConfig,
)
from verifiers.v1.trace import Error, TraceTask
from verifiers.v1.types import SamplingConfig
from verifiers.v1.utils.sampling import sample

logger = logging.getLogger(__name__)


def _merge_sampling(
    base: SamplingConfig, override: SamplingConfig | None
) -> SamplingConfig:
    """Agent sampling overrides layer onto the run sampling, field by field."""
    if override is None:
        return base
    return base.model_copy(update=override.model_dump(exclude_none=True))


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
        if (
            topology.config.taskset is not None
            and topology.config.taskset.id
            and type(self.topology).load_tasks is not Topology.load_tasks
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
            agent_binding = bindings["agent"]
            assert isinstance(agent_binding, AgentBinding)
            validate_pairing(agent_binding.harness, topology.taskset)
        for _, _, binding in self._iter_bindings():
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
        self._override_clients: dict[tuple[str, int | None], Client] = {}

    def _iter_bindings(self) -> Iterator[tuple[str, int | None, AgentBinding]]:
        """Flatten the declared agents to `(name, seat, binding)`: a scalar role yields
        seat None, a list role one entry per seat."""
        for name, value in self.topology.agents.items():
            if isinstance(value, list):
                for seat, binding in enumerate(value):
                    yield name, seat, binding
            else:
                yield name, None, value

    @property
    def infinite(self) -> bool:
        """Whether this topology's seed source is infinite (see `Topology.infinite`)."""
        return self.topology.infinite

    def load_tasks(self) -> Iterable[Task]:
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

    @contextlib.asynccontextmanager
    async def serving(self):
        """Hold worker-scoped services while request contexts remain invocation-local:
        the run's one interception, the taskset's shared tool servers, per-agent override
        clients, and the topology's own `setup`/`teardown` resources."""
        if self._interception is not None:
            raise RuntimeError("TopologyRunner.serving() is already active")
        from verifiers.v1.mcp import serve_shared
        from verifiers.v1.runtimes import runtime_is_local

        async with contextlib.AsyncExitStack() as stack:
            # Every agent's harness placement — any remote seat must reach the shared
            # servers and the interception through a tunnel.
            local = all(
                runtime_is_local(binding.harness.config.runtime)
                for _, _, binding in self._iter_bindings()
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
            # task's tool/user server — is off the host network. Tasks minted in `run`
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
            override_clients: dict[tuple[str, int | None], Client] = {}
            for name, seat, binding in self._iter_bindings():
                if binding.config.client is not None:
                    client = resolve_client(binding.config.client)
                    stack.push_async_callback(client.close)
                    override_clients[(name, seat)] = client
            await self.topology.setup()
            # Paired with the successful setup(): teardown runs first on unwind (LIFO),
            # while the framework services it may depend on are still up.
            stack.push_async_callback(self.topology.teardown)
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
        graph: AgentGraph,
        semaphore: asyncio.Semaphore | None = None,
        on_rollout: Callable[[Rollout], None] | None = None,
    ) -> Agents:
        """One instance's executable agents, mirroring the config declaration: the run's
        ctx with per-agent overrides merged, the serving resources, the eval budgets and
        retry policy, and this instance's graph recording — all bound at construction."""
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

        def build(name: str, seat: int | None, binding: AgentBinding) -> Agent:
            agent_ctx = ModelContext(
                model=binding.config.model or ctx.model,
                client=self._override_clients.get((name, seat), ctx.client),
                sampling=_merge_sampling(ctx.sampling, binding.config.sampling),
            )
            return Agent(
                binding.harness,
                agent_ctx,
                binding.harness.config.runtime,
                name=name,
                seat=seat,
                trainable=binding.config.trainable,
                limits=limits,
                timeout=self.config.timeout,
                interception=interception,
                shared_tools=self._shared_tools,
                retry=self.config.retries.rollout,
                semaphore=semaphore,
                on_trace=graph.add,
                on_rollout=on_rollout,
            )

        agents: dict[str, Agent | list[Agent]] = {}
        for name, value in self.topology.agents.items():
            if isinstance(value, list):
                agents[name] = [
                    build(name, seat, binding) for seat, binding in enumerate(value)
                ]
            else:
                agents[name] = build(name, None, value)
        return Agents(agents)

    async def run_instance(
        self,
        task: Task,
        ctx: ModelContext,
        semaphore: asyncio.Semaphore | None = None,
        on_rollout: Callable[[Rollout], None] | None = None,
    ) -> AgentGraph:
        """Run one topology instance — a single `Topology.run` over one seed task, then
        the declared instance judgement (`Topology.score`) — and return its agent graph.
        A failure in topology-authored code is classified (`TopologyError`) and captured
        on the graph, never raised: completed traces remain data (agents record them as
        they finish), and sibling instances keep running (mirrors `Rollout.run`'s
        a-bad-rollout-is-data stance one level up)."""
        graph = AgentGraph(
            topology=self.topology.config.id,
            task=TraceTask(type=type(task).__name__, data=task.data),
        )
        agents = self._agents_for(ctx, graph, semaphore, on_rollout)
        try:
            async with boundary(
                TopologyError, f"topology {self.topology.config.id!r} run"
            ):
                await self.topology.run(task, agents)
            # Instance judgement: the declared @reward/@metric methods over the finished
            # graph. Skipped when `run` itself failed — a broken instance isn't scored.
            async with boundary(
                TopologyError, f"topology {self.topology.config.id!r} scoring"
            ):
                await self.topology.score(graph)
        except TopologyError as e:
            logger.exception("topology instance failed (seed task %s)", task.data.idx)
            graph.error = Error(type=type(e).__name__, message=str(e))
        return graph
