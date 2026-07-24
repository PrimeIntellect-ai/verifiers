"""Compose a taskset, its agents, and how one task becomes one episode."""

import asyncio
import contextlib
import logging
import traceback
from abc import ABC, abstractmethod
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import (
    Generic,
    TypeVar,
)


from verifiers.v1.agent import Agent, Agents, _EpisodeAgent
from verifiers.v1.configs.agent import AgentConfig
from verifiers.v1.configs.env import (
    EnvConfig,
    _declared_agent_configs,
    default_agent_harness,
)
from verifiers.v1.harness import Harness, HarnessConfig
from verifiers.v1.clients import Client, ClientConfig, ModelContext, resolve_client
from verifiers.v1.interception import (
    Interception,
    make_interception,
    requires_tunnel,
)
from verifiers.v1.retries import run_episode_with_retry
from verifiers.v1.runtimes import SubprocessConfig, runtime_is_local
from verifiers.v1.errors import EnvError, boundary
from verifiers.v1.task import Task, resolve_server_config
from verifiers.v1.episode import Episode
from verifiers.v1.trace import Error, Trace
from verifiers.v1.utils.generic import generic_type
from verifiers.v1.utils.memory import trim_memory_periodically
from verifiers.v1.mcp import SharedToolServer, serve_shared


logger = logging.getLogger(__name__)


def _as_error(e: Exception) -> Error:
    """`e` as an episode-level `Error`. Call inside the `except` handling `e` — the
    traceback comes from the active exception context."""
    return Error(
        type=type(e).__name__, message=str(e), traceback=traceback.format_exc()
    )


@dataclass
class RunSlot:
    """One planned episode, observable while it happens: `traces` collects the
    current attempt (a retry restarts the list), `episode`/`done` land when final."""

    task: Task
    traces: list[Trace] = field(default_factory=list)
    episode: Episode | None = None
    done: bool = False

    @classmethod
    def finished(cls, episode: Episode) -> "RunSlot":
        return cls(
            task=Task(episode.traces[0].task.data),
            traces=list(episode.traces),
            episode=episode,
            done=True,
        )


ConfigT = TypeVar("ConfigT", bound=EnvConfig)


class Env(ABC, Generic[ConfigT]):
    """A taskset, the agents that play it, and how one task becomes one episode.

    Abstract: every run gets a concrete subclass — `SingleAgentEnv` for every plain
    taskset. A multi-agent env declares each role as an `AgentConfig` field on an
    `EnvConfig` subclass (bound via `Env[YourConfig]`) and writes
    `run(task, agents)`; optional overrides: `setup`, `finalize`, `start`/`stop`
    — the base owns everything else. Task x agent fit is validated per run, on the
    task the agent actually receives — an env-minted task carries its own needs."""

    def __init__(self, config: ConfigT) -> None:
        from verifiers.v1.loaders import load_harness, load_taskset

        config_cls = generic_type(type(self), EnvConfig, origin=Env) or EnvConfig
        if not isinstance(config, config_cls):
            raise TypeError(
                f"{type(self).__name__} declares Env[{config_cls.__name__}], "
                f"but was handed a {type(config).__name__}; build the run config "
                "naming this env (its `env` field narrows to it), or pass "
                f"{config_cls.__name__}(...) explicitly"
            )
        self.config: ConfigT = config
        if not config.taskset.id:
            raise ValueError(
                f"{type(self).__name__} needs a seed taskset — every rollout starts "
                "from one of its tasks: set --env.taskset.id (or the positional "
                "`eval <taskset-id>`)"
            )
        self.taskset = load_taskset(config.taskset)
        self._default_harness = default_agent_harness(config.taskset.id)
        task_cls = type(self.taskset).task_type()
        self._task_cls: type[Task] = task_cls
        self._agent_specs: dict[str, AgentConfig] = _declared_agent_configs(self.config)
        if not self._agent_specs:
            raise ValueError(
                f"{type(self).__name__} declares no agents; declare each as an "
                "AgentConfig field on the env's config "
                "(`solver: vf.AgentConfig = vf.AgentConfig()`) — the field name is "
                "the agent's name. The single-agent case is SingleAgentEnv."
            )
        # Same harness config -> one loaded object (harnesses are stateless values).
        loaded: dict[str, Harness] = {}
        self._harnesses: dict[str, Harness] = {}
        for name, spec in self._agent_specs.items():
            cfg = self._agent_harness(spec)
            key = cfg.model_dump_json()
            if key not in loaded:
                loaded[key] = load_harness(cfg)
            self._harnesses[name] = loaded[key]
        warned: set[str] = set()
        for name, harness in self._harnesses.items():
            # Warn once per distinct harness; tool-less chat loops are exempt.
            if (
                harness.EXECUTES_CODE
                and isinstance(self._agent_specs[name].runtime, SubprocessConfig)
                and harness.config.id not in warned
            ):
                warned.add(harness.config.id)
                logger.warning(
                    "Harness %r is running in the subprocess runtime on the local system. "
                    "Local files and settings may affect the evaluation; use subprocess only "
                    "for debugging. Use --env.<role>.runtime.type docker or prime "
                    "for an isolated run.",
                    harness.config.id,
                )
        # Serving resources, live only inside `serving()`; the env's agents borrow them.
        self._shared_tools: dict[str, SharedToolServer] = {}
        self._interception: Interception | None = None
        # Clients for endpoint-pinning roles, cached by config, closed with serving().
        self._agent_clients: dict[str, Client] = {}
        # Resource warnings dedupe env-wide (agents are per-episode).
        self._warned_resources: set = set()

    # --- the multi-agent surface (override these) ------------------------------

    async def setup(self, agents: Agents) -> None:
        """Before `run()` sees this episode's agents — standing the env hardcodes,
        today `trainable` (`agents.judge.trainable = False`); once per episode."""

    @abstractmethod
    async def run(self, task: Task, agents: Agents) -> None:
        """One episode: how the agents interact on `task`, returning nothing —
        every finished run joins the episode automatically, stamped with its seat's
        standing. An agent-run failure is data on its trace (this hook decides what
        it means); an exception raised here is the episode itself failing."""

    async def finalize(self, task: Task, episode: Episode) -> None:
        """Cross-agent judgement — THE programmable judgement surface: plain
        imperative Python over the finished episode (per-trace judgement already
        ran on each trace's own task). `episode.traces` is the flat episode in
        completion order, each trace's `agent_name` stamp naming its agent; attach
        signals via `record_reward`/`record_metric`, in program order. A raise
        fails the episode (the retryable unit) — validate strictly, never
        record a guess."""

    def complete(self, episode: Episode) -> bool:
        """Whether a finished episode is a valid result — what `--resume` keeps
        vs. redoes (default `episode.ok`). An env whose `run()` tolerates a
        failed participant overrides this, else resume re-runs accepted rollouts."""
        return episode.ok

    async def start(self) -> None:
        """Bring up env-owned shared resources — inside `serving()`, after the
        framework's resources are live and before any rollout. Default: no-op."""

    async def stop(self) -> None:
        """Tear down what `start()` built. Runs when `serving()` exits, even when
        `start()` failed partway — so it must tolerate partial state. Default: no-op."""

    # --- machinery (the base owns everything below) -----------------------------

    def _agent_harness(self, agent: AgentConfig) -> HarnessConfig:
        """The role's harness config: its own pin, else the taskset's default."""
        return agent.harness if agent.harness is not None else self._default_harness

    def _episode_agents(
        self,
        ctx: ModelContext,
        gate: "asyncio.Semaphore | None",
        completed: list[Trace],
        on_trace: Callable[[Trace], None] | None,
        on_discard: Callable[[Trace], None] | None,
    ) -> Agents:
        """One episode's `Agents` — fresh value objects riding the live serving
        resources (nothing shared across concurrent episodes); `setup()` sees them first."""

        def make(name: str, spec: AgentConfig) -> Agent:
            # Unpinned fields fall back to the run's ctx / the taskset's harness.
            resolved = spec.model_copy(
                update={
                    "harness": spec.harness
                    if spec.harness is not None
                    else self._default_harness,
                    "model": spec.model if spec.model is not None else ctx.model,
                    "sampling": spec.sampling
                    if spec.sampling is not None
                    else ctx.sampling,
                }
            )
            return _EpisodeAgent(
                resolved,
                client=self._client_for(spec.client)
                if spec.client is not None
                else ctx.client,
                interception=self._interception,
                name=name,
                shared_tools=self._shared_tools,
                task_cls=self._task_cls,
                gate=gate,
                completed=completed,
                on_trace=on_trace,
                on_discard=on_discard,
                warned_resources=self._warned_resources,
            )

        agents = Agents(self.config, make)
        return agents

    def _client_for(self, config: ClientConfig) -> Client:
        """Resolve (and cache by config) an agent-pinned endpoint's client."""
        key = config.model_dump_json()
        if key not in self._agent_clients:
            self._agent_clients[key] = resolve_client(config)
        return self._agent_clients[key]

    async def run_episode(
        self,
        task: Task,
        ctx: ModelContext,
        *,
        on_trace: Callable[[Trace], None] | None = None,
        on_discard: Callable[[Trace], None] | None = None,
        gate: asyncio.Semaphore | None = None,
    ) -> Episode:
        """One episode of `task`, minted as the wire atom: `setup()` then `run()`
        over fresh agents, then `finalize()`; `gate` bounds
        the agent runs, so internal fan-out counts against `--max-concurrent` too.
        Traces join the episode as runs complete — a hook raising mid-way yields the
        completed subset, its exception on the episode's `errors`. `on_trace` observes
        each agent-run's trace at mint; `on_discard` its abandonment (a per-agent
        retry mints a replacement)."""
        episode = Episode(env=self.config.env_id)
        agents = self._episode_agents(ctx, gate, episode.traces, on_trace, on_discard)
        try:
            async with asyncio.timeout(self.config.timeout.episode):
                async with boundary(EnvError, f"{type(self).__name__}.setup()"):
                    await self.setup(agents)
                async with boundary(EnvError, f"{type(self).__name__}.run()"):
                    await self.run(task, agents)
                    if not episode.traces:
                        raise ValueError(
                            f"{type(self).__name__}.run() ran no agent — every "
                            "episode must carry at least one run"
                        )
        except Exception as e:
            # Only the deadline's expiry: inner TimeoutErrors became EnvError already.
            if isinstance(e, TimeoutError):
                e = TimeoutError(
                    f"{type(self).__name__}.run() exceeded its "
                    f"{self.config.timeout.episode:g}s deadline (--env.timeout.episode)"
                )
            episode.errors.append(_as_error(e))
            # The completed subset is the crash-safe episode; ok stays False.
            return episode
        try:
            async with asyncio.timeout(self.config.timeout.finalize):
                async with boundary(EnvError, f"{type(self).__name__}.finalize()"):
                    await self.finalize(task, episode)
        except Exception as e:
            # As above: a TimeoutError here is the deadline's own expiry.
            if isinstance(e, TimeoutError):
                e = TimeoutError(
                    f"{type(self).__name__}.finalize() exceeded its "
                    f"{self.config.timeout.finalize:g}s deadline (--env.timeout.finalize)"
                )
            episode.errors.append(_as_error(e))
            return episode
        # Both hooks and every trace concluded — stamp the attempt's verdict
        # (retry history merges into `errors` later without touching it).
        episode.ok = all(t.ok for t in episode.traces)
        return episode

    def slots(self, task: Task, n: int = 1) -> list[RunSlot]:
        """Plan `n` independent episodes of `task` (`-r n`): nothing couples them."""
        if n < 1:
            raise ValueError("a task needs at least one rollout (n >= 1)")
        return [RunSlot(task) for _ in range(n)]

    async def run_slot(
        self,
        slot: RunSlot,
        ctx: ModelContext,
        semaphore: asyncio.Semaphore | None = None,
        on_complete: Callable[[Episode], Awaitable[None]] | None = None,
    ) -> Episode:
        """Run one planned episode to completion, with whole-episode
        retries per `--env.retries`; `semaphore` gates the agent RUNS, not the
        episode; `on_complete` (the runners' persistence hook) fires when final."""

        async def attempt() -> Episode:
            slot.traces = []  # a retry shows the fresh attempt's traces
            live = slot.traces

            def discard(trace: Trace) -> None:
                # A retried agent attempt abandons its trace; drop it from the view.
                with contextlib.suppress(ValueError):
                    live.remove(trace)

            return await self.run_episode(
                slot.task,
                ctx,
                on_trace=live.append,
                on_discard=discard,
                gate=semaphore,
            )

        episode = await run_episode_with_retry(attempt, self.config.retries)
        slot.traces = list(episode.traces)
        slot.episode = episode
        slot.done = True
        if on_complete is not None:
            await on_complete(episode)
        # hand freed per-turn request bodies (base64 images) back to the OS
        await trim_memory_periodically()
        return episode

    @contextlib.asynccontextmanager
    async def serving(self):
        """Hold the env-level serving resources for the duration of an eval; plan and
        run slots inside. Torn down on exit (`teardown()`, then the framework's)."""
        async with self.shared_tools() as shared:
            interception = make_interception(
                self.config.interception, requires_tunnel=self._requires_tunnel(shared)
            )
            async with interception:
                self._shared_tools = shared
                self._interception = interception
                try:
                    await self.start()
                    yield
                finally:
                    try:
                        # stop() sees what start() saw; the framework unwinds after.
                        await self.stop()
                    finally:
                        self._shared_tools = {}
                        self._interception = None
                        clients, self._agent_clients = self._agent_clients, {}
                        for client in clients.values():
                            with contextlib.suppress(Exception):
                                await client.close()

    def _runs_local(self) -> bool:
        """Whether every role's runtime policy is local (any remote role means tunnels)."""
        return all(
            runtime_is_local(spec.runtime) for spec in self._agent_specs.values()
        )

    def _requires_tunnel(self, shared: dict[str, SharedToolServer]) -> bool:
        """`requires_tunnel` over the consumers known before any rollout: role
        runtimes, live `shared` servers, and the task class's tool servers;
        a class overriding `server_config` conservatively counts as remote."""
        task_cls = type(self.taskset).task_type()
        server_classes = [*task_cls.tools]
        if server_classes and task_cls.server_config is not Task.server_config:
            return True
        sole = len({*task_cls.tools}) == 1
        configs = [
            resolve_server_config(
                task_cls.__name__, self.taskset.config.task, server_cls, sole=sole
            )
            for server_cls in server_classes
        ]
        return requires_tunnel(self._runs_local(), configs, shared.values())

    @contextlib.asynccontextmanager
    async def shared_tools(self):
        servers = self.taskset.tool_servers()
        if not servers:
            yield {}
            return
        async with serve_shared(servers, harness_is_local=self._runs_local()) as shared:
            yield shared
