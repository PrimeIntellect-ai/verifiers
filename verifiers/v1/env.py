"""The abstract Env: a taskset, its agents, and how one task becomes one episode."""

import asyncio
import contextlib
import logging
import traceback
from abc import ABC, abstractmethod
from collections.abc import Callable
from typing import TYPE_CHECKING, Generic, TypeVar

from pydantic import SerializeAsAny, model_validator
from pydantic_config import BaseConfig

from verifiers.v1.agents import (
    AgentConfig,
    Agents,
    agent_config_fields,
    contains_agent_config,
)
from verifiers.v1.clients import Client, ClientConfig, ModelContext, resolve_client
from verifiers.v1.errors import EnvError, boundary
from verifiers.v1.harness import HarnessConfig
from verifiers.v1.interception import (
    ElasticInterceptionPoolConfig,
    Interception,
    InterceptionConfig,
    make_interception,
    requires_tunnel,
)
from verifiers.v1.mcp import SharedToolServer, serve_shared
from verifiers.v1.retries import RetryConfig, run_episode_with_retry
from verifiers.v1.runtimes import SubprocessConfig, runtime_is_local
from verifiers.v1.task import Task, resolve_server_config
from verifiers.v1.taskset import Taskset, TasksetConfig
from verifiers.v1.trace import EpisodeInfo, Error, Trace, episode_ok
from verifiers.v1.types import ID
from verifiers.v1.utils.generic import deep_merge, generic_type
from verifiers.v1.utils.memory import trim_memory_periodically

if TYPE_CHECKING:
    from verifiers.v1.agents import Agent

logger = logging.getLogger(__name__)


class EnvTimeoutConfig(BaseConfig):
    """Wall-clock timeouts for the env's own hooks, in seconds (None = no limit)."""

    finalize: float | None = None
    """Max wall-clock for the env's cross-trace `finalize()` hook, run once per
    episode."""


class EnvConfig(BaseConfig):
    """An env's config — the run's `[env]` block. One subclass per `Env` class
    (bound via `Env[YourConfig]`, available as `self.config`): declare each agent
    as an `AgentConfig` field with a default instance, plus any env-level knobs.
    Per-run caps (turns, tokens, stage timeouts) are each agent's own
    (`--env.<agent>.max_turns`)."""

    id: ID = ""
    """Which `Env` runs. Empty = the taskset's own (the subclass its package
    exports, else `SingleAgentEnv`); set it to pair a reusable env with any
    taskset (`--env.id best-of-n`)."""
    # SerializeAsAny: keep the resolved subclass's fields in model_dump(); the
    # env-server wire would otherwise drop them.
    taskset: SerializeAsAny[TasksetConfig] | None = None
    """The seed taskset — the rows every rollout starts from (`--env.taskset.id`,
    positional shorthand `uv run eval <taskset-id>`)."""
    timeout: EnvTimeoutConfig = EnvTimeoutConfig()
    retries: RetryConfig = RetryConfig()
    max_concurrent: int | None = None
    """Bounds concurrent agent runs on a SERVED env, per worker (None = no limit);
    the in-process eval gates with its run-level `--max-concurrent` instead."""
    interception: InterceptionConfig = ElasticInterceptionPoolConfig()
    """The interception shape: `elastic` (default), `server`, or `static`."""

    @property
    def env_id(self) -> str:
        """The run's identifier — the taskset id, prefixed by the paired env id
        (`best-of-n+gsm8k-v1`)."""
        taskset_id = self.taskset.id if self.taskset is not None else ""
        if taskset_id and self.id:
            return f"{self.id}+{taskset_id}"
        return taskset_id or self.id

    def agent_harnesses(self) -> dict[str, HarnessConfig]:
        """Each declared agent's resolved harness config (pin, else the taskset's
        default) — known without constructing the env."""
        default = default_agent_harness(
            self.taskset.id if self.taskset is not None else ""
        )
        return {
            name: cfg.harness if cfg.harness is not None else default
            for name, cfg in agent_config_fields(self).items()
        }

    @model_validator(mode="before")
    @classmethod
    def _refuse_env_level_harness(cls, data):
        # Point the v0 muscle-memory spelling at the agent that owns it.
        if (
            isinstance(data, dict)
            and "harness" in data
            and "harness" not in cls.model_fields
        ):
            raise ValueError(
                "a harness belongs to an agent: --env.agent.harness.* on the "
                "single-agent env, --env.<agent>.harness.* on a multi-agent env "
                "(TOML: [env.agent.harness])"
            )
        return data

    @model_validator(mode="before")
    @classmethod
    def _resolve_taskset(cls, data):
        # Narrow `taskset` to its concrete config type by `id`. Lazy import: the
        # loaders import this module.
        if isinstance(data, dict) and data.get("taskset") is not None:
            from verifiers.v1.loaders import narrow_plugin_field, taskset_config_type

            narrow_plugin_field(data, "taskset", taskset_config_type)
        return data

    @model_validator(mode="before")
    @classmethod
    def _merge_agent_defaults(cls, data):
        # Deep-merge partial agent data over the declared default instance, so a
        # partial override (`--env.grader.sampling.temperature`) doesn't reset the
        # agent's other pins.
        if isinstance(data, dict):
            for name, field in cls.model_fields.items():
                if isinstance(field.default, AgentConfig) and isinstance(
                    data.get(name), dict
                ):
                    data[name] = deep_merge(
                        field.default.model_dump(exclude_none=True), data[name]
                    )
        return data

    @classmethod
    def __pydantic_init_subclass__(cls, **kwargs):
        # An agent field must carry a default instance (the deep-merge and the
        # `Agents` scrape read it) and must not shadow a base field.
        super().__pydantic_init_subclass__(**kwargs)
        for name, info in cls.model_fields.items():
            is_agent = isinstance(info.default, AgentConfig)
            if not is_agent and not contains_agent_config(info.annotation):
                continue
            if is_agent and name in EnvConfig.model_fields:
                raise TypeError(
                    f"{cls.__name__}.{name}: an agent can't shadow the base EnvConfig "
                    f"field {name!r}; pick another name"
                )
            if not is_agent:
                raise TypeError(
                    f"{cls.__name__}.{name}: declare the agent with a default "
                    f"instance (`{name}: vf.AgentConfig = vf.AgentConfig(...)`), "
                    "not default_factory or a bare annotation"
                )


def default_agent_harness(taskset_id: str) -> HarnessConfig:
    """What an unpinned agent's `harness=None` resolves to: the taskset's bundled
    harness when it ships one, else the built-in `bash`."""
    from verifiers.v1.loaders import default_harness_id, harness_config_type

    ident = default_harness_id(taskset_id)
    return harness_config_type(ident).model_validate({"id": ident})


def _as_error(e: Exception) -> Error:
    # Call inside the `except` handling `e` — the traceback comes from the active
    # exception context.
    return Error(
        type=type(e).__name__, message=str(e), traceback=traceback.format_exc()
    )


ConfigT = TypeVar("ConfigT", bound=EnvConfig)


class Env(ABC, Generic[ConfigT]):
    """A taskset, the agents that play it, and how one task becomes one episode.

    Abstract: every run gets a concrete subclass — `SingleAgentEnv` for every
    plain taskset. A multi-agent env declares each agent as an `AgentConfig` field
    on an `EnvConfig` subclass (bound via `Env[YourConfig]`) and writes
    `run(task, agents)`: imperative Python over the constructed `Agents`,
    returning nothing — every finished agent run joins the episode's traces
    automatically, stamped with its agent's name and the shared `EpisodeInfo`.
    Optional overrides: `setup(agents)` (per-agent standing, e.g. trainability)
    and `finalize(task, traces)` (cross-trace judgement, mutating traces in
    place). Task x agent fit is validated per run by `Agent.run` — nothing is
    compiled upfront."""

    def __init__(self, config: ConfigT) -> None:
        from verifiers.v1.loaders import harness_class, load_taskset

        config_cls = generic_type(type(self), EnvConfig, origin=Env) or EnvConfig
        if not isinstance(config, config_cls):
            raise TypeError(
                f"{type(self).__name__} declares Env[{config_cls.__name__}], "
                f"but was handed a {type(config).__name__}; build the run config "
                "naming this env (its `env` field narrows to it), or pass "
                f"{config_cls.__name__}(...) explicitly"
            )
        self.config: ConfigT = config
        if config.taskset is None:
            raise ValueError(
                f"{type(self).__name__} needs a seed taskset — set --env.taskset.id "
                "(or the positional `eval <taskset-id>`)"
            )
        self.taskset = load_taskset(config.taskset)
        task_cls = generic_type(type(self.taskset), Task, origin=Taskset) or Task
        self._task_cls: type[Task] = task_cls
        if not contains_agent_config(self.config):
            raise ValueError(
                f"{type(self).__name__} declares no agents; declare each as an "
                "AgentConfig field on the env's config "
                "(`solver: vf.AgentConfig = vf.AgentConfig()`) — the field name is "
                "the agent's name. The single-agent case is SingleAgentEnv."
            )
        warned: set[str] = set()
        for name, harness_config in self.config.agent_harnesses().items():
            # Warn once per distinct harness; tool-less chat loops are exempt.
            if (
                harness_class(harness_config.id).EXECUTES_CODE
                and isinstance(harness_config.runtime, SubprocessConfig)
                and harness_config.id not in warned
            ):
                warned.add(harness_config.id)
                logger.warning(
                    "Harness %r is running in the subprocess runtime on the local system. "
                    "Local files and settings may affect the evaluation; use subprocess only "
                    "for debugging. Use --env.<agent>.harness.runtime.type docker or prime "
                    "for an isolated run.",
                    harness_config.id,
                )
        # Eval-level serving resources, live only inside `serving()`; the env's
        # agents borrow them.
        self._shared_tools: dict[str, SharedToolServer] = {}
        self._interception: Interception | None = None
        self._agent_clients: dict[str, Client] = {}
        self._warned_resources: set = set()

    # --- the multi-agent surface (override these) ------------------------------

    def setup(self, agents: Agents) -> None:
        """Set up this episode's agents before `run()` sees them — per-agent
        standing the env hardcodes rather than exposes as config; today that is
        `trainable` (`agents.grader.trainable = False`). Runs once per episode."""

    @abstractmethod
    async def run(self, task: Task, agents: Agents) -> None:
        """One episode: how the agents interact on `task`. An agent-run failure is
        data on its trace (this hook decides what it means); an exception raised
        here is the episode itself failing."""

    async def finalize(self, task: Task, traces: list[Trace]):
        """Cross-trace judgement over one episode's finished traces (per-trace
        judgement already ran on each trace's own task): read the flat list —
        each trace's `agent.name` stamp names its agent — and record results in
        place (`record_reward`/`record_metric`). Bounded by `timeout.finalize`.
        Default: no-op."""

    def complete(self, traces: list[Trace]) -> bool:
        """Whether a finished episode is a valid result — what `--resume` keeps
        vs. redoes. Override to tolerate a failed participant (a forfeited
        player)."""
        return episode_ok(traces)

    # --- machinery (the base owns everything below) -----------------------------

    def _episode_agents(
        self,
        ctx: ModelContext,
        episode: EpisodeInfo,
        gate: "asyncio.Semaphore | None",
        completed: list[Trace],
        on_trace: Callable[[Trace], None] | None,
    ) -> Agents:
        """One episode's `Agents`, scraped off the config — fresh value objects
        riding the live serving resources, set up by `setup()` before `run()`
        sees them."""
        from verifiers.v1.agents import _EpisodeAgent

        default_harness = default_agent_harness(
            self.config.taskset.id if self.config.taskset is not None else ""
        )

        def make(name: str, spec: AgentConfig) -> "Agent":
            resolved = spec.model_copy(
                update={
                    "harness": spec.harness
                    if spec.harness is not None
                    else default_harness,
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
                episode=episode,
                shared_tools=self._shared_tools,
                task_cls=self._task_cls,
                gate=gate,
                completed=completed,
                on_trace=on_trace,
                warned_resources=self._warned_resources,
            )

        agents = Agents(self.config, make)
        self.setup(agents)
        return agents

    def _client_for(self, config: ClientConfig) -> Client:
        # Cached by config; closed when serving() exits.
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
        gate: asyncio.Semaphore | None = None,
    ) -> list[Trace]:
        """One episode of `task`: `run()` over the episode's agents, then
        `finalize()` over its traces, with whole-episode retries per
        `retries.rollout`. Returns the episode's traces, completion order, linked
        through the shared `EpisodeInfo` stamped at mint; a hook failure lands on
        `EpisodeInfo.errors` (mirrored on every completed trace), never on one
        trace. `gate` bounds the agent runs themselves, so an env's internal
        fan-out counts against `--max-concurrent` too."""
        traces = await run_episode_with_retry(
            lambda: self._attempt(task, ctx, on_trace, gate),
            self.config.retries.rollout,
        )
        # hand freed per-turn request bodies (base64 images) back to the OS
        await trim_memory_periodically()
        return traces

    async def _attempt(
        self,
        task: Task,
        ctx: ModelContext,
        on_trace: Callable[[Trace], None] | None,
        gate: asyncio.Semaphore | None,
    ) -> tuple[EpisodeInfo, list[Trace]]:
        episode = EpisodeInfo(env=self.config.env_id)
        completed: list[Trace] = []
        agents = self._episode_agents(ctx, episode, gate, completed, on_trace)
        try:
            async with boundary(EnvError, f"{type(self).__name__}.run()"):
                await self.run(task, agents)
                if not completed:
                    raise ValueError(
                        f"{type(self).__name__}.run() ran no agent — every "
                        "episode must carry at least one run"
                    )
        except Exception as e:
            # The completed subset is the crash-safe episode.
            episode.errors.append(_as_error(e))
            return episode, list(completed)
        try:
            async with asyncio.timeout(self.config.timeout.finalize):
                async with boundary(EnvError, f"{type(self).__name__}.finalize()"):
                    await self.finalize(task, list(completed))
        except Exception as e:
            # A TimeoutError here can only be the deadline's own expiry — one
            # raised inside finalize() became an EnvError at the boundary.
            if isinstance(e, TimeoutError):
                e = TimeoutError(
                    f"{type(self).__name__}.finalize() exceeded its "
                    f"{self.config.timeout.finalize:g}s deadline "
                    "(--env.timeout.finalize)"
                )
            episode.errors.append(_as_error(e))
        return episode, list(completed)

    @contextlib.asynccontextmanager
    async def serving(self):
        """Hold the env-level serving resources for the duration of an eval —
        shared tool servers and the interception. Run episodes inside; torn down
        on exit."""
        async with self.shared_tools() as shared:
            interception = make_interception(
                self.config.interception, requires_tunnel=self._requires_tunnel(shared)
            )
            async with interception:
                self._shared_tools = shared
                self._interception = interception
                try:
                    yield
                finally:
                    self._shared_tools = {}
                    self._interception = None
                    clients, self._agent_clients = self._agent_clients, {}
                    for client in clients.values():
                        with contextlib.suppress(Exception):
                            await client.close()

    def _runs_local(self) -> bool:
        # Any remote agent runtime means tunnels.
        return all(
            runtime_is_local(harness.runtime)
            for harness in self.config.agent_harnesses().values()
        )

    def _requires_tunnel(self, shared: dict[str, SharedToolServer]) -> bool:
        """`requires_tunnel` over the consumers known before any rollout: agent
        runtimes, live `shared` servers, and the task class's tool/user servers. A
        task class that overrides `server_config` isn't statically knowable, so it
        conservatively counts as remote."""
        task_cls = self._task_cls
        server_classes = [*task_cls.tools, *([task_cls.user] if task_cls.user else [])]
        if server_classes and task_cls.server_config is not Task.server_config:
            return True
        sole = len({*task_cls.tools} | ({task_cls.user} - {None})) == 1
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
