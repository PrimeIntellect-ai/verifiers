"""Compose a taskset, its agents, and how one task becomes one env-rollout."""

import asyncio
import contextlib
import logging
import traceback
from abc import ABC, abstractmethod
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import (
    ClassVar,
    Generic,
    TypeVar,
    get_args,
)

from pydantic import SerializeAsAny, model_validator
from pydantic_config import BaseConfig

from verifiers.v1.agent import Agent, AgentConfig, Agents, _EpisodeAgent
from verifiers.v1.harness import Harness, HarnessConfig
from verifiers.v1.clients import Client, ClientConfig, ModelContext, resolve_client
from verifiers.v1.types import ID
from verifiers.v1.interception import (
    ElasticInterceptionPoolConfig,
    Interception,
    InterceptionConfig,
    make_interception,
    requires_tunnel,
)
from verifiers.v1.retries import RetryConfig, run_episode_with_retry
from verifiers.v1.runtimes import SubprocessConfig, runtime_is_local
from verifiers.v1.decorators import discover_decorated, invoke
from verifiers.v1.errors import EnvError, boundary
from verifiers.v1.task import Task, _record_result, resolve_server_config
from verifiers.v1.taskset import Taskset, TasksetConfig
from verifiers.v1.trace import Error, Episode, Trace, TraceTask
from verifiers.v1.utils.generic import deep_merge, generic_type
from verifiers.v1.utils.memory import trim_memory_periodically
from verifiers.v1.mcp import SharedToolServer, serve_shared


class EnvTimeoutConfig(BaseConfig):
    """Wall-clock timeouts for the env's own hooks, in seconds (None = no limit).
    Per-run stage timeouts are each agent's (`--env.<agent>.timeout.*`)."""

    episode: float | None = None
    """Max wall-clock for the env's `rollout()` hook — the whole interaction
    (each agent run inside it is additionally bounded by its own stages)."""
    score: float | None = None
    """Max wall-clock for the env's cross-trace `score()` hook, run once per
    env-rollout (per-trace scoring is bounded by each agent's `timeout.scoring`)."""


def _mentions_agent_config(annotation) -> bool:
    """Whether `annotation` names an `AgentConfig` — directly or inside
    Optional/union/Annotated/container forms — i.e. anything an author plausibly
    meant as an agent declaration."""
    if isinstance(annotation, type):
        return issubclass(annotation, AgentConfig)
    return any(_mentions_agent_config(arg) for arg in get_args(annotation))


class EnvConfig(BaseConfig):
    """An environment's config — the run's single `[env]` block. One subclass per
    `Environment` class (bound via `Environment[YourConfig]`, available as
    `self.config`): declare each role as an `AgentConfig` field with a default
    instance, plus any env-level knobs. The run's `env` field narrows to it by the
    env `id` (else the taskset id), which is what gives `--env.<role>.model`-style
    CLI/TOML addressing. The base carries what every environment has: which env
    and the seed taskset — per-run caps (turns, tokens, stage timeouts) are each
    seat's own (`--env.<role>.max_turns`)."""

    id: ID = ""
    """Which `Environment` runs. Empty = the taskset's own: the subclass its package
    exports, else `SingleAgentEnv`. Set it to pair a reusable env with any taskset
    (`--env.id best-of-n`): a bundled env, a local package, or a Hub
    `org/name[@version]`. An explicit id wins over the taskset's bundled env."""
    # SerializeAsAny: keep the resolved subclass's fields in model_dump(); the
    # env-server wire would otherwise drop them.
    taskset: SerializeAsAny[TasksetConfig] | None = None
    """The seed taskset — the rows every rollout starts from (`--env.taskset.id`,
    positional shorthand `uv run eval <taskset-id>`). None only for an env that
    mints its tasks without a dataset; every bundled env requires one."""
    timeout: EnvTimeoutConfig = EnvTimeoutConfig()
    retries: RetryConfig = RetryConfig()
    max_concurrent: int | None = None
    """Bounds concurrent agent runs on a SERVED env, per worker — an env's internal
    fan-out counts, so best-of-n under many requests can't run unbounded (None = no
    limit). The in-process eval CLI gates with its run-level `--max-concurrent`
    instead."""
    interception: InterceptionConfig = ElasticInterceptionPoolConfig()
    """The interception shape (see `verifiers.v1.interception`): `elastic` (default —
    servers grown on demand), `server` (one server, with a tunnel choice), or
    `static` (a fixed list of servers)."""

    @property
    def env_id(self) -> str:
        """The run's identifier — the taskset id, prefixed by the paired env id
        (`best-of-n+gsm8k-v1`), so paired and plain runs stay distinguishable on
        episodes and output dirs."""
        taskset_id = self.taskset.id if self.taskset is not None else ""
        if taskset_id and self.id:
            return f"{self.id}+{taskset_id}"
        return taskset_id or self.id

    def agent_harnesses(self) -> dict[str, HarnessConfig]:
        """Each declared role's resolved harness config (pin, else the taskset's
        default) — known without constructing the env, for output naming and the
        dashboard."""
        default = default_agent_harness(
            self.taskset.id if self.taskset is not None else ""
        )
        return {
            name: cfg.harness if cfg.harness is not None else default
            for name, cfg in _declared_agent_configs(self).items()
        }

    @model_validator(mode="before")
    @classmethod
    def _refuse_env_level_harness(cls, data):
        """Point an env-level `harness` key (the v0 muscle-memory spelling) at the
        seat that owns it, instead of a bare `extra_forbidden`. A subclass that
        declares a role named `harness` keeps the key."""
        if (
            isinstance(data, dict)
            and "harness" in data
            and "harness" not in cls.model_fields
        ):
            raise ValueError(
                "a harness belongs to a seat: --env.agent.harness.* on the "
                "single-agent env, --env.<role>.harness.* on a multi-agent role "
                "(TOML: [env.agent.harness])"
            )
        return data

    @model_validator(mode="before")
    @classmethod
    def _resolve_taskset(cls, data):
        """Narrow `taskset` to its concrete config type by `id`, so taskset-specific
        fields validate typed. Lazy import for the same reason as
        `AgentConfig._resolve_harness`."""
        if isinstance(data, dict) and data.get("taskset") is not None:
            from verifiers.v1.loaders import narrow_plugin_field, taskset_config_type

            narrow_plugin_field(data, "taskset", taskset_config_type)
        return data

    @model_validator(mode="before")
    @classmethod
    def _merge_role_defaults(cls, data):
        """Deep-merge partial role data over the field's declared default instance.
        Plain validation would replace the instance wholesale, so a partial override
        (`--env.user.sampling.temperature`) would silently reset the role's other
        pins (model, trainability)."""
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
        """A role field must carry a default *instance* (the deep-merge and the
        role machinery read it) and must not shadow a base field; refuse both at
        class definition rather than silently dropping the role. Membership is the
        role machinery's (`_merge_role_defaults`/`_declared_agent_configs`): the
        default instance, not the annotation form, is what makes a field a role."""
        super().__pydantic_init_subclass__(**kwargs)
        for name, info in cls.model_fields.items():
            is_role = isinstance(info.default, AgentConfig)
            if not is_role and not _mentions_agent_config(info.annotation):
                continue
            if is_role and name in EnvConfig.model_fields:
                raise TypeError(
                    f"{cls.__name__}.{name}: a role can't shadow the base EnvConfig "
                    f"field {name!r}; pick another role name"
                )
            if not is_role:
                raise TypeError(
                    f"{cls.__name__}.{name}: declare the role with a default "
                    f"instance (`{name}: vf.AgentConfig = vf.AgentConfig(...)`), "
                    "not default_factory or a bare annotation — the declared "
                    "instance is the role's author default (CLI overrides "
                    "deep-merge onto it, and the seat plays under the field's name)"
                )


def _declared_agent_configs(config: EnvConfig) -> dict[str, AgentConfig]:
    """The `AgentConfig` fields declared on an env's config, in declaration order —
    the env's roles, each seat keyed by its field name (the only naming site).
    Membership test (a default instance) matches `_merge_role_defaults`."""
    return {
        name: getattr(config, name)
        for name, field in type(config).model_fields.items()
        if isinstance(field.default, AgentConfig)
    }


def default_agent_harness(taskset_id: str) -> HarnessConfig:
    """What an unpinned role's `harness=None` resolves to: the taskset's bundled
    harness when it ships one, else the built-in `bash`."""
    from verifiers.v1.loaders import default_harness_id, harness_config_type

    ident = default_harness_id(taskset_id)
    return harness_config_type(ident).model_validate({"id": ident})


logger = logging.getLogger(__name__)


def _as_error(e: Exception) -> Error:
    """`e` as an episode-level `Error`. Call inside the `except` handling `e` — the
    traceback comes from the active exception context."""
    return Error(
        type=type(e).__name__, message=str(e), traceback=traceback.format_exc()
    )


@dataclass
class RunSlot:
    """One planned env-rollout of a task, observable while it happens: `traces`
    collects the current attempt's live traces (a retry restarts the list),
    `episode`/`done` land when the rollout is final. The dashboard renders slots;
    `--resume` preloads kept episodes as `finished` slots."""

    task: Task
    traces: list[Trace] = field(default_factory=list)
    episode: Episode | None = None
    done: bool = False

    @classmethod
    def finished(cls, episode: Episode) -> "RunSlot":
        return cls(
            task=Task(episode.task.data),
            traces=list(episode.traces),
            episode=episode,
            done=True,
        )


ConfigT = TypeVar("ConfigT", bound=EnvConfig)


class Environment(ABC, Generic[ConfigT]):
    """A taskset, the agents that play it, and how one task becomes one env-rollout.

    Abstract: every run gets a concrete subclass — `SingleAgentEnv` for every plain
    taskset. A multi-agent env declares each role as an `AgentConfig` field on an
    `EnvConfig` subclass (bound via `Environment[YourConfig]`, available as
    `self.config`) and writes

      - `rollout(task, agents)` — how the agents interact on one task. It returns
        nothing: every finished run joins the episode automatically, stamped with
        its seat's standing.

    Optional overrides: `brief(agents)` (per-agent standing the env hardcodes —
    a frozen judge opts out of training), `score(task, traces)` (sibling-dependent
    judgement), and `setup()`/`teardown()` (env-owned shared resources). The base
    owns everything else: agent construction, episodes, retries, persistence/
    resume, serving. Task x agent fit is validated per run, on the task the agent
    actually receives — an env-minted task carries its own needs (`tools`,
    `NEEDS_CONTAINER`), so there is nothing to declare here."""

    _stamp_roles: ClassVar[bool] = True
    """Whether traces are stamped with their seat name at mint. True for every env
    but `SingleAgentEnv`, whose sole implicit seat stays nameless — a plain eval's
    trace carries no role."""

    def __init__(self, config: ConfigT) -> None:
        from verifiers.v1.loaders import load_harness, load_taskset

        config_cls = (
            generic_type(type(self), EnvConfig, origin=Environment) or EnvConfig
        )
        if not isinstance(config, config_cls):
            raise TypeError(
                f"{type(self).__name__} declares Environment[{config_cls.__name__}], "
                f"but was handed a {type(config).__name__}; build the run config "
                "naming this env (its `env` field narrows to it), or pass "
                f"{config_cls.__name__}(...) explicitly"
            )
        self.config: ConfigT = config
        if config.taskset is None:
            raise ValueError(
                f"{type(self).__name__} needs a seed taskset — every rollout starts "
                "from one of its tasks: set --env.taskset.id (or the positional "
                "`eval <taskset-id>`)"
            )
        self.taskset = load_taskset(config.taskset)
        self._default_harness = default_agent_harness(config.taskset.id)
        task_cls = generic_type(type(self.taskset), Task, origin=Taskset) or Task
        self._task_cls: type[Task] = task_cls
        self._agent_specs: dict[str, AgentConfig] = _declared_agent_configs(self.config)
        if not self._agent_specs:
            raise ValueError(
                f"{type(self).__name__} declares no agents; declare each as an "
                "AgentConfig field on the env's config "
                "(`solver: vf.AgentConfig = vf.AgentConfig()`) — the field name is "
                "the agent's name. The single-agent case is SingleAgentEnv."
            )
        for fn in (
            *discover_decorated(self, "metric"),
            *discover_decorated(self, "reward"),
        ):
            agent = getattr(fn, "_vf_agent", None)
            if agent is not None and agent not in self._agent_specs:
                name = getattr(fn, "__name__", repr(fn))
                raise ValueError(
                    f"{type(self).__name__}.{name} is decorated with "
                    f"agent={agent!r}, but the env's config declares agents "
                    f"{sorted(self._agent_specs)}"
                )
        # Seats resolving to the same harness config share the loaded object
        # (harnesses are stateless values).
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
                and isinstance(harness.config.runtime, SubprocessConfig)
                and harness.config.id not in warned
            ):
                warned.add(harness.config.id)
                logger.warning(
                    "Harness %r is running in the subprocess runtime on the local system. "
                    "Local files and settings may affect the evaluation; use subprocess only "
                    "for debugging. Use --env.<role>.harness.runtime.type docker or prime "
                    "for an isolated run.",
                    harness.config.id,
                )
        # Eval-level serving resources, live only inside `serving()`; the env's
        # agents borrow them, so runners never thread them through `run_slot`.
        self._shared_tools: dict[str, SharedToolServer] = {}
        self._interception: Interception | None = None
        # Clients for endpoint-pinning roles, cached by config, closed with serving().
        self._agent_clients: dict[str, Client] = {}
        # Resource warnings dedupe env-wide (agents are per-episode).
        self._warned_resources: set = set()

    # --- the multi-agent surface (override these) ------------------------------

    def brief(self, agents: Agents) -> None:
        """Brief this rollout's agents before `rollout()` sees them — the in-place
        spot for per-agent standing the env hardcodes rather than exposes as
        config. Today that is `trainable` (every agent defaults True; a fixed one
        opts out: `agents.judge.trainable = False`). Agents are built fresh per
        env-rollout, so this runs once per episode — keep it cheap and in-place.
        Single-agent envs never write this."""

    @abstractmethod
    async def rollout(self, task: Task, agents: Agents) -> None:
        """One env-rollout: how the agents interact on `task` — imperative Python
        over the handed-in agents, returning nothing. Every finished run is
        captured as the episode's traces automatically, each stamped with its
        seat's standing (`agent.name`/`trainable`/`episode`/`env`). An agent-run
        failure is data on its trace (this hook decides what it means); an
        exception raised here is the env-rollout itself failing."""

    async def score(self, task: Task, traces: list[Trace]) -> None:
        """Sibling-dependent judgement over one env-rollout's finished traces
        (per-trace judgement already ran on each trace's own task). The flat list
        is the episode, completion order; each trace's `agent_name` stamp names
        its seat. The default runs the env's decorated `@vf.reward`/`@vf.metric`
        methods, each invoked once per target trace and recorded there, with
        `task`, `trace` (the target), and `traces` (all of them) in reach —
        `agent=` narrows the targets, unset means every trace. Override it for
        imperative control; `await super().score(task, traces)` keeps the
        decorated ones. Bounded by `timeout.score`."""
        metrics = discover_decorated(self, "metric")
        rewards = discover_decorated(self, "reward")

        async def run(fns) -> list[tuple]:
            pairs = [
                (fn, target)
                for fn in fns
                for target in self._signal_targets(fn, traces)
            ]
            results = await asyncio.gather(
                *(
                    invoke(
                        fn,
                        {
                            "task": task,
                            "trace": target,
                            "traces": traces,
                        },
                    )
                    for fn, target in pairs
                )
            )
            return list(zip(pairs, results))

        # Metrics record before rewards run, so a reward may read `trace.metrics`
        # (the same staging as task scoring).
        for (fn, target), result in await run(metrics):
            _record_result(target, fn.__name__, result)
        for (fn, target), result in await run(rewards):
            _record_result(target, fn.__name__, result, getattr(fn, "_vf_weight", 1.0))

    def complete(self, episode: Episode) -> bool:
        """Whether a finished env-rollout is a valid result — what `--resume` keeps
        vs. redoes. Default: `episode.ok`. An env whose `rollout()` deliberately
        tolerates a failed participant (a forfeited player) overrides this —
        typically `not episode.errors` — else resume re-runs rollouts it already
        accepted. The server eval path keeps the strict default (its env lives in
        the workers)."""
        return episode.ok

    async def setup(self) -> None:
        """Bring up env-owned shared resources. Runs inside `serving()`, after the
        framework's resources (shared tool servers, interception) are live and before
        any rollout. Default: no-op."""

    async def teardown(self) -> None:
        """Tear down what `setup()` built. Runs when `serving()` exits, even when
        `setup()` failed partway — so it must tolerate partial state. Default: no-op."""

    # --- machinery (the base owns everything below) -----------------------------

    def _agent_harness(self, agent: AgentConfig) -> HarnessConfig:
        """The harness config a role resolves to: its own pin, else the taskset's
        default (`default_agent_harness`)."""
        return agent.harness if agent.harness is not None else self._default_harness

    def _signal_targets(self, fn: Callable, traces: list[Trace]) -> list[Trace]:
        """Which traces a decorated env signal records onto: every trace unless
        `agent=` narrows it. Membership is the trace's agent-name stamp — except
        in `SingleAgentEnv`'s nameless shape, where every trace belongs to the
        sole implicit seat."""
        agent = getattr(fn, "_vf_agent", None)
        if agent is None or not self._stamp_roles:
            return list(traces)
        return [t for t in traces if t.agent_name == agent]

    def _episode_agents(
        self,
        ctx: ModelContext,
        episode_id: str,
        gate: "asyncio.Semaphore | None",
        completed: list[Trace],
        on_trace: Callable[[Trace], None] | None,
    ) -> Agents:
        """One env-rollout's `Agents`, scraped off the config — fresh value objects
        riding the live serving resources (everything expensive is env-owned and
        borrowed, so construction is cheap and no state is shared across concurrent
        episodes), briefed before `rollout()` sees them."""

        def make(name: str, spec: AgentConfig) -> Agent:
            # The episode's resolved config: every unpinned field falls back to
            # the run's own context (model, sampling) or the taskset's default
            # (harness); the live client is injected, never configured here.
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
                role=name if self._stamp_roles else None,
                episode=episode_id,
                env=self.config.env_id,
                shared_tools=self._shared_tools,
                task_cls=self._task_cls,
                gate=gate,
                completed=completed,
                on_trace=on_trace,
                warned_resources=self._warned_resources,
            )

        agents = Agents(self.config, make)
        self.brief(agents)
        return agents

    def _client_for(self, config: ClientConfig) -> Client:
        """Resolve (and cache by config) an agent-pinned endpoint's client; closed
        when `serving()` exits."""
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
    ) -> Episode:
        """One env-rollout of `task`, minted as the wire atom: run `rollout()` over the
        role agents, then `score()` over its traces (bounded by `timeout.score`).
        `gate` bounds the agent runs themselves — every run acquires it, so an env's
        internal fan-out counts against `--max-concurrent` too.

        The agents are built fresh for this episode (`_episode_agents`, briefed by
        `brief()`): every trace gets its episode standing written the moment it's
        created and joins the episode the moment its run completes (completion
        order) — `rollout()` returns nothing, and a hook that raises after some
        runs finished still yields an episode with the completed subset. A hook
        exception lands on the episode's `errors`, never on a trace."""
        episode: Episode = Episode(
            env=self.config.env_id,
            task=TraceTask(type=type(task).__name__, data=task.data),
        )
        completed: list[Trace] = []
        agents = self._episode_agents(ctx, episode.id, gate, completed, on_trace)
        try:
            async with asyncio.timeout(self.config.timeout.episode):
                async with boundary(EnvError, f"{type(self).__name__}.rollout()"):
                    await self.rollout(task, agents)
                    if not completed:
                        raise ValueError(
                            f"{type(self).__name__}.rollout() ran no agent — every "
                            "episode must carry at least one run"
                        )
        except Exception as e:
            # A TimeoutError here can only be the deadline's own expiry — one
            # raised inside rollout() became an EnvError at the boundary.
            if isinstance(e, TimeoutError):
                e = TimeoutError(
                    f"{type(self).__name__}.rollout() exceeded its "
                    f"{self.config.timeout.episode:g}s deadline (--env.timeout.episode)"
                )
            episode.errors.append(_as_error(e))
            # The completed subset is the crash-safe episode.
            episode.traces = list(completed)
            return episode
        episode.traces = list(completed)
        try:
            async with asyncio.timeout(self.config.timeout.score):
                async with boundary(EnvError, f"{type(self).__name__}.score()"):
                    await self.score(task, list(completed))
        except Exception as e:
            # A TimeoutError here can only be the deadline's own expiry — one
            # raised inside score() became an EnvError at the boundary.
            if isinstance(e, TimeoutError):
                e = TimeoutError(
                    f"{type(self).__name__}.score() exceeded its "
                    f"{self.config.timeout.score:g}s deadline (--env.timeout.score)"
                )
            episode.errors.append(_as_error(e))
        return episode

    def slots(self, task: Task, n: int = 1) -> list[RunSlot]:
        """Plan `n` independent env-rollouts of `task` — one observable `RunSlot`
        each, run to an episode by `run_slot`. `-r n` means exactly this: n episodes
        per task, nothing coupling them."""
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
        """Run one planned env-rollout to its finished episode, with whole-episode
        retries per `retries.rollout`. `semaphore` gates the agent RUNS, not the
        episode — `--max-concurrent` holds even when `rollout()` fans out
        internally. `on_complete` fires the moment the episode is final — the
        runners' persistence hook."""

        async def attempt() -> Episode:
            slot.traces = []  # a retry shows the fresh attempt's traces
            return await self.run_episode(
                slot.task, ctx, on_trace=slot.traces.append, gate=semaphore
            )

        episode = await run_episode_with_retry(attempt, self.config.retries.rollout)
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
        """Hold the env-level serving resources for the duration of an eval — shared
        tool servers, interception, and whatever `setup()` brings up. Plan and run
        slots inside; torn down on exit (`teardown()`, then the framework's)."""
        async with self.shared_tools() as shared:
            interception = make_interception(
                self.config.interception, requires_tunnel=self._requires_tunnel(shared)
            )
            async with interception:
                self._shared_tools = shared
                self._interception = interception
                try:
                    await self.setup()
                    yield
                finally:
                    try:
                        # teardown() sees the same live resources setup() saw;
                        # the framework's own unwind comes after.
                        await self.teardown()
                    finally:
                        self._shared_tools = {}
                        self._interception = None
                        clients, self._agent_clients = self._agent_clients, {}
                        for client in clients.values():
                            with contextlib.suppress(Exception):
                                await client.close()

    def _runs_local(self) -> bool:
        """Whether every role's runtime policy is local — the env-level stand-in for
        the single harness's `runtime_is_local` (any remote role means tunnels)."""
        return all(
            runtime_is_local(harness.config.runtime)
            for harness in self._harnesses.values()
        )

    def _requires_tunnel(self, shared: dict[str, SharedToolServer]) -> bool:
        """`requires_tunnel` over the consumers known before any rollout: role
        runtimes, live `shared` servers, and the task class's tool/user servers with
        their configs resolved the way `Task.server_config` resolves them. A task
        class that overrides `server_config` isn't statically knowable, so it
        conservatively counts as remote — a wrongly-assumed localhost would reach
        nothing."""
        task_cls = generic_type(type(self.taskset), Task, origin=Taskset) or Task
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
