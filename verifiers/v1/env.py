"""The environment: a program of agent interactions over a taskset.

The Environment is the eval-level composition — it does not itself run rollouts. It
holds the taskset and the timeouts/limits config, lists the tasks, and turns one task
into a runnable `Episode`. Its program is the simplest one: a single `Agent` (the
configured harness, the caller's model context, the harness's runtime policy) run `n`
times on the task — every episode's rollouts are built through that internal Agent, so
an eval run and a hand-written agent program resolve placement, timeouts, and pairing
identically. Execution lives one level down: an `Episode` runs `n` `Rollout`s of a task
and scores them (per-rollout `@reward`/`@metric`, then cross-rollout `@group_reward`);
each `Rollout` runs one trajectory. The taskset's `@reward`/`@metric` get the rollout's
runtime (read/exec inside it), so a task scores correctly under any harness;
`@group_reward`s compare a task's rollouts.
"""

import contextlib
import logging
from typing import Annotated, Literal

from pydantic import Field, SerializeAsAny, model_validator
from pydantic_config import BaseConfig

from verifiers.v1.agent import Agent
from verifiers.v1.harness import HarnessConfig
from verifiers.v1.clients import ModelContext
from verifiers.v1.decorators import discover_decorated
from verifiers.v1.episode import Episode
from verifiers.v1.types import ID
from verifiers.v1.interception import InterceptionPool, RolloutLimits
from verifiers.v1.resolve import TimeoutConfig, validate_pairing
from verifiers.v1.retries import RetryConfig
from verifiers.v1.runtimes import (
    SubprocessConfig,
    runtime_is_local,
)
from verifiers.v1.task import Task
from verifiers.v1.taskset import TasksetConfig
from verifiers.v1.mcp import serve_shared


class StaticPoolConfig(BaseConfig):
    """Fixed env-server pool: pre-spawn `num_workers` workers up front."""

    type: Literal["static"] = "static"
    num_workers: int = Field(4, ge=1)
    """Worker processes to pre-spawn (1 = a single in-process server, no pool)."""


class ElasticPoolConfig(BaseConfig):
    """Elastic env-server pool: start at one worker and scale up on demand."""

    type: Literal["elastic"] = "elastic"
    max_workers: int | None = None
    """Upper bound on workers (None = unbounded)."""
    multiplex: int = Field(128, ge=1)
    """Rollouts per worker for the scale-up trigger: add a worker once in-flight rollouts
    reach 90% of `workers * multiplex`."""


# Discriminated on `type` so the CLI selects with `--pool.type static|elastic`.
PoolConfig = Annotated[
    StaticPoolConfig | ElasticPoolConfig, Field(discriminator="type")
]


def pool_serve_kwargs(pool: StaticPoolConfig | ElasticPoolConfig) -> dict:
    """Unpack a pool config into `serve_env` kwargs (`max_workers` / `multiplex` / `elastic`)."""
    if isinstance(pool, ElasticPoolConfig):
        return {
            "max_workers": pool.max_workers,
            "multiplex": pool.multiplex,
            "elastic": True,
        }
    return {"max_workers": pool.num_workers, "elastic": False}


class EnvConfig(BaseConfig):
    """The rollout's two peers: the taskset (data + scoring) and the harness (which
    program drives it, and where it runs — `harness.runtime`). Both are chosen at eval
    time, not by the env — only `taskset` is narrowed per env (to its config type,
    inferred from `load_taskset`). Tool-server placement lives on `taskset.tools`."""

    # SerializeAsAny: these hold resolved subclasses (e.g. MathConfig, DefaultHarnessConfig);
    # without it model_dump() narrows to the base type and drops the subclass fields, so the
    # env-server subconfig the orchestrator writes would lose taskset/harness-specific knobs.
    taskset: SerializeAsAny[TasksetConfig] = TasksetConfig()
    harness: SerializeAsAny[HarnessConfig] = HarnessConfig(id="default")
    timeout: TimeoutConfig = TimeoutConfig()
    retries: RetryConfig = RetryConfig()
    max_turns: int | None = None
    """Max model turns per rollout (None = no limit). Enforced by the framework (the
    interception server refuses turns past it), so it applies to any harness — turn
    capping is a framework concern, never an harness or task field."""
    max_input_tokens: int | None = None
    """Max input (prompt) tokens per rollout (None = no limit). Caps the trace's
    `num_input_tokens`; framework-enforced between turns."""
    max_output_tokens: int | None = None
    """Max output (completion) tokens per rollout (None = no limit). Caps the trace's
    `num_output_tokens`; framework-enforced between turns."""
    max_total_tokens: int | None = None
    """Max total (prompt + completion) tokens per rollout (None = no limit). Caps the
    trace's `num_total_tokens`; framework-enforced between turns."""
    multiplex: int = Field(32, ge=1)
    """Rollouts that share one interception server (and, behind a remote runtime, one
    tunnel). N concurrent rollouts use ~N/multiplex servers + tunnels instead of one each —
    key past the per-token tunnel cap. 1 = a server (+ tunnel) per rollout."""
    # --- legacy (v0) backwards-compat -----------------------------------------
    # Run a classic `verifiers.load_environment(id, **args)` env, bridged to v1 Traces (see
    # `verifiers.v1.legacy`), instead of a v1 taskset/harness. Set `id` (leave `taskset`
    # unset) to opt in; native v1 envs leave these untouched. Mirrors prime-rl's EnvConfig
    # so it inherits these (a v0 env is driven the same way in eval and the env server).
    id: ID | None = None
    """Classic (v0) env id (`name`, `org/name`, or `org/name@version` — installed from the
    hub on demand), loaded via `verifiers.load_environment` and run through the legacy
    bridge. Set this *instead of* `taskset` to run a v0 environment."""
    args: dict = {}
    """Construction kwargs forwarded to `load_environment(id, **args)`."""
    extra_env_kwargs: dict = {}
    """Post-load kwargs applied to the v0 env via `env.set_kwargs(**extra_env_kwargs)` (e.g.
    `max_total_completion_tokens`, `max_seq_len`, `timeout_seconds`) — typically
    auto-populated by the orchestrator, distinct from the `args` passed at construction."""

    @property
    def is_legacy(self) -> bool:
        """A v0/legacy env (run via the bridge): a legacy `id` is set and no v1 `taskset`."""
        return self.id is not None and not self.taskset.id

    @property
    def env_id(self) -> str:
        """The env identifier — the v1 taskset id, else the legacy v0 env id."""
        return self.taskset.id or self.id or ""

    # --- end legacy -----------------------------------------------------------

    @model_validator(mode="before")
    @classmethod
    def _resolve_plugins(cls, data):
        """Resolve the generic `taskset` / `harness` to its specific config type by `id`, so
        env-specific fields validate against the real plugin config (no untyped args dict)."""
        from verifiers.v1.loaders import (
            default_harness_id,
            harness_config_type,
            narrow_plugin_field,
            taskset_config_type,
        )

        narrow_plugin_field(data, "taskset", taskset_config_type)
        taskset = data.get("taskset")
        taskset_id = (
            taskset.get("id")
            if isinstance(taskset, dict)
            else getattr(taskset, "id", None)
        )
        # A taskset that bundles its own harness runs with it by default; an explicit
        # `--harness.id` / toml id (already on the field) takes precedence.
        narrow_plugin_field(
            data, "harness", harness_config_type, default_harness_id(taskset_id or "")
        )
        return data


class EnvServerConfig(EnvConfig):
    """An env plus how it's *served*: the env definition (`EnvConfig`) and the worker-pool
    sizing. Shared by the `serve` CLI, server-backed eval, and prime-rl's orchestrator, so
    they all configure the pool the same way (`--pool.type elastic|static`)."""

    pool: PoolConfig = ElasticPoolConfig()
    """Worker-pool sizing for the env server. `elastic` (default) starts at one worker and
    scales up on demand; `static` pre-spawns a fixed `num_workers`."""


logger = logging.getLogger(__name__)


class Environment:
    def __init__(self, config: EnvConfig) -> None:
        from verifiers.v1.loaders import load_harness, load_taskset

        self.config = config
        self.taskset = load_taskset(config.taskset)
        self.harness = load_harness(config.harness)
        validate_pairing(self.harness, self.taskset, self.harness.config.runtime)
        # The warning is about the *agent* running arbitrary code on the host: every harness hands
        # it local execution (bash/edit, or a CLI agent) except the tool-less `null` chat loop,
        # whose program only relays the model and remote MCP tools — so exempt `null`, warn for the
        # rest. (`null` still runs its fixed chat-loop program locally, but nothing agent-authored.)
        if self.harness.config.id != "null" and isinstance(
            self.harness.config.runtime, SubprocessConfig
        ):
            logger.warning(
                "Harness %r is running in the subprocess runtime on the local system. "
                "Local files and settings may affect the evaluation; use subprocess only "
                "for debugging. Use --harness.runtime.type docker or prime for an isolated "
                "run.",
                self.harness.config.id,
            )
        self.limits = RolloutLimits(
            max_turns=config.max_turns,
            max_input_tokens=config.max_input_tokens,
            max_output_tokens=config.max_output_tokens,
            max_total_tokens=config.max_total_tokens,
        )
        self._agent: Agent | None = None
        """The env's internal Agent — the single-agent program every episode runs. Built
        on the first `episode()` (an Agent binds a `ModelContext` at construction, and the
        env only sees one then); each episode passes its own `ctx` per rollout, so a cached
        agent never pins a stale model."""
        self._shared_urls: dict[str, str] = {}
        self._interception: InterceptionPool | None = None
        """Eval-level serving resources, live only inside `serving()`: shared tool servers
        ({name: url}) and the interception pool. `episode()` injects them into every rollout
        so neither runner has to thread them through `Episode.run`/`Rollout.run`."""

    def agent(self, ctx: ModelContext) -> Agent:
        """The env's Agent: the configured harness with the harness's runtime policy and
        the env's timeouts/limits. One instance backs every episode (its resolution state —
        e.g. warn dedup — spans the eval); `ctx` only seeds the first construction."""
        if self._agent is None:
            self._agent = Agent(
                self.harness, ctx, limits=self.limits, timeout=self.config.timeout
            )
        return self._agent

    def episode(self, task: Task, ctx: ModelContext, n: int = 1) -> Episode:
        """Resolve `task` into a runnable episode of `n` rollouts of the env's agent —
        each rollout resolved by `Agent.rollout` (runtime from image/resources, timeouts
        cli/toml > task > default, pairing validation) with the env's serving resources
        injected — and wrap them in an `Episode` (which runs them and applies the
        taskset's `@group_reward`s across their traces).

        A taskset with `@group_reward`s compares a task's rollouts, so it needs >=2 of
        them — refuse `n < 2` there (rather than silently scoring a group of one)."""
        if n < 2 and discover_decorated(self.taskset, "group_reward"):
            raise ValueError(
                f"taskset defines @group_reward(s), which compare a task's rollouts and "
                f"need >=2; got n={n} (pass -r/--num-rollouts >= 2)"
            )
        agent = self.agent(ctx)
        rollouts = [
            agent.rollout(
                task,
                taskset=self.taskset,
                ctx=ctx,
                shared_urls=self._shared_urls,
                interception=self._interception,
            )
            for _ in range(n)
        ]
        return Episode(rollouts, self.taskset, retry=self.config.retries.rollout)

    @contextlib.asynccontextmanager
    async def serving(self, tasks: list[Task]):
        """Hold the env-level serving resources for the duration of an eval: the shared tool
        servers (built once, see `shared_tools`) and the interception pool. Stash them so
        every `episode()` built inside this context injects them into its rollouts — that's
        what keeps both eval runners (in-process and env-server) on one serving path. Build
        episodes inside this context; the resources are torn down on exit."""
        async with (
            self.shared_tools(tasks) as shared_urls,
            self.interception_pool() as interception,
        ):
            self._shared_urls = shared_urls
            self._interception = interception
            try:
                yield
            finally:
                self._shared_urls = {}
                self._interception = None

    def interception_pool(self) -> InterceptionPool:
        """The shared interception pool for this env's rollouts — one server (+ tunnel
        behind a remote runtime) per `multiplex` rollouts, grown on demand. Built here,
        where the harness runtime and `multiplex` live; the caller (eval runner / env
        server) enters it for the run and tears it down. Pass it to `Episode.run`."""
        return InterceptionPool(self.harness.config.runtime, self.config.multiplex)

    @contextlib.asynccontextmanager
    async def shared_tools(self, tasks: list[Task]):
        """Start any tool servers whose placement is `shared` ONCE for the eval (each in its
        own `runtime`) and yield `{name: url}` to inject into every rollout — so an expensive
        corpus is built once, not per rollout. No-op ({}) when none are shared. A shared server
        must be task-agnostic: its `setup` gets no task (so it can't silently serve one task's
        data to every rollout); `tools(tasks[0])` here only builds the toolset instances. A shared
        server on a host runtime is bridged to the host once (a tunnel) when the harness runs
        remotely, so an in-sandbox harness can still reach it (see `serve_shared`)."""
        servers = self.taskset.tools(tasks[0]) if tasks else []
        if not any(server.config.shared for server in servers):
            yield {}
            return
        harness_is_local = runtime_is_local(self.harness.config.runtime)
        async with serve_shared(servers, harness_is_local=harness_is_local) as urls:
            yield urls
