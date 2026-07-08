"""The environment: a taskset composed with an harness and a runtime.

The Environment is the eval-level composition and *resolver* — it does not itself run
rollouts. It holds the taskset (the loader), harness, runtime config, and timeouts; lists
the tasks; and turns one task into a runnable `Episode` of `n` `Rollout`s, resolving per
task the runtime (image + resources, with cli/task/default precedence) and the timeouts.
Execution lives one level down: an `Episode` runs `n` `Rollout`s of a task and scores them
(per-rollout `@reward`/`@metric`, then cross-rollout `@group_reward`); each `Rollout`
runs one trajectory. The task's `@reward`/`@metric` get the rollout's runtime
(read/exec inside it), so a task scores correctly under any harness; `@group_reward`s
compare a task's rollouts. Harness capability checks (tools, user sim, container) are
per task, by value — one task class per taskset, but its instances can still differ
row to row (`tools()`/`user()` read the task's fields).
"""

import contextlib
import logging
from typing import Annotated, Literal

from pydantic import Field, SerializeAsAny, model_validator
from pydantic_config import BaseConfig

from verifiers.v1.harness import HarnessConfig
from verifiers.v1.clients import ModelContext
from verifiers.v1.decorators import discover_decorated
from verifiers.v1.episode import Episode
from verifiers.v1.types import ID
from verifiers.v1.interception import InterceptionPool, RolloutLimits
from verifiers.v1.retries import RetryConfig
from verifiers.v1.rollout import Rollout
from verifiers.v1.runtimes import (
    RuntimeConfig,
    SubprocessConfig,
    runtime_is_local,
)
from verifiers.v1.task import Task
from verifiers.v1.taskset import TasksetConfig
from verifiers.v1.mcp import Toolset, serve_shared


class TimeoutConfig(BaseConfig):
    """Framework-enforced wall-clock timeouts per rollout stage, in seconds (None = no
    limit). Each bounds one stage of `Rollout.run`: the task's `setup` hook, the harness
    run, the task's `finalize` hook, then scoring."""

    setup: float | None = None
    """Max wall-clock for the task's `setup` hook (per-task runtime prep)."""
    rollout: float | None = None
    """Max wall-clock for the rollout (the harness run)."""
    finalize: float | None = None
    """Max wall-clock for the task's `finalize` hook (post-run work, before scoring)."""
    scoring: float | None = None
    """Max wall-clock for scoring — verify + rewards/metrics."""


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
    inferred from `load_taskset`). Tool-server placement lives on `Task.tools`."""

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


def resolve_runtime_config(
    base: RuntimeConfig, task: Task, warned: set[tuple[str, str]] | None = None
) -> RuntimeConfig:
    """Resolve a task's runtime config from a `base`: inject the task's `image` (a task with
    an image must run in a container — refuse subprocess), and apply its `workdir` and
    requested `resources` to the fields the runtime supports. Precedence is cli/toml > task >
    default; a resource the runtime doesn't support warns once (deduped via `warned`). Shared
    by `Environment.runtime_for` (rollouts) and the `validate` entrypoint."""
    config = base
    updates: dict = {}
    if task.image is not None:
        if isinstance(config, SubprocessConfig):
            raise ValueError(
                f"task {task.idx!r} requires image {task.image!r}, but the subprocess "
                "runtime has no container; use the docker or prime runtime"
            )
        updates["image"] = task.image
    workdir_spec = type(config).model_fields.get("workdir")
    if (
        task.workdir is not None
        and workdir_spec is not None
        and getattr(config, "workdir") == workdir_spec.default
    ):
        updates["workdir"] = task.workdir
    for field, value in task.resources.model_dump(exclude_none=True).items():
        spec = type(config).model_fields.get(field)
        if spec is None:
            key = (config.type, field)
            if warned is not None and key not in warned:
                warned.add(key)
                logger.warning(
                    "runtime %r doesn't support resource %r; ignoring it",
                    config.type,
                    field,
                )
        elif (
            getattr(config, field) == spec.default
        ):  # still the default → task may set it
            updates[field] = value
        # else: cli/toml changed it from the default → it wins over the task
    return config.model_copy(update=updates) if updates else config


class Environment:
    def __init__(self, config: EnvConfig) -> None:
        from verifiers.v1.loaders import load_harness, load_taskset

        self.config = config
        self.taskset = load_taskset(config.taskset)
        self.harness = load_harness(config.harness)
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
        self.setup_timeout = config.timeout.setup
        self.harness_timeout = config.timeout.rollout
        self.finalize_timeout = config.timeout.finalize
        self.scoring_timeout = config.timeout.scoring
        self.limits = RolloutLimits(
            max_turns=config.max_turns,
            max_input_tokens=config.max_input_tokens,
            max_output_tokens=config.max_output_tokens,
            max_total_tokens=config.max_total_tokens,
        )
        self._warned_resources: set[tuple[str, str]] = set()
        self._shared_urls: dict[str, str] = {}
        self._interception: InterceptionPool | None = None
        """Eval-level serving resources, live only inside `serving()`: shared tool servers
        ({name: url}) and the interception pool. `episode()` injects them into every rollout
        so neither runner has to thread them through `Episode.run`/`Rollout.run`."""

    def runtime_for(self, task: Task) -> RuntimeConfig:
        """Resolve the runtime config for a task off the harness's runtime (see
        `resolve_runtime_config`)."""
        return resolve_runtime_config(
            self.harness.config.runtime, task, self._warned_resources
        )

    def episode(self, task: Task, ctx: ModelContext, n: int = 1) -> Episode:
        """Resolve `task` into a runnable episode of `n` rollouts: check the harness
        supports what this task needs (tools / user sim / a container — per task, by
        value, since instances differ per row), pick its runtime (image + resources) and its
        timeouts (cli/toml > task > default, None = no limit), build one `Rollout` per
        sample sharing them, and wrap them in an `Episode` (which runs them and applies
        the task's `@group_reward`s across their traces).

        A task with `@group_reward`s compares its rollouts, so it needs >=2 of
        them — refuse `n < 2` there (rather than silently scoring a group of one)."""
        # Capability checks are by value: `tools()`/`user()` read the task's fields, so
        # an override may yield nothing for this row — a class-override check would be
        # cheaper but wrongly reject those. The short-circuit means the calls only run
        # when the harness lacks the capability, and they construct (never launch) the
        # declared servers; the real per-rollout instances are built in `Rollout.run`.
        if not self.harness.SUPPORTS_MCP and task.tools():
            raise ValueError(
                f"Harness {self.harness.config.id!r} does not support MCP tools, but task "
                f"{task.idx!r} ({type(task).__name__}) exposes tool servers (MCP). Run it with "
                f"a harness that supports MCP (e.g. --harness.id default), or use tasks without tools."
            )
        if not self.harness.SUPPORTS_USER_SIM and task.user() is not None:
            raise ValueError(
                f"Harness {self.harness.config.id!r} does not drive a user simulator, but task "
                f"{task.idx!r} ({type(task).__name__}) defines one (Task.user). Run it with a "
                f"harness that supports user simulation (e.g. --harness.id default), or use tasks without one."
            )
        if task.NEEDS_CONTAINER and isinstance(
            self.harness.config.runtime, SubprocessConfig
        ):
            raise ValueError(
                f"Task {task.idx!r} ({type(task).__name__}) needs a container runtime "
                "(NEEDS_CONTAINER), but the harness runs on the subprocess runtime; "
                "use --harness.runtime.type docker or prime."
            )
        if n < 2 and discover_decorated(task, "group_reward"):
            raise ValueError(
                f"task {task.idx!r} defines @group_reward(s), which compare a task's rollouts "
                f"and need >=2; got n={n} (pass -r/--num-rollouts >= 2)"
            )
        runtime_config = self.runtime_for(task)
        setup_timeout = (
            self.setup_timeout if self.setup_timeout is not None else task.timeout.setup
        )
        harness_timeout = (
            self.harness_timeout
            if self.harness_timeout is not None
            else task.timeout.harness
        )
        if (
            harness_timeout is not None
            and harness_timeout > 24 * 60 * 60
            and not runtime_is_local(runtime_config)
        ):
            logger.warning(
                "task %r resolves to a %.1f-hour harness timeout, but %s sandboxes have a "
                "maximum lifetime of 24 hours; capping it at 24 hours",
                task.idx,
                harness_timeout / (60 * 60),
                runtime_config.type,
            )
            harness_timeout = 24 * 60 * 60
        finalize_timeout = (
            self.finalize_timeout
            if self.finalize_timeout is not None
            else task.timeout.finalize
        )
        scoring_timeout = (
            self.scoring_timeout
            if self.scoring_timeout is not None
            else task.timeout.scoring
        )
        retries = self.config.retries
        rollouts = [
            Rollout(
                task=task,
                harness=self.harness,
                ctx=ctx,
                runtime_config=runtime_config,
                setup_timeout=setup_timeout,
                harness_timeout=harness_timeout,
                finalize_timeout=finalize_timeout,
                scoring_timeout=scoring_timeout,
                limits=self.limits,
                shared_urls=self._shared_urls,
                interception=self._interception,
            )
            for _ in range(n)
        ]
        return Episode(rollouts, retry=retries.rollout)

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
        data to every rollout); calling `tools()` here only builds the toolset instances. Every
        task is swept — one task class per taskset, but `tools()` reads the instance's fields,
        so rows can expose different servers (a representative-task check would miss them; the
        sweep is by value, runs once per eval, and constructs without launching) — deduped by
        server name (same name = the same task-agnostic server, started once). A shared
        server on a host runtime is bridged to the host once (a tunnel) when the harness runs
        remotely, so an in-sandbox harness can still reach it (see `serve_shared`)."""
        servers: dict[str, Toolset] = {}
        for task in tasks:
            for server in task.tools():
                if not server.config.shared:
                    continue
                seen = servers.get(server.server_name)
                if seen is None:
                    servers[server.server_name] = server
                # The name is the routing key (`shared_urls[name]`), so same-named shared
                # servers must be the same server — refuse to silently route two different
                # ones to whichever started.
                elif type(seen) is not type(server) or seen.config != server.config:
                    raise ValueError(
                        f"shared tool server name {server.server_name!r} is claimed by two "
                        f"different servers ({type(seen).__name__} / {type(server).__name__}); "
                        f"every rollout routes to one eval-level server per name, so "
                        f"same-named shared servers must be identical — rename one"
                    )
        if not servers:
            yield {}
            return
        harness_is_local = runtime_is_local(self.harness.config.runtime)
        async with serve_shared(
            list(servers.values()), harness_is_local=harness_is_local
        ) as urls:
            yield urls
