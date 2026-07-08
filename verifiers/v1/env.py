"""The environment: a taskset composed with an harness and a runtime.

The Environment is the eval-level composition and *resolver* — it does not itself run
rollouts. It holds the taskset (the task factory), harness, runtime config, and timeouts;
lists the tasks; and turns one task into a runnable `Episode` of `n` `Rollout`s, resolving
per task the runtime (image + resources, with cli/task/default precedence) and the timeouts. Execution
lives one level down: an `Episode` runs `n` `Rollout`s of a task and scores them
(per-rollout `@reward`/`@metric`, then cross-rollout `@group_reward`); each `Rollout`
runs one trajectory. The task class's `@reward`/`@metric` get the rollout's runtime
(read/exec inside it), so a task scores correctly under any harness; `@group_reward`s
compare a task's rollouts.
"""

import contextlib
import logging
from typing import TYPE_CHECKING, Annotated, Literal

if TYPE_CHECKING:
    from verifiers.v1.harness import Harness

from pydantic import Field, SerializeAsAny, model_validator
from pydantic_config import BaseConfig

from verifiers.v1.clients import ModelContext
from verifiers.v1.decorators import discover_decorated
from verifiers.v1.episode import Episode
from verifiers.v1.harness import HarnessConfig
from verifiers.v1.interception import InterceptionPool, RolloutLimits
from verifiers.v1.mcp import SharedServers
from verifiers.v1.retries import RetryConfig
from verifiers.v1.rollout import Rollout
from verifiers.v1.runtimes import (
    RuntimeConfig,
    SubprocessConfig,
    runtime_is_local,
)
from verifiers.v1.services import RunServices
from verifiers.v1.task import Task
from verifiers.v1.taskset import TasksetConfig
from verifiers.v1.types import ID


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
    """The rollout's two peers: the taskset (the task factory) and the harness (which
    program drives it, and where it runs — `harness.runtime`). Both are chosen at eval
    time, not by the env — only `taskset` is narrowed per env (to its config type,
    inferred from `load_taskset`). Tool-server placement lives on each task's `tools`."""

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

    @property
    def limits(self) -> RolloutLimits:
        """The per-rollout budgets (`max_turns` + token caps) as the `RolloutLimits` the
        interception server enforces — shared by every runner that builds rollouts from
        this config (the `Environment` and a `TopologyRunner`)."""
        return RolloutLimits(
            max_turns=self.max_turns,
            max_input_tokens=self.max_input_tokens,
            max_output_tokens=self.max_output_tokens,
            max_total_tokens=self.max_total_tokens,
        )

    @model_validator(mode="before")
    @classmethod
    def _resolve_plugins(cls, data):
        """Resolve the generic `taskset` / `harness` to its specific config type by `id`, so
        env-specific fields validate against the real plugin config (no untyped args dict).
        A taskset that bundles its own harness runs with it by default; an explicit
        `--harness.id` / toml id (already on the field) takes precedence."""
        from verifiers.v1.loaders import narrow_taskset_and_harness

        return narrow_taskset_and_harness(data)


class EnvServerConfig(EnvConfig):
    """An env plus how it's *served*: the env definition (`EnvConfig`) and the worker-pool
    sizing. Shared by the `serve` CLI, server-backed eval, and prime-rl's orchestrator, so
    they all configure the pool the same way (`--pool.type elastic|static`)."""

    pool: PoolConfig = ElasticPoolConfig()
    """Worker-pool sizing for the env server. `elastic` (default) starts at one worker and
    scales up on demand; `static` pre-spawns a fixed `num_workers`."""


logger = logging.getLogger(__name__)


def validate_pairing(
    task_cls: type[Task],
    harness: "Harness",
    runtime_config: RuntimeConfig | None = None,
) -> None:
    """Refuse an incompatible task × harness pair up front: MCP tools / a user simulator
    under a harness that can't drive them, or a container-only task class on the subprocess
    runtime. Also warns when a code-running harness executes on the local host. Class-level
    (override detection), so it runs before any data is loaded — at Environment load via the
    factory's declared task type, and once per `(task class, harness)` for a topology's
    derived tasks (memoized at rollout construction). When an Agent places a rollout into
    a borrowed runtime, `runtime_config` is that actual placement rather than the agent's
    default harness runtime."""
    actual_runtime = runtime_config or harness.config.runtime
    if not harness.SUPPORTS_MCP and task_cls.load_tools is not Task.load_tools:
        raise ValueError(
            f"Harness {harness.config.id!r} does not support MCP tools, but task "
            f"{task_cls.__name__!r} exposes tool servers (MCP). Run it with a harness "
            f"that supports MCP (e.g. --harness.id default), or use a task without tools."
        )
    if not harness.SUPPORTS_USER_SIM and task_cls.load_user is not Task.load_user:
        raise ValueError(
            f"Harness {harness.config.id!r} does not drive a user simulator, but task "
            f"{task_cls.__name__!r} defines one (Task.load_user). Run it with a harness that "
            f"supports user simulation (e.g. --harness.id default), or use a task without one."
        )
    if task_cls.NEEDS_CONTAINER and isinstance(actual_runtime, SubprocessConfig):
        raise ValueError(
            f"Task {task_cls.__name__!r} needs a container runtime "
            "(NEEDS_CONTAINER), but this run resolves to the subprocess runtime; "
            "use --harness.runtime.type docker or prime."
        )
    # The warning is about the *agent* running arbitrary code on the host: every harness hands
    # it local execution (bash/edit, or a CLI agent) except the tool-less `null` chat loop,
    # whose program only relays the model and remote MCP tools — so exempt `null`, warn for the
    # rest. (`null` still runs its fixed chat-loop program locally, but nothing agent-authored.)
    if harness.config.id != "null" and isinstance(actual_runtime, SubprocessConfig):
        logger.warning(
            "Harness %r is running in the subprocess runtime on the local system. "
            "Local files and settings may affect the evaluation; use subprocess only "
            "for debugging. Use --harness.runtime.type docker or prime for an isolated "
            "run.",
            harness.config.id,
        )


def resolve_stage_timeouts(
    timeout: TimeoutConfig, task: Task, runtime_config: RuntimeConfig
) -> TimeoutConfig:
    """Resolve a task's per-stage wall-clock budgets into a concrete `TimeoutConfig`:
    cli/toml > task > default (None = no limit), with remote sandbox lifetimes capping the
    harness stage at 24 hours. Shared by `Environment.episode` and the topology's rollouts."""
    harness = timeout.rollout if timeout.rollout is not None else task.timeout.harness
    if (
        harness is not None
        and harness > 24 * 60 * 60
        and not runtime_is_local(runtime_config)
    ):
        logger.warning(
            "task %r resolves to a %.1f-hour harness timeout, but %s sandboxes have a "
            "maximum lifetime of 24 hours; capping it at 24 hours",
            task.idx,
            harness / (60 * 60),
            runtime_config.type,
        )
        harness = 24 * 60 * 60
    return TimeoutConfig(
        setup=timeout.setup if timeout.setup is not None else task.timeout.setup,
        rollout=harness,
        finalize=timeout.finalize
        if timeout.finalize is not None
        else task.timeout.finalize,
        scoring=timeout.scoring
        if timeout.scoring is not None
        else task.timeout.scoring,
    )


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
        from verifiers.v1.loaders import load_harness, load_taskset, taskset_task_type

        self.config = config
        self.taskset = load_taskset(config.taskset)
        self.harness = load_harness(config.harness)
        # The factory's generic declares its task class, so the pairing is checked before
        # any dataset is loaded.
        validate_pairing(taskset_task_type(type(self.taskset)), self.harness)
        self.limits = config.limits
        self._warned_resources: set[tuple[str, str]] = set()
        self._shared: SharedServers | None = None
        self._interception: InterceptionPool | None = None
        """Run-level serving resources, live only inside `serving()`: the lazy shared
        tool-server registry and the interception pool. `episode()` injects them into every
        rollout so neither runner has to thread them through `Episode.run`/`Rollout.run`."""

    def runtime_for(self, task: Task) -> RuntimeConfig:
        """Resolve the runtime config for a task off the harness's runtime (see
        `resolve_runtime_config`)."""
        return resolve_runtime_config(
            self.harness.config.runtime, task, self._warned_resources
        )

    def episode(self, task: Task, ctx: ModelContext, n: int = 1) -> Episode:
        """Resolve `task` into a runnable episode of `n` rollouts: pick its runtime
        (image + resources) and its timeouts (cli/toml > task > default, None = no limit),
        build one `Rollout` per sample sharing them, and wrap them in an `Episode` (which
        runs them and applies the task's `@group_reward`s across their traces).

        A task with `@group_reward`s compares its rollouts, so it needs >=2 of
        them — refuse `n < 2` there (rather than silently scoring a group of one)."""
        if n < 2 and discover_decorated(task, "group_reward"):
            raise ValueError(
                f"task defines @group_reward(s), which compare a task's rollouts and "
                f"need >=2; got n={n} (pass -r/--num-rollouts >= 2)"
            )
        runtime_config = self.runtime_for(task)
        timeouts = resolve_stage_timeouts(self.config.timeout, task, runtime_config)
        retries = self.config.retries
        rollouts = [
            Rollout(
                task=task,
                harness=self.harness,
                ctx=ctx,
                runtime_config=runtime_config,
                timeouts=timeouts,
                limits=self.limits,
                shared=self._shared,
                interception=self._interception,
            )
            for _ in range(n)
        ]
        return Episode(rollouts, retry=retries.rollout)

    @contextlib.asynccontextmanager
    async def serving(self):
        """Hold the run-level serving resources for the duration of an eval: the lazy
        shared tool-server registry (each `shared` server starts on first use, deduped —
        see `SharedServers`) and the interception pool. Stash them so every `episode()`
        built inside this context injects them into its rollouts — that's what keeps both
        eval runners (in-process and env-server) on one serving path. Build episodes inside
        this context; the resources are torn down on exit."""
        async with RunServices(self.config.multiplex) as services:
            self._shared = services.shared
            self._interception = await services.pool_for(self.harness.config.runtime)
            try:
                yield
            finally:
                self._shared = None
                self._interception = None
