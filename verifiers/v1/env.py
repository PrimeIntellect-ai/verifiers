"""Native environment configuration and execution-policy helpers."""

import logging
from typing import Annotated, Literal

from pydantic import Field, SerializeAsAny, model_validator
from pydantic_config import BaseConfig

from verifiers.v1.harness import Harness, HarnessConfig
from verifiers.v1.interception import (
    ElasticInterceptionPoolConfig,
    InterceptionConfig,
)
from verifiers.v1.retries import RetryConfig
from verifiers.v1.runtimes import (
    RuntimeConfig,
    SubprocessConfig,
    runtime_is_local,
)
from verifiers.v1.task import Task, resolve_server_config
from verifiers.v1.taskset import Taskset, TasksetConfig
from verifiers.v1.topology import TopologyConfig
from verifiers.v1.types import ID
from verifiers.v1.utils.generic import generic_type


class TimeoutConfig(BaseConfig):
    """Framework-enforced wall-clock timeouts per rollout stage, in seconds (None = no
    limit). Each bounds one stage of `Rollout.run`: task and harness setup, the harness
    run, the task's `finalize` hook, then scoring."""

    setup: float | None = None
    """Shared wall-clock budget for task setup and harness provisioning."""
    rollout: float | None = None
    """Max wall-clock for the rollout (the harness run)."""
    finalize: float | None = None
    """Max wall-clock for the task's `finalize` hook (post-run work, before scoring)."""
    scoring: float | None = None
    """Max wall-clock for task and harness scoring."""


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
    """Graph requests per worker for the scale-up trigger: add a worker once in-flight requests
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
    """What to run: a taskset × harness pair (lowered into the built-in single-agent
    topology), or an explicit multi-agent `topology`."""

    # SerializeAsAny: these hold resolved subclasses (e.g. MathConfig, DefaultHarnessConfig);
    # without it model_dump() narrows to the base type and drops the subclass fields, so the
    # env-server subconfig the orchestrator writes would lose taskset/harness-specific knobs.
    taskset: SerializeAsAny[TasksetConfig] = TasksetConfig()
    harness: SerializeAsAny[HarnessConfig] = HarnessConfig(id="default")
    topology: SerializeAsAny[TopologyConfig] | None = None
    """A multi-agent topology to run *instead of* the single `taskset` × `harness` pair
    (see `verifiers.v1.topology`), selected by `--topology.id`. Seeds come from its
    `taskset` slot (`--topology.taskset.id <id>`); each agent binds its own
    harness/routing (`--topology.<agent>.harness.id`, ...)."""
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
    interception: InterceptionConfig = ElasticInterceptionPoolConfig()
    """The interception shape (see `verifiers.v1.interception`): `elastic` (the
    default — servers grown on demand, `multiplex` rollouts each), `server` (one server,
    with a tunnel choice incl. a bring-your-own endpoint), or `static` (a fixed list of
    such servers)."""
    # --- legacy (v0) backwards-compat -----------------------------------------
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
        """Resolve the generic `topology` / `taskset` / `harness` to its specific config
        type by `id`, so env-specific fields validate against the real plugin config (no
        untyped args dict). Topology first, before the default harness is manufactured
        below — so a *user-supplied* `harness` alongside a topology is still
        distinguishable, and refused (it would be silently ignored; agents bind their
        own harnesses)."""
        from verifiers.v1.loaders import (
            default_harness_id,
            harness_config_type,
            narrow_plugin_field,
            taskset_config_type,
            topology_config_type,
        )

        if isinstance(data, dict) and data.get("topology"):
            if data.get("harness"):
                raise ValueError(
                    "`--harness.*` is ignored under a topology — each agent binds its "
                    "own harness (`--topology.<agent>.harness.*`); drop the flag."
                )
            narrow_plugin_field(data, "topology", topology_config_type)
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

    @model_validator(mode="after")
    def _check_topology(self):
        """A topology replaces the config's own taskset × harness pair — reject
        combinations that would silently ignore half the config."""
        if self.topology is not None and (self.taskset.id or self.id):
            raise ValueError(
                "`--topology.id` runs the topology's own agents; drop `--taskset.id` / "
                "`--id` (choose the seed tasks via --topology.taskset.id)."
            )
        return self


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
    by topology agents and the `validate` entrypoint."""
    config = base
    updates: dict = {}
    if task.data.image is not None:
        if isinstance(config, SubprocessConfig):
            raise ValueError(
                f"task {task.data.idx!r} requires image {task.data.image!r}, but the subprocess "
                "runtime has no container; use the docker or prime runtime"
            )
        updates["image"] = task.data.image
    workdir_spec = type(config).model_fields.get("workdir")
    if (
        task.data.workdir is not None
        and workdir_spec is not None
        and getattr(config, "workdir") == workdir_spec.default
    ):
        updates["workdir"] = task.data.workdir
    for field, value in task.data.resources.model_dump(exclude_none=True).items():
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


def resolve_stage_timeouts(
    timeout: TimeoutConfig, task: Task, runtime_config: RuntimeConfig
) -> TimeoutConfig:
    """Resolve a task's per-stage wall-clock budgets into a concrete `TimeoutConfig`:
    cli/toml > task > default (None = no limit), with remote sandbox lifetimes capping
    the harness stage at 24 hours."""
    harness = (
        timeout.rollout if timeout.rollout is not None else task.data.timeout.harness
    )
    if (
        harness is not None
        and harness > 24 * 60 * 60
        and not runtime_is_local(runtime_config)
    ):
        logger.warning(
            "task %r resolves to a %.1f-hour harness timeout, but %s sandboxes have a "
            "maximum lifetime of 24 hours; capping it at 24 hours",
            task.data.idx,
            harness / (60 * 60),
            runtime_config.type,
        )
        harness = 24 * 60 * 60
    return TimeoutConfig(
        setup=timeout.setup if timeout.setup is not None else task.data.timeout.setup,
        rollout=harness,
        finalize=timeout.finalize
        if timeout.finalize is not None
        else task.data.timeout.finalize,
        scoring=timeout.scoring
        if timeout.scoring is not None
        else task.data.timeout.scoring,
    )


def validate_task_pairing(
    harness: Harness,
    task_cls: type[Task],
    runtime_config: RuntimeConfig,
    shared_tools: tuple = (),
) -> None:
    """Reject an impossible harness/task-class/runtime combination. Every check reads
    class-level facts (`Task.tools` / `Task.user` / `NEEDS_CONTAINER`, plus any
    taskset-scoped `shared_tools`), so a failure holds for every instance of the class.
    `runtime_config` is where the run actually lands — the harness's policy at
    construction (`validate_pairing`), the *resolved* config per run for topology agents
    (a borrowed box may differ from the harness's default, in either direction)."""
    if not harness.SUPPORTS_MCP and (task_cls.tools or shared_tools):
        raise ValueError(
            f"Harness {harness.config.id!r} does not support MCP tools, but "
            f"{task_cls.__name__} exposes tool servers (MCP). Run it with a harness that "
            f"supports MCP (e.g. --harness.id default), or use tasks without tools."
        )
    if not harness.SUPPORTS_USER_SIM and task_cls.user is not None:
        raise ValueError(
            f"Harness {harness.config.id!r} does not drive a user simulator, but "
            f"{task_cls.__name__} defines one (Task.user). Run it with a harness that "
            f"supports user simulation (e.g. --harness.id default), or use tasks without one."
        )
    if task_cls.NEEDS_CONTAINER and isinstance(runtime_config, SubprocessConfig):
        raise ValueError(
            f"{task_cls.__name__} needs a container runtime (NEEDS_CONTAINER), but this "
            "run resolves to the subprocess runtime; use --harness.runtime.type docker or prime."
        )


def validate_pairing(harness: Harness, taskset: Taskset) -> None:
    """Reject an impossible harness/taskset pairing at construction — before any dataset
    load, shared-server launch, or first agent run (see `validate_task_pairing`; one task
    type per taskset, read off the `Taskset[TaskT, ...]` generic). On the env server this
    fails worker startup instead of every request."""
    task_cls = generic_type(type(taskset), Task, origin=Taskset) or Task
    validate_task_pairing(
        harness, task_cls, harness.config.runtime, shared_tools=type(taskset).tools
    )


def taskset_server_configs(taskset: Taskset) -> list[BaseConfig] | None:
    """The taskset's statically-knowable tool/user server configs — the task class's
    servers (read class-level, like `validate_pairing`) with their configs resolved the
    way `Task.server_config` resolves them — for the run-wide `requires_tunnel` verdict.
    `None` means not statically knowable: the task class *overrides* that pairing, so it
    conservatively counts as remote (the tunnel then reaches everything; a wrongly-assumed
    localhost would reach nothing remote)."""
    task_cls = generic_type(type(taskset), Task, origin=Taskset) or Task
    server_classes = [*task_cls.tools, *([task_cls.user] if task_cls.user else [])]
    if server_classes and task_cls.server_config is not Task.server_config:
        return None
    sole = len({*task_cls.tools} | ({task_cls.user} - {None})) == 1
    return [
        resolve_server_config(
            task_cls.__name__, taskset.config.task, server_cls, sole=sole
        )
        for server_cls in server_classes
    ]
