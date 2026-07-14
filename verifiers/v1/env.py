"""Compose a taskset and harness into runnable episodes."""

import contextlib
import logging
from typing import Annotated, Literal

from pydantic import Field, SerializeAsAny, model_validator
from pydantic_config import BaseConfig

from verifiers.v1.harness import Harness, HarnessConfig
from verifiers.v1.clients import ModelContext
from verifiers.v1.decorators import discover_decorated
from verifiers.v1.episode import Episode
from verifiers.v1.types import ID
from verifiers.v1.interception import (
    ElasticInterceptionPoolConfig,
    Interception,
    InterceptionConfig,
    make_interception,
    requires_tunnel,
)
from verifiers.v1.session import RolloutLimits
from verifiers.v1.retries import RetryConfig
from verifiers.v1.rollout import Rollout
from verifiers.v1.runtimes import (
    RuntimeConfig,
    SubprocessConfig,
    runtime_is_local,
)
from verifiers.v1.task import Task, resolve_server_config
from verifiers.v1.taskset import Taskset, TasksetConfig
from verifiers.v1.utils.generic import generic_type
from verifiers.v1.mcp import SharedToolServer, serve_shared


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
    """The taskset that loads tasks and the harness that runs them."""

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


def validate_pairing(harness: Harness, taskset: Taskset) -> None:
    """Reject an impossible harness/taskset pairing at construction — before any dataset
    load, shared-server launch, or first episode. Every check reads class-level facts
    (`Task.tools` / `Task.user` / `NEEDS_CONTAINER`, `Taskset.tools` — one task type per
    taskset, read off the `Taskset[TaskT, ...]` generic), so a failure here holds for
    every row the taskset can produce; on the env server it fails worker startup instead
    of every request."""
    task_cls = generic_type(type(taskset), Task, origin=Taskset) or Task
    if not harness.SUPPORTS_MCP and (task_cls.tools or type(taskset).tools):
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
    if task_cls.NEEDS_CONTAINER and isinstance(
        harness.config.runtime, SubprocessConfig
    ):
        raise ValueError(
            f"{task_cls.__name__} needs a container runtime (NEEDS_CONTAINER), but the "
            "harness runs on the subprocess runtime; use --harness.runtime.type docker or prime."
        )


class Environment:
    def __init__(self, config: EnvConfig) -> None:
        from verifiers.v1.loaders import load_harness, load_taskset

        self.config = config
        self.taskset = load_taskset(config.taskset)
        self.harness = load_harness(config.harness)
        validate_pairing(self.harness, self.taskset)
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
        self._shared_tools: dict[str, SharedToolServer] = {}
        self._interception: Interception | None = None
        """Eval-level serving resources, live only inside `serving()`: shared tool servers
        ({name: SharedToolServer}) and the interception. `episode()` injects them into every rollout
        so neither runner has to thread them through `Episode.run`/`Rollout.run`."""

    def runtime_for(self, task: Task) -> RuntimeConfig:
        """Resolve the runtime config for a task off the harness's runtime (see
        `resolve_runtime_config`)."""
        return resolve_runtime_config(
            self.harness.config.runtime, task, self._warned_resources
        )

    def episode(self, task: Task, ctx: ModelContext, n: int = 1) -> Episode:
        """Resolve `task` into a runnable episode of `n` rollouts: pick its runtime
        (image + resources) and its
        timeouts (cli/toml > task > default, None = no limit), build one `Rollout` per
        sample sharing them, and wrap them in an `Episode` (which runs them and applies
        the task's `@group_reward`s across their traces).

        A task with `@group_reward`s compares its rollouts, so it needs >=2 of
        them — refuse `n < 2` there (rather than silently scoring a group of one).
        Harness capability (tools / user sim / container) is class-level and already
        checked at construction (`validate_pairing`)."""
        if n < 2 and discover_decorated(task, "group_reward"):
            raise ValueError(
                f"task {task.data.idx!r} defines @group_reward(s), which compare a task's rollouts "
                f"and need >=2; got n={n} (pass -r/--num-rollouts >= 2)"
            )
        runtime_config = self.runtime_for(task)
        setup_timeout = (
            self.setup_timeout
            if self.setup_timeout is not None
            else task.data.timeout.setup
        )
        harness_timeout = (
            self.harness_timeout
            if self.harness_timeout is not None
            else task.data.timeout.harness
        )
        if (
            harness_timeout is not None
            and harness_timeout > 24 * 60 * 60
            and not runtime_is_local(runtime_config)
        ):
            logger.warning(
                "task %r resolves to a %.1f-hour harness timeout, but %s sandboxes have a "
                "maximum lifetime of 24 hours; capping it at 24 hours",
                task.data.idx,
                harness_timeout / (60 * 60),
                runtime_config.type,
            )
            harness_timeout = 24 * 60 * 60
        finalize_timeout = (
            self.finalize_timeout
            if self.finalize_timeout is not None
            else task.data.timeout.finalize
        )
        scoring_timeout = (
            self.scoring_timeout
            if self.scoring_timeout is not None
            else task.data.timeout.scoring
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
                shared_tools=self._shared_tools,
                interception=self._interception,
            )
            for _ in range(n)
        ]
        return Episode(rollouts, retry=retries.rollout)

    @contextlib.asynccontextmanager
    async def serving(self, *, warm_interception: bool = False):
        """Hold the env-level serving resources for the duration of an eval: the shared tool
        servers (built once, see `shared_tools`) and the interception. Stash them so
        every `episode()` built inside this context injects them into its rollouts — that's
        what keeps both eval runners (in-process and env-server) on one serving path. Build
        episodes inside this context; the resources are torn down on exit."""
        async with self.shared_tools() as shared:
            interception = make_interception(
                self.config.interception,
                requires_tunnel=self._requires_tunnel(shared),
                warm=warm_interception,
            )
            async with interception:
                self._shared_tools = shared
                self._interception = interception
                try:
                    yield
                finally:
                    self._shared_tools = {}
                    self._interception = None

    def _requires_tunnel(self, shared: dict[str, SharedToolServer]) -> bool:
        """`requires_tunnel` over the consumers known before any rollout: the harness
        runtime (by config), the live `shared` servers, and the task class's tool/user
        servers — read class-level (like `validate_pairing`) with their configs resolved
        the way `Task.server_config` resolves them. A task that *overrides* that pairing
        isn't statically knowable, so it conservatively counts as remote (the tunnel then
        reaches everything; a wrongly-assumed localhost would reach nothing remote)."""
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
        return requires_tunnel(
            runtime_is_local(self.harness.config.runtime), configs, shared.values()
        )

    @contextlib.asynccontextmanager
    async def shared_tools(self):
        servers = self.taskset.tool_servers()
        if not servers:
            yield {}
            return
        harness_is_local = runtime_is_local(self.harness.config.runtime)
        async with serve_shared(servers, harness_is_local=harness_is_local) as shared:
            yield shared
