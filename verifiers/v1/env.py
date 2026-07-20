"""Compose a taskset, its agents, and how one task becomes one env-rollout."""

import asyncio
import contextlib
import logging
import traceback
from abc import ABC, abstractmethod
from collections.abc import Awaitable, Callable, Collection, Mapping
from dataclasses import dataclass, field
from typing import (
    TYPE_CHECKING,
    Annotated,
    ClassVar,
    Generic,
    Literal,
    TypeVar,
    get_args,
)

from pydantic import Field, SerializeAsAny, ValidationError, model_validator
from pydantic_config import BaseConfig

from verifiers.v1.harness import Harness, HarnessConfig
from verifiers.v1.clients import Client, ClientConfig, ModelContext, resolve_client
from verifiers.v1.types import ID, SamplingConfig
from verifiers.v1.interception import (
    ElasticInterceptionPoolConfig,
    Interception,
    InterceptionConfig,
    make_interception,
    requires_tunnel,
)
from verifiers.v1.session import RolloutLimits
from verifiers.v1.retries import RetryConfig, run_episode_with_retry
from verifiers.v1.runtimes import (
    RuntimeConfig,
    SubprocessConfig,
    runtime_is_local,
)
from verifiers.v1.decorators import discover_decorated, invoke
from verifiers.v1.errors import EnvError, boundary
from verifiers.v1.task import Task, _record_result, resolve_server_config
from verifiers.v1.taskset import Taskset, TasksetConfig
from verifiers.v1.trace import Error, Episode, Trace, TraceTask
from verifiers.v1.utils.generic import generic_type
from verifiers.v1.utils.memory import trim_memory_periodically
from verifiers.v1.mcp import SharedToolServer, serve_shared

if TYPE_CHECKING:
    from verifiers.v1.agent import Agent


class TimeoutConfig(BaseConfig):
    """Framework-enforced wall-clock timeouts per rollout stage, in seconds (None = no
    limit)."""

    setup: float | None = None
    """Shared wall-clock budget for task setup and harness provisioning."""
    rollout: float | None = None
    """Max wall-clock for the rollout (the harness run)."""
    finalize: float | None = None
    """Max wall-clock for the task's `finalize` hook (post-run work, before scoring)."""
    scoring: float | None = None
    """Max wall-clock for task and harness scoring."""
    score: float | None = None
    """Max wall-clock for the env's cross-trace `score()` hook, run once per
    env-rollout (per-trace scoring is bounded by `scoring`)."""


class AgentConfig(BaseConfig):
    """One env role: who plays it. A role pins only what makes it a different actor;
    everything unpinned falls back — the model leg to the run's own, the harness to
    the taskset's default."""

    harness: SerializeAsAny[HarnessConfig] | None = None
    """The role's program + runtime policy (None = the taskset's default harness)."""
    model: str | None = None
    """Model id (None = the run's model, i.e. the policy under evaluation/training)."""
    client: ClientConfig | None = None
    """Endpoint override (None = the run's client) — routes a fixed role (a frozen
    judge, a pinned user sim) off the training endpoint."""
    sampling: SamplingConfig | None = None
    """Sampling override (None = the run's sampling)."""
    max_turns: int | None = None
    """Max model turns per rollout for this role (None = the env's)."""
    max_input_tokens: int | None = None
    """Max input tokens per rollout for this role (None = the env's)."""
    max_output_tokens: int | None = None
    """Max output tokens per rollout for this role (None = the env's)."""
    max_total_tokens: int | None = None
    """Max total tokens per rollout for this role (None = the env's)."""

    @model_validator(mode="before")
    @classmethod
    def _resolve_harness(cls, data):
        """Narrow a pinned `harness` to its concrete config type by `id`; an absent
        harness stays None (the taskset's default, resolved at env construction).
        The lazy import keeps class-body `AgentConfig()` defaults constructible
        while this module is still initializing."""
        if isinstance(data, dict) and data.get("harness") is not None:
            from verifiers.v1.loaders import harness_config_type, narrow_plugin_field

            narrow_plugin_field(data, "harness", harness_config_type, "bash")
        return data


def _mentions_agent_config(annotation) -> bool:
    """Whether `annotation` names an `AgentConfig` — directly or inside
    Optional/union/Annotated/container forms — i.e. anything an author plausibly
    meant as a role declaration."""
    if isinstance(annotation, type):
        return issubclass(annotation, AgentConfig)
    return any(_mentions_agent_config(arg) for arg in get_args(annotation))


def prefix_validation_error(e: ValidationError, prefix: tuple) -> ValidationError:
    """`e` with `prefix` prepended to every error's loc. A sub-model validated
    inside a `mode="before"` validator surfaces its errors at the validator's own
    loc, so without re-raising prefixed the CLI renders a flag path missing the
    segments the user actually typed."""
    return ValidationError.from_exception_data(
        e.title,
        [
            {**err, "loc": prefix + tuple(err["loc"])}
            for err in e.errors(include_url=False)
        ],
    )


def _deep_merge(base: dict, override: dict) -> dict:
    """`override` onto `base`, recursing into dicts, so a partial nested override
    keeps the untouched keys of the declared default. An override that switches a
    subtree's discriminator (`id`/`type`) replaces the subtree wholesale — the old
    plugin's fields must not leak into the new type's validation."""
    merged = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            switched = any(
                k in value and k in merged[key] and value[k] != merged[key][k]
                for k in ("id", "type")
            )
            merged[key] = value if switched else _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


class EnvConfig(BaseConfig):
    """An environment's config — the run's single `[env]` block. One subclass per
    `Environment` class (bound via `Environment[YourConfig]`, available as
    `self.config`): declare each role as an `AgentConfig` field with a default
    instance, plus any env-level knobs. The run's `env` field narrows to it by the
    env `id` (else the taskset id), which is what gives `--env.<role>.model`-style
    CLI/TOML addressing. The base carries what every environment has: which env,
    the seed taskset, and the run limits."""

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
    timeout: TimeoutConfig = TimeoutConfig()
    retries: RetryConfig = RetryConfig()
    max_concurrent: int | None = None
    """Bounds concurrent agent runs on a SERVED env, per worker — an env's internal
    fan-out counts, so best-of-n under many requests can't run unbounded (None = no
    limit). The in-process eval CLI gates with its run-level `--max-concurrent`
    instead."""
    max_turns: int | None = None
    """Max model turns per rollout (None = no limit). Framework-enforced (the
    interception server refuses turns past it), so it applies to any harness."""
    max_input_tokens: int | None = None
    """Max input (prompt) tokens per rollout (None = no limit); framework-enforced
    between turns."""
    max_output_tokens: int | None = None
    """Max output (completion) tokens per rollout (None = no limit); framework-enforced
    between turns."""
    max_total_tokens: int | None = None
    """Max total (prompt + completion) tokens per rollout (None = no limit);
    framework-enforced between turns."""
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

    def seat_harnesses(self) -> dict[str, HarnessConfig]:
        """Each declared role's resolved harness config (pin, else the taskset's
        default) — known without constructing the env, for output naming and the
        dashboard."""
        default = default_seat_harness(
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
                    data[name] = _deep_merge(
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


class SingleAgentEnvConfig(EnvConfig):
    """`SingleAgentEnv`'s config: the one `agent` seat over the seed taskset."""

    agent: AgentConfig = AgentConfig()
    """The one seat — the policy under evaluation/training; pin
    `--env.agent.harness.*` to choose its program or runtime."""


def _declared_agent_configs(config: EnvConfig) -> dict[str, AgentConfig]:
    """The `AgentConfig` fields declared on an env's config, in declaration order —
    the env's roles, each seat keyed by its field name (the only naming site).
    Membership test (a default instance) matches `_merge_role_defaults`."""
    return {
        name: getattr(config, name)
        for name, field in type(config).model_fields.items()
        if isinstance(field.default, AgentConfig)
    }


def default_seat_harness(taskset_id: str) -> HarnessConfig:
    """What an unpinned role's `harness=None` resolves to: the taskset's bundled
    harness when it ships one, else the built-in `bash`."""
    from verifiers.v1.loaders import default_harness_id, harness_config_type

    ident = default_harness_id(taskset_id)
    return harness_config_type(ident).model_validate({"id": ident})


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


def resolve_env_field(data: dict, narrowed: "type[EnvConfig] | None" = None) -> dict:
    """Shared `mode="before"` body for every run config owning an `env` field
    (`EnvServerConfig`, `GEPAConfig`): refuse the retired top-level axes with a
    pointer home, and narrow `env` to the concrete env's config class. `narrowed`
    is the annotation the CLI pre-resolved (`narrow_config`) — its id is
    authoritative, so validate against it directly."""
    if not isinstance(data, dict):
        return data
    if "taskset" in data:
        raise ValueError(
            "the taskset lives on the env now: --env.taskset.id <id> "
            "(TOML: [env.taskset]), or the positional `eval <taskset-id>`"
        )
    if "harness" in data:
        raise ValueError(
            "a harness belongs to a seat now: --env.agent.harness.* on the "
            "single-agent env, --env.<role>.harness.* on a multi-agent role "
            "(TOML: [env.agent.harness])"
        )
    raw = data.get("env")
    if raw is None:
        return data
    try:
        if narrowed is not None:
            if not isinstance(raw, narrowed):
                data["env"] = narrowed.model_validate(
                    raw.model_dump() if isinstance(raw, BaseConfig) else raw
                )
            return data
        from verifiers.v1.loaders import resolve_env_config

        data["env"] = resolve_env_config(raw)
    except ValidationError as e:
        # Validating here (inside the owner's mode="before" validator) would
        # surface the errors without their `env` segment — the CLI would render
        # `--agent.model` for the `--env.agent.model` the user typed.
        raise prefix_validation_error(e, ("env",)) from None
    return data


def _narrowed_env_annotation(cls) -> "type[EnvConfig] | None":
    """The env field's annotation when the CLI pre-narrowed it (`narrow_config`
    swaps in a concrete subclass). The base declaration reads as `EnvConfig` itself
    (SerializeAsAny unwraps), so only a proper subclass counts."""
    annotation = cls.model_fields["env"].annotation
    if (
        isinstance(annotation, type)
        and issubclass(annotation, EnvConfig)
        and annotation is not EnvConfig
    ):
        return annotation
    return None


class EnvServerConfig(BaseConfig):
    """A run's environment plus how it's *served*: the `env` block and the worker-pool
    sizing. Shared by the `serve` CLI, server-backed eval, and prime-rl's orchestrator, so
    they all configure the pool the same way (`--pool.type elastic|static`)."""

    # SerializeAsAny: see EnvConfig.taskset — model_dump() must keep the subclass's
    # role fields and knobs.
    env: SerializeAsAny[EnvConfig] = SingleAgentEnvConfig()
    """The environment — the run's `[env]` block: which env, its seed taskset, each
    seat, its knobs, and the run limits. Narrowed to the selected env's config
    class by the env id, else the taskset id."""
    pool: PoolConfig = ElasticPoolConfig()
    """Worker-pool sizing for the env server. `elastic` (default) starts at one worker and
    scales up on demand; `static` pre-spawns a fixed `num_workers`."""
    # --- legacy (v0) backwards-compat -----------------------------------------
    id: ID | None = None
    """Classic (v0) env id (`name`, `org/name`, or `org/name@version` — installed from the
    hub on demand), loaded via `verifiers.load_environment` and run through the legacy
    bridge. Set this *instead of* `env.taskset` to run a v0 environment."""
    args: dict = {}
    """Construction kwargs forwarded to `load_environment(id, **args)`."""
    extra_env_kwargs: dict = {}
    """Post-load kwargs applied to the v0 env via `env.set_kwargs(**extra_env_kwargs)` (e.g.
    `max_total_completion_tokens`, `max_seq_len`, `timeout_seconds`) — typically
    auto-populated by the orchestrator, distinct from the `args` passed at construction."""

    @property
    def is_legacy(self) -> bool:
        """A v0/legacy env (run via the bridge): a legacy `id` is set and no v1 taskset."""
        return self.id is not None and (
            self.env.taskset is None or not self.env.taskset.id
        )

    @property
    def env_id(self) -> str:
        """The run's identifier: the v1 env's (`EnvConfig.env_id`), else the legacy
        v0 env id."""
        return self.env.env_id or self.id or ""

    @model_validator(mode="after")
    def _refuse_legacy_id_with_taskset(self):
        """A legacy `id` next to a v1 `env.taskset` would be silently inert
        (`is_legacy` is False and the v0 env never loads); refuse the mix."""
        if self.id is not None and self.env.taskset is not None and self.env.taskset.id:
            raise ValueError(
                f"--id {self.id!r} is the legacy (v0) env id and can't combine with "
                f"the v1 taskset {self.env.taskset.id!r}. Pairing an env with a "
                f"taskset is --env.id {self.id!r} (TOML: id under [env]); to run the "
                "v0 env instead, drop the taskset."
            )
        return self

    # --- end legacy -----------------------------------------------------------

    @model_validator(mode="before")
    @classmethod
    def _resolve_env(cls, data):
        return resolve_env_field(data, _narrowed_env_annotation(cls))


logger = logging.getLogger(__name__)


def resolve_runtime_config(
    base: RuntimeConfig, task: Task, warned: set[tuple[str, str]] | None = None
) -> RuntimeConfig:
    """Resolve a task's runtime config from `base`: inject the task's `image` (an
    image needs a container — refuse subprocess), apply its `workdir` and requested
    `resources` where the runtime supports them. Precedence: cli/toml > task >
    default; an unsupported resource warns once (deduped via `warned`)."""
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
    for resource, value in task.data.resources.model_dump(exclude_none=True).items():
        spec = type(config).model_fields.get(resource)
        if spec is None:
            key = (config.type, resource)
            if warned is not None and key not in warned:
                warned.add(key)
                logger.warning(
                    "runtime %r doesn't support resource %r; ignoring it",
                    config.type,
                    resource,
                )
        elif (
            getattr(config, resource) == spec.default
        ):  # still the default → task may set it
            updates[resource] = value
        # else: cli/toml changed it from the default → it wins over the task
    return config.model_copy(update=updates) if updates else config


def validate_pairing(
    harness: Harness,
    task_cls: type[Task],
    runtime_config: RuntimeConfig,
    *,
    shared_tools: Collection = (),
) -> None:
    """Reject an impossible harness/task/runtime combination before any work happens.
    Every check reads class-level facts, so a failure holds for every row the task
    class can carry. `Agent.run` runs this per run as the backstop; an `Environment`
    applies the same rules role-need-aware at construction. For `shared_tools` only
    emptiness matters — declarations and live servers alike mean MCP is in play."""
    if not harness.SUPPORTS_MCP and (task_cls.tools or shared_tools):
        raise ValueError(
            f"Harness {harness.config.id!r} does not support MCP tools, but "
            f"{task_cls.__name__} exposes tool servers (MCP). Run it with a harness that "
            f"supports MCP (e.g. --env.agent.harness.id bash), or use tasks without tools."
        )
    if not harness.SUPPORTS_USER_SIM and task_cls.user is not None:
        raise ValueError(
            f"Harness {harness.config.id!r} does not drive a user simulator, but "
            f"{task_cls.__name__} defines one (Task.user). Run it with a harness that "
            f"supports user simulation (e.g. --env.agent.harness.id bash), or use tasks "
            "without one."
        )
    if task_cls.NEEDS_CONTAINER and isinstance(runtime_config, SubprocessConfig):
        raise ValueError(
            f"{task_cls.__name__} needs a container runtime (NEEDS_CONTAINER), but "
            "this run resolves to the subprocess runtime; use "
            "--env.<role>.harness.runtime.type docker or prime."
        )


def cap_remote_harness_timeout(
    harness_timeout: float | None, runtime_config: RuntimeConfig, task: Task
) -> float | None:
    """Remote sandboxes live at most 24 hours: cap the harness timeout there (with a
    warning) so a long run times out cleanly instead of the provider killing the box
    mid-run."""
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
        return 24 * 60 * 60
    return harness_timeout


def _as_error(e: Exception) -> Error:
    """`e` as an episode-level `Error`. Call inside the `except` handling `e` — the
    traceback comes from the active exception context."""
    return Error(
        type=type(e).__name__, message=str(e), traceback=traceback.format_exc()
    )


Views = Mapping[str, "Trace | list[Trace]"]
"""What `Environment.rollout` returns: the episode's local views, one entry per
participant's experience — a name to a `Trace`, or to a `list[Trace]` for a
fanned-out seat. A flat bag: names are for the author's own `score()`, nothing
more — no order, no lineage, no structure."""


def _view_traces(env_name: str, views: Views) -> list[Trace]:
    """Flatten a rollout's views into the episode's trace list (mapping order,
    list views in order — physical, not semantic), refusing anything that isn't
    a named `Trace`/`list[Trace]` bag."""
    if not isinstance(views, Mapping):
        raise TypeError(
            f"{env_name}.rollout() must return its local views as a mapping "
            f"(name -> Trace | list[Trace]), got {type(views).__name__}"
        )
    traces: list[Trace] = []
    seen: set[int] = set()
    for name, view in views.items():
        for trace in view if isinstance(view, list) else [view]:
            if not isinstance(trace, Trace):
                raise TypeError(
                    f"{env_name}.rollout() view {name!r} must be a Trace or "
                    f"list[Trace], got {type(trace).__name__}"
                )
            if (
                id(trace) in seen
            ):  # an alias view names an existing trace, not a new one
                continue
            seen.add(id(trace))
            traces.append(trace)
    if not traces:
        raise ValueError(
            f"{env_name}.rollout() returned no traces — every episode must carry "
            "at least one run; return the traces your rollout produced"
        )
    return traces


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

      - `rollout(task, agents)` — how the agents interact on one task, returning
        the episode's named local views.

    Optional overrides: `brief(agents)` (per-agent standing the env hardcodes —
    a frozen judge opts out of training), `score(task, views)` (sibling-dependent
    judgement), and `setup()`/`teardown()` (env-owned shared resources). The base
    owns everything else: agent construction, episodes, retries, persistence/
    resume, serving. Task x agent fit is validated per run, on the task the agent
    actually receives — an env-minted task carries its own needs (`tools`,
    `NEEDS_CONTAINER`), so there is nothing to declare here."""

    _stamp_roles: ClassVar[bool] = True
    """Whether traces are stamped with their role name at mint. True for every env
    but `SingleAgentEnv`, whose one unstamped trace keeps the wire identical to a
    plain eval's."""

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
        self._default_harness = default_seat_harness(config.taskset.id)
        task_cls = generic_type(type(self.taskset), Task, origin=Taskset) or Task
        self._task_cls: type[Task] = task_cls
        self._roles: dict[str, AgentConfig] = _declared_agent_configs(self.config)
        if not self._roles:
            raise ValueError(
                f"{type(self).__name__} declares no roles; declare each seat as an "
                "AgentConfig field on the env's config "
                "(`solver: vf.AgentConfig = vf.AgentConfig()`) — the field name is "
                "the role. The single-agent case is SingleAgentEnv."
            )
        for fn in (
            *discover_decorated(self, "metric"),
            *discover_decorated(self, "reward"),
        ):
            role = getattr(fn, "_vf_role", None)
            if role is not None and role not in self._roles:
                name = getattr(fn, "__name__", repr(fn))
                raise ValueError(
                    f"{type(self).__name__}.{name} is decorated with "
                    f"role={role!r}, but the env's config declares roles "
                    f"{sorted(self._roles)}"
                )
        # Seats resolving to the same harness config share the loaded object
        # (harnesses are stateless values).
        loaded: dict[str, Harness] = {}
        self._harnesses: dict[str, Harness] = {}
        for name, spec in self._roles.items():
            cfg = self._seat_harness(spec)
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
        self.limits = RolloutLimits(
            max_turns=config.max_turns,
            max_input_tokens=config.max_input_tokens,
            max_output_tokens=config.max_output_tokens,
            max_total_tokens=config.max_total_tokens,
        )
        # Eval-level serving resources, live only inside `serving()`; the env's
        # agents borrow them, so runners never thread them through `run_slot`.
        self._shared_tools: dict[str, SharedToolServer] = {}
        self._interception: Interception | None = None
        # Clients for endpoint-pinning roles, cached by config, closed with serving().
        self._role_clients: dict[str, Client] = {}
        # Resource warnings dedupe env-wide (agents are per-episode).
        self._warned_resources: set = set()

    # --- the multi-agent surface (override these) ------------------------------

    def brief(self, agents: Mapping[str, "Agent"]) -> None:
        """Brief this rollout's agents before `rollout()` sees them — the in-place
        spot for per-agent standing the env hardcodes rather than exposes as
        config. Today that is `trainable` (every agent defaults True; a fixed seat
        opts out: `agents["judge"].trainable = False`). Agents are built fresh per
        env-rollout, so this runs once per episode — keep it cheap and in-place.
        Single-agent envs never write this."""

    @abstractmethod
    async def rollout(self, task: Task, agents: Mapping[str, "Agent"]) -> Views:
        """One env-rollout: how the agents interact on `task` — imperative Python
        over the handed-in agents. Returns the episode's local views: what each
        participant experienced, named; a list value is a fanned-out seat. An
        agent-run failure is data on its trace (this hook decides what it means);
        an exception raised here is the env-rollout itself failing."""

    async def score(self, task: Task, views: Views) -> None:
        """Sibling-dependent judgement over one env-rollout's finished views
        (per-trace judgement already ran on each trace's own task). The default runs
        the env's decorated `@vf.reward`/`@vf.metric` methods, each invoked once per
        target trace and recorded there, with `task`, `trace` (the target), `traces`
        (all of them), and `views` in reach — `role=` narrows the targets, unset
        means every trace. Override it for imperative control; `await
        super().score(task, views)` keeps the decorated ones. Bounded by
        `timeout.score`."""
        traces = _view_traces(type(self).__name__, views)
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
                            "views": views,
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

    def _seat_harness(self, agent: AgentConfig) -> HarnessConfig:
        """The harness config a role resolves to: its own pin, else the taskset's
        default (`default_seat_harness`)."""
        return agent.harness if agent.harness is not None else self._default_harness

    def _signal_targets(self, fn: Callable, traces: list[Trace]) -> list[Trace]:
        """Which traces a decorated env signal records onto: every trace unless
        `role=` narrows it. Membership is the role stamp — except in
        `SingleAgentEnv`'s unstamped shape, where every trace belongs to the sole
        implicit role."""
        role = getattr(fn, "_vf_role", None)
        if role is None or not self._stamp_roles:
            return list(traces)
        return [t for t in traces if t.role == role]

    def _episode_agents(
        self,
        ctx: ModelContext,
        gate: "asyncio.Semaphore | None",
        completed: list[Trace],
        on_trace: Callable[[Trace], None] | None,
    ) -> dict[str, "Agent"]:
        """One env-rollout's agents, one per role — fresh value objects riding the
        live serving resources (everything expensive is env-owned and borrowed, so
        construction is cheap and no state is shared across concurrent episodes),
        briefed before `rollout()` sees them."""
        from verifiers.v1.agent import _EpisodeAgent  # env <-> agent import cycle

        agents: dict[str, Agent] = {}
        for name, spec in self._roles.items():
            role_ctx = self._role_ctx(spec, ctx)
            agents[name] = _EpisodeAgent(
                self._harnesses[name],
                role_ctx.model,
                role_ctx.client,
                sampling=role_ctx.sampling,
                interception=self._interception,
                limits=self._role_limits(spec),
                timeout=self.config.timeout,
                name=name,
                role=name if self._stamp_roles else None,
                shared_tools=self._shared_tools,
                task_cls=self._task_cls,
                gate=gate,
                completed=completed,
                on_trace=on_trace,
                warned_resources=self._warned_resources,
            )
        self.brief(agents)
        return agents

    def _role_ctx(self, spec: AgentConfig, ctx: ModelContext) -> ModelContext:
        """The role's model leg: the run's `ctx` unless the role pins its own model,
        endpoint, or sampling (each falls back to the run's independently)."""
        if spec.model is None and spec.client is None and spec.sampling is None:
            return ctx
        return ModelContext(
            model=spec.model if spec.model is not None else ctx.model,
            client=self._client_for(spec.client)
            if spec.client is not None
            else ctx.client,
            sampling=spec.sampling if spec.sampling is not None else ctx.sampling,
        )

    def _client_for(self, config: ClientConfig) -> Client:
        """Resolve (and cache by config) a role-pinned endpoint's client; closed when
        `serving()` exits."""
        key = config.model_dump_json()
        if key not in self._role_clients:
            self._role_clients[key] = resolve_client(config)
        return self._role_clients[key]

    def _role_limits(self, spec: AgentConfig) -> RolloutLimits:
        """The role's per-rollout limits: each cap the role's own when set, else the
        env's."""
        return RolloutLimits(
            max_turns=spec.max_turns
            if spec.max_turns is not None
            else self.limits.max_turns,
            max_input_tokens=spec.max_input_tokens
            if spec.max_input_tokens is not None
            else self.limits.max_input_tokens,
            max_output_tokens=spec.max_output_tokens
            if spec.max_output_tokens is not None
            else self.limits.max_output_tokens,
            max_total_tokens=spec.max_total_tokens
            if spec.max_total_tokens is not None
            else self.limits.max_total_tokens,
        )

    async def run_episode(
        self,
        task: Task,
        ctx: ModelContext,
        *,
        on_trace: Callable[[Trace], None] | None = None,
        gate: asyncio.Semaphore | None = None,
    ) -> Episode:
        """One env-rollout of `task`, minted as the wire atom: run `rollout()` over the
        role agents, then `score()` over its views (bounded by `timeout.score`).
        `gate` bounds the agent runs themselves — every run acquires it, so an env's
        internal fan-out counts against `--max-concurrent` too.

        The agents are built fresh for this episode (`_episode_agents`, briefed by
        `brief()`): every trace gets `role`/`trainable` written the moment it's
        created and is captured the moment its run completes — a `rollout()` that
        raises after some runs finished still yields an episode with the completed
        subset. Once `rollout()` returns, its views decide membership, kept even
        when `score()` then fails. A hook exception lands on the episode's
        `errors`, never on a trace."""
        completed: list[Trace] = []
        agents = self._episode_agents(ctx, gate, completed, on_trace)
        episode: Episode = Episode(
            env=self.config.env_id,
            task=TraceTask(type=type(task).__name__, data=task.data),
        )
        try:
            async with boundary(EnvError, f"{type(self).__name__}.rollout()"):
                views = await self.rollout(task, agents)
                traces = _view_traces(type(self).__name__, views)
        except Exception as e:
            episode.errors.append(_as_error(e))
            # The hook never returned views: keep the crash-safe completed subset.
            episode.traces = list(completed)
            return episode
        # Membership is set before scoring, so a score() failure can't demote the
        # episode to the completed buffer.
        episode.traces = traces
        try:
            async with asyncio.timeout(self.config.timeout.score):
                async with boundary(EnvError, f"{type(self).__name__}.score()"):
                    await self.score(task, views)
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
                        clients, self._role_clients = self._role_clients, {}
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


class SingleAgentEnv(Environment[SingleAgentEnvConfig]):
    """The single-agent case — the env every plain taskset resolves to: one `agent`
    seat playing the seed taskset (`--env.agent.*`). Its one trace per episode stays
    unstamped, so the wire is identical to a plain eval's."""

    _stamp_roles = False

    def __init__(self, config: SingleAgentEnvConfig) -> None:
        super().__init__(config)
        # The one seat definitionally plays the seed taskset, so an impossible
        # pairing is knowable from class facts alone — refuse at construction,
        # before any work (multi-agent envs validate per run instead, on the
        # task each agent actually receives).
        harness = self._harnesses["agent"]
        validate_pairing(
            harness,
            self._task_cls,
            harness.config.runtime,
            shared_tools=type(self.taskset).tools,
        )

    async def rollout(self, task: Task, agents: Mapping[str, "Agent"]) -> Views:
        return {"agent": await agents["agent"].run(task)}
