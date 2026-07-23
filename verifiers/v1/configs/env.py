"""An environment's config — the run's single `[env]` block: the seed taskset,
each role as an `AgentConfig` field, and the env-level knobs."""

from typing import get_args

from pydantic import SerializeAsAny, model_validator
from pydantic_config import BaseConfig

from verifiers.v1.configs.agent import AgentConfig
from verifiers.v1.configs.harness import HarnessConfig
from verifiers.v1.configs.retries import RetryConfig
from verifiers.v1.configs.taskset import TasksetConfig
from verifiers.v1.interception import ElasticInterceptionPoolConfig, InterceptionConfig
from verifiers.v1.types import ID
from verifiers.v1.utils.generic import deep_merge


class TimeoutConfig(BaseConfig):
    """Wall-clock timeouts for the env's own `run()`/`finalize()` hooks, in seconds
    (None = no limit); per-run stage timeouts are each agent's (`--env.<agent>.timeout.*`)."""

    episode: float | None = None
    finalize: float | None = None


def _mentions_agent_config(annotation) -> bool:
    """Whether `annotation` names an `AgentConfig` — directly or inside
    Optional/union/Annotated/container forms."""
    if isinstance(annotation, type):
        return issubclass(annotation, AgentConfig)
    return any(_mentions_agent_config(arg) for arg in get_args(annotation))


class EnvConfig(BaseConfig):
    """An environment's config — the run's single `[env]` block, one subclass per
    `Env` class (bound via `Env[YourConfig]`): each role is an
    `AgentConfig` field with a default instance, plus env-level knobs. The run's `env`
    field narrows to it by env `id` (else taskset id) — `--env.<role>.model` addressing."""

    id: ID = ""
    """Which `Env` runs. Empty = the taskset's own, else `SingleAgentEnv`; set
    to pair a reusable env with any taskset (an explicit id wins over the bundled)."""
    # SerializeAsAny: the env-server wire needs the resolved subclass's fields.
    taskset: SerializeAsAny[TasksetConfig] = TasksetConfig()
    """The seed taskset — the rows every rollout starts from (`--env.taskset.id`).
    The id stays empty only for a legacy (v0) run, which sets the top-level `id`."""
    timeout: TimeoutConfig = TimeoutConfig()
    retries: RetryConfig = RetryConfig()
    """Whole-EPISODE retries — the coarse fallback for faults no agent owns; a
    retried episode reruns whole (a half-played sibling context isn't reproducible)."""
    max_concurrent: int | None = None
    """Bounds concurrent agent runs on a SERVED env, per worker (None = no limit);
    the in-process eval CLI gates with its run-level `--max-concurrent` instead."""
    interception: InterceptionConfig = ElasticInterceptionPoolConfig()
    """The interception shape: `elastic` (default), `server`, or `static`."""

    @property
    def env_id(self) -> str:
        """The taskset id, prefixed by the paired env id (`best-of-n+gsm8k-v1`)."""
        if self.taskset.id and self.id:
            return f"{self.id}+{self.taskset.id}"
        return self.taskset.id or self.id

    def agent_harnesses(self) -> dict[str, HarnessConfig]:
        """Each declared role's resolved harness config (pin, else the taskset's
        default) — known without constructing the env."""
        default = default_agent_harness(self.taskset.id)
        return {
            name: cfg.harness if cfg.harness is not None else default
            for name, cfg in _declared_agent_configs(self).items()
        }

    @model_validator(mode="before")
    @classmethod
    def _refuse_env_level_harness(cls, data):
        """Point an env-level `harness` key (v0 muscle memory) at the seat that owns
        it; a subclass declaring a role named `harness` keeps the key."""
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
        """Narrow `taskset` to its concrete config type by `id`; lazy import for
        the same reason as `AgentConfig._resolve_harness`."""
        if isinstance(data, dict) and data.get("taskset") is not None:
            from verifiers.v1.loaders import narrow_plugin_field, taskset_config_type

            narrow_plugin_field(data, "taskset", taskset_config_type)
        return data

    @model_validator(mode="before")
    @classmethod
    def _merge_role_defaults(cls, data):
        """Deep-merge partial role data over the field's declared default — plain
        validation would replace the instance wholesale, resetting its other pins."""
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
        """Refuse, at class definition, a role field without a default *instance*
        or one shadowing a base field — the default instance is what makes a field a role."""
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
    the env's roles, each seat keyed by its field name (the only naming site)."""
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
