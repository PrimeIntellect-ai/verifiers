"""Run-config plumbing around the `[env]` block: narrowing the `env` field of the
configs that own one, and how a served env sizes its worker pool."""

from typing import Annotated, Literal

from pydantic import Field, SerializeAsAny, ValidationError, model_validator
from pydantic_config import BaseConfig

from verifiers.v1.env import EnvConfig
from verifiers.v1.types import ID
from verifiers.v1.utils.generic import prefix_validation_error


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
            "a harness belongs to an agent now: --env.agent.harness.* on the "
            "single-agent env, --env.<agent>.harness.* on a multi-agent one "
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


def narrowed_env_annotation(cls) -> "type[EnvConfig] | None":
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


def _single_agent_env_config() -> EnvConfig:
    """The default `env` block: the single-agent shape. Lazy — the concrete env
    lives in `envs/`, which imports `env.py`."""
    from verifiers.v1.envs.single_agent import SingleAgentEnvConfig

    return SingleAgentEnvConfig()


class EnvServerConfig(BaseConfig):
    """A run's environment plus how it's *served*: the `env` block and the worker-pool
    sizing. Shared by the `serve` CLI, server-backed eval, and prime-rl's orchestrator, so
    they all configure the pool the same way (`--pool.type elastic|static`)."""

    # SerializeAsAny: see EnvConfig.taskset — model_dump() must keep the subclass's
    # agent fields and knobs.
    env: SerializeAsAny[EnvConfig] = Field(default_factory=_single_agent_env_config)
    """The environment — the run's `[env]` block: which env, its seed taskset, each
    agent, its knobs, and the run limits. Narrowed to the selected env's config
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
        return self.id is not None and not self.env.taskset.id

    @property
    def env_id(self) -> str:
        """The run's identifier: the v1 env's (`EnvConfig.env_id`), else the legacy
        v0 env id."""
        return self.env.env_id or self.id or ""

    @model_validator(mode="after")
    def _refuse_legacy_id_with_taskset(self):
        """A legacy `id` next to a v1 `env.taskset` would be silently inert
        (`is_legacy` is False and the v0 env never loads); refuse the mix."""
        if self.id is not None and self.env.taskset.id:
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
        return resolve_env_field(data, narrowed_env_annotation(cls))
