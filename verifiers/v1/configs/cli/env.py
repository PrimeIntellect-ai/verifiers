"""Run-config plumbing around the `[env]` block: narrowing the `env` field of every
config that owns one, and the retired keys such a config refuses.

A run composes the blocks it needs — `[env]` (what runs, `configs/env.py`),
`[serve]` (how it's hosted, `configs/serve.py`), `[legacy]` (the v0 bridge,
`configs/legacy.py`) — plus its own fields. Nothing here is a base class: the eval
CLI, the `serve` CLI, GEPA and a trainer each declare their blocks and call these."""

from pydantic import Field, ValidationError
from pydantic_config import BaseConfig

from verifiers.v1.configs.env import EnvConfig
from verifiers.v1.utils.generic import prefix_validation_error

RETIRED = {
    "taskset": "the taskset lives on the env: --env.taskset.id <id> (TOML: "
    "[env.taskset]), or the positional `eval <taskset-id>`",
    "harness": "a harness belongs to an agent: --env.agent.harness.* on the "
    "single-agent env, --env.<agent>.harness.* on a multi-agent one (TOML: "
    "[env.agent.harness])",
    "pool": "the worker pool is serving, not the env: --serve.pool.type "
    "elastic|static (TOML: [serve.pool])",
    "address": "the bind address is serving, not the env: --serve.address "
    "(TOML: address under [serve])",
    "id": "a classic (v0) env is its own block: --legacy.id <env-id>; pairing a "
    "reusable v1 env with a taskset is --env.id",
    "args": "v0 construction kwargs live with the v0 env: --legacy.args",
    "extra_env_kwargs": "v0 post-load kwargs live with the v0 env: "
    "--legacy.extra-env-kwargs",
}
"""Top-level keys a run config no longer owns, each pointing at its home. Every one
would otherwise fail as a bare `extra_forbidden`, saying nothing about where it went."""


def env_field() -> Field:  # type: ignore[valid-type]
    """The default for the `env` field every run config declares: the single-agent
    shape. Annotate the field `SerializeAsAny[EnvConfig]` — pydantic serializes by
    declared type, so a plain `EnvConfig` silently drops a narrowed subclass's agents
    and knobs from `model_dump()`, which is the env-server wire's payload."""
    return Field(default_factory=_single_agent_env_config)


def resolve_env_field(data: dict, narrowed: "type[EnvConfig] | None" = None) -> dict:
    """Shared `mode="before"` body for every run config owning an `env` field: refuse
    the retired top-level keys with a pointer home, and narrow `env` to the concrete
    env's config class. `narrowed` is the annotation the CLI pre-resolved
    (`narrow_config`) — its id is authoritative, so validate against it directly."""
    if not isinstance(data, dict):
        return data
    for key, pointer in RETIRED.items():
        if key in data:
            raise ValueError(pointer)
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


def _single_agent_env_config() -> EnvConfig:
    """The default `env` block: the single-agent shape. Lazy — the concrete env
    lives in `envs/`, which imports `env.py`."""
    from verifiers.v1.envs.single_agent import SingleAgentEnvConfig

    return SingleAgentEnvConfig()
