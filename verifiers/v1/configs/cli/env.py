"""Run-config plumbing around the `[env]` block: narrowing the `env` field of every
config that owns one, and the retired keys such a config refuses.

A run composes the blocks it needs — `[env]` (what runs, `configs/env.py`),
`[serve]` (how it's hosted, `configs/serve.py`), `[legacy]` (the v0 bridge,
`configs/legacy.py`) — plus its own fields. Nothing here is a base class: the eval
CLI, the `serve` CLI, GEPA and a trainer each declare their blocks and call these.

Declare the env field as `SerializeAsAny[EnvConfig] = Field(default_factory=
single_agent_env_config)`. The `SerializeAsAny` is load-bearing: pydantic serializes
by declared type, so a plain `EnvConfig` silently drops a narrowed subclass's agents
and knobs from `model_dump()` — the env-server wire's payload."""

from pydantic import ValidationError
from pydantic_config import BaseConfig

from verifiers.v1.configs.env import EnvConfig
from verifiers.v1.utils.generic import prefix_validation_error


def resolve_env_field(data: dict, narrowed: "type[EnvConfig] | None" = None) -> dict:
    """Shared `mode="before"` body for every run config owning an `env` field: narrow
    `env` to the concrete env's config class. `narrowed` is the annotation the CLI
    pre-resolved (`narrow_config`) — its id is authoritative, so validate against it
    directly."""
    if not isinstance(data, dict):
        return data
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
        from verifiers.v1.utils.loaders import resolve_env_config

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
