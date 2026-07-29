"""The `[legacy]` block: running a classic (v0) environment through the bridge.

Quarantined in its own block, not mixed into `[env]` — a v0 env is a different
thing to run, not a variant of a v1 env's config."""

from pydantic import Field
from pydantic_config import BaseConfig

from verifiers.v1.configs.env import EnvConfig
from verifiers.v1.types import ID


class LegacyEnvConfig(BaseConfig):
    """A classic (v0) `verifiers` environment, loaded via `verifiers.load_environment`
    and run through the legacy bridge. Set `id` *instead of* a v1 `env.taskset`."""

    id: ID | None = None
    """v0 env id: `name`, `org/name`, or `org/name@version` — installed from the hub on
    demand."""
    args: dict = Field(default_factory=dict)
    """Construction kwargs forwarded to `load_environment(id, **args)`."""
    extra_env_kwargs: dict = Field(default_factory=dict)
    """Post-load kwargs applied via `env.set_kwargs(**extra_env_kwargs)` (e.g.
    `max_total_completion_tokens`, `max_seq_len`, `timeout_seconds`) — typically
    auto-populated by a trainer, distinct from the `args` passed at construction."""


def is_legacy(env: EnvConfig, legacy: LegacyEnvConfig) -> bool:
    """Whether this run goes through the v0 bridge: a legacy id is set and no v1 taskset."""
    return legacy.id is not None and not env.taskset.id


def run_env_id(env: EnvConfig, legacy: LegacyEnvConfig) -> str:
    """The run's identifier: the v1 env's (`EnvConfig.env_id`), else the v0 env id."""
    return env.env_id or legacy.id or ""


def refuse_mixed_run(env: EnvConfig, legacy: LegacyEnvConfig) -> None:
    """Refuse a v0 id next to any v1 env identity — one of the two would go nowhere,
    and which one depends on `is_legacy`: a taskset makes it False, so the v0 env never
    loads; a bare `--env.id` leaves it True, so the v0 env runs under the v1 env's name."""
    if legacy.id is None or not env.env_id:
        return
    if env.taskset.id:
        raise ValueError(
            f"--legacy.id {legacy.id!r} is a classic (v0) env and can't combine with "
            f"the v1 taskset {env.taskset.id!r}. Pairing a reusable env with a taskset "
            f"is --env.id {legacy.id!r} (TOML: id under [env]); to run the v0 env "
            "instead, drop the taskset."
        )
    raise ValueError(
        f"--legacy.id {legacy.id!r} is a classic (v0) env and can't combine with the "
        f"v1 env --env.id {env.id!r}: the v0 env is what would run, stamped with the "
        "v1 env's name. Keep whichever one you meant to run."
    )
