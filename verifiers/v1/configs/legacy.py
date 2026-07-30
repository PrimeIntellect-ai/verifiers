"""The `[legacy]` block: running a classic (v0) environment through the bridge.

Quarantined in its own block, not mixed into `[env]` — a v0 env is a different
thing to run, not a variant of a v1 env's config."""

from pydantic import Field
from pydantic_config import BaseConfig

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
