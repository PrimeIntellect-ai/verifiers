"""Environment-server CLI configuration."""

from pydantic import AliasChoices, Field, SerializeAsAny, model_validator
from pydantic_config import BaseConfig

from verifiers.v1.configs.cli.env import (
    narrowed_env_annotation,
    resolve_env_field,
    single_agent_env_config,
)
from verifiers.v1.configs.env import EnvConfig
from verifiers.v1.configs.legacy import (
    LegacyEnvConfig,
    is_legacy,
    refuse_mixed_run,
    run_env_id,
)
from verifiers.v1.configs.serve import ServingConfig


class ServeConfig(BaseConfig):
    """`uv run serve`: what to serve (`[env]`, or `[legacy]` for a classic v0 env) and
    how it's hosted (`[serve]`)."""

    env: SerializeAsAny[EnvConfig] = Field(default_factory=single_agent_env_config)
    """The environment — which env, its seed taskset, each agent, its knobs. Narrowed to
    the selected env's config class by the env id, else the taskset id."""
    serve: ServingConfig = ServingConfig()
    """How it's served: the worker pool, the bind address, each worker's episode bound."""
    legacy: LegacyEnvConfig = LegacyEnvConfig()
    """A classic (v0) environment to serve through the bridge instead of `[env]`."""
    verbose: bool = Field(False, validation_alias=AliasChoices("verbose", "v"))
    """Log at debug level instead of info."""
    dry_run: bool = False
    """Resolve + validate the config and dump it, then exit."""

    @property
    def is_legacy(self) -> bool:
        return is_legacy(self.env, self.legacy)

    @property
    def env_id(self) -> str:
        return run_env_id(self.env, self.legacy)

    @model_validator(mode="after")
    def _refuse_mixed_run(self):
        refuse_mixed_run(self.env, self.legacy)
        return self

    @model_validator(mode="before")
    @classmethod
    def _resolve_env(cls, data):
        return resolve_env_field(data, narrowed_env_annotation(cls))
