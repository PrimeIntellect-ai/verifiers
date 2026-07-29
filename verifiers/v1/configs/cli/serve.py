"""Environment-server CLI configuration."""

from pydantic import AliasChoices, Field, SerializeAsAny, model_validator
from pydantic_config import BaseConfig

from verifiers.v1.configs.cli.env import narrowed_env_annotation, resolve_env_field
from verifiers.v1.configs.env import EnvConfig
from verifiers.v1.configs.legacy import LegacyEnvConfig
from verifiers.v1.configs.serve import ServingConfig
from verifiers.v1.envs.single_agent import SingleAgentEnvConfig


class ServeConfig(BaseConfig):
    """`uv run serve`: what to serve (`[env]`, or `[legacy]` for a classic v0 env) and
    how it's hosted (`[serve]`)."""

    env: SerializeAsAny[EnvConfig] = SingleAgentEnvConfig()
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
        """Whether this run goes through the v0 bridge: a legacy id and no v1 taskset."""
        return self.legacy.id is not None and not self.env.taskset.id

    @property
    def env_id(self) -> str:
        """The run's identifier: the v1 env's, else the v0 env id."""
        return self.env.env_id or self.legacy.id or ""

    @model_validator(mode="after")
    def _refuse_mixed_run(self):
        # A v0 id next to any v1 env identity leaves one of the two going nowhere, and
        # which one depends on `is_legacy`: a taskset makes it False, so the v0 env never
        # loads; a bare `--env.id` leaves it True, so the v0 env runs under the v1 name.
        if self.legacy.id is None or not self.env.env_id:
            return self
        if self.env.taskset.id:
            raise ValueError(
                f"--legacy.id {self.legacy.id!r} is a classic (v0) env and can't combine "
                f"with the v1 taskset {self.env.taskset.id!r}. Pairing a reusable env with "
                f"a taskset is --env.id {self.legacy.id!r} (TOML: id under [env]); to run "
                "the v0 env instead, drop the taskset."
            )
        raise ValueError(
            f"--legacy.id {self.legacy.id!r} is a classic (v0) env and can't combine with "
            f"the v1 env --env.id {self.env.id!r}: the v0 env is what would run, stamped "
            "with the v1 env's name. Keep whichever one you meant to run."
        )

    @model_validator(mode="before")
    @classmethod
    def _resolve_env(cls, data):
        return resolve_env_field(data, narrowed_env_annotation(cls))
