"""Environment-server CLI configuration."""

from pydantic import AliasChoices, Field

from verifiers.v1.configs.env import EnvServerConfig


class ServeConfig(EnvServerConfig):
    address: str = Field(
        "tcp://127.0.0.1:5000", validation_alias=AliasChoices("address", "a")
    )
    """ZMQ address the ROUTER binds (and clients connect to)."""
    verbose: bool = Field(False, validation_alias=AliasChoices("verbose", "v"))
    """Log at debug level instead of info."""
    dry_run: bool = False
    """Resolve + validate the config and dump it, then exit."""
