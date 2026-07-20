"""Environment-server CLI configuration."""

from pydantic import AliasChoices, Field

from verifiers.v1.env import EnvServerConfig


class ServeConfig(EnvServerConfig):
    address: str = Field(
        "tcp://127.0.0.1:5000", validation_alias=AliasChoices("address", "a")
    )
    """ZMQ address the ROUTER binds (and clients connect to)."""
    verbose: bool = Field(False, validation_alias=AliasChoices("verbose", "v"))
    """Log at debug level instead of info."""
    dry_run: bool = False
    """Resolve + validate the config and dump it, then exit."""
    metrics_address: str = "127.0.0.1"
    """Address for the optional Prometheus HTTP endpoint."""
    metrics_port: int | None = Field(None, ge=1, le=65535)
    """Port for the optional Prometheus HTTP endpoint; unset disables metrics."""
