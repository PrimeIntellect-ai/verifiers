"""The run's configuration: seats are `AgentConfig` fields declared on a subclass."""

from __future__ import annotations

from pydantic import Field, field_validator
from pydantic_config import BaseConfig

from verifiers.v1.clients import ClientConfig
from verifiers.v1.interception import ElasticInterceptionPoolConfig, InterceptionConfig


class FlowConfig(BaseConfig):
    model: str | None = None
    """Model for seats that pin none."""
    client: ClientConfig | None = None
    """Endpoint for seats that pin none."""
    max_concurrent_rows: int = Field(default=4, gt=0)
    interception: InterceptionConfig = ElasticInterceptionPoolConfig()
    """The interception shape, as in `EnvConfig`: `elastic` (default), `server`, or
    `static`. Tunneled when any seat's runtime is remote; a task whose tool servers sit
    in a remote runtime behind local seats needs a `server` with a tunnel configured."""
    pools: dict[str, int] = Field(default_factory=lambda: {"runtimes": 8})
    """Named capacity pools: boxes (and anything else named) held at once."""
    payload_cap: int = Field(default=1_000_000, gt=0)
    """Largest step value the ledger records, in bytes of JSON; bulk belongs in traces or files."""

    @field_validator("pools")
    @classmethod
    def _pools_positive(cls, pools: dict[str, int]) -> dict[str, int]:
        if empty := {name: size for name, size in pools.items() if size < 1}:
            raise ValueError(f"pool sizes must be >= 1, got {empty}")
        return pools
