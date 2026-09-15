"""The run's configuration: seats are `AgentConfig` fields declared on a subclass."""

from __future__ import annotations

from pydantic import Field, PositiveInt
from pydantic_config import BaseConfig

from verifiers.v1.clients import ClientConfig
from verifiers.v1.interception import ElasticInterceptionPoolConfig, InterceptionConfig
from verifiers.v1.types import SamplingConfig

RUNTIMES = "runtimes"
"""The pool held around every box a step provisions."""


class FlowConfig(BaseConfig):
    model: str | None = None
    """Model for seats that pin none."""
    client: ClientConfig | None = None
    """Endpoint for seats that pin none."""
    sampling: SamplingConfig | None = None
    """Sampling for seats that pin none; a seat's own values merge on top."""
    max_concurrent_rows: PositiveInt = 4
    interception: InterceptionConfig = ElasticInterceptionPoolConfig()
    """The interception shape, as in `EnvConfig`: `elastic` (default), `server`, or
    `static`. Tunneled when any seat's runtime is remote; a task whose tool servers sit
    in a remote runtime behind local seats needs a `server` with a tunnel configured."""
    pools: dict[str, PositiveInt] = Field(default_factory=lambda: {RUNTIMES: 8})
    """Named capacity pools: how many holders at once; `runtimes` bounds live boxes."""
    payload_cap: PositiveInt = 1_000_000
    """Largest step value the ledger records, in bytes of JSON; bulk belongs in traces or files."""
