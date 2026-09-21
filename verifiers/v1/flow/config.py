"""The flow's configuration: seats are `AgentConfig` fields declared on a subclass."""

from __future__ import annotations

from pydantic import Field, PositiveInt
from pydantic_config import BaseConfig

from verifiers.v1.clients import ClientConfig
from verifiers.v1.interception import ElasticInterceptionPoolConfig, InterceptionConfig
from verifiers.v1.types import SamplingConfig

PoolLimits = dict[str, PositiveInt]


class FlowConfig(BaseConfig):
    model: str | None = None
    """Model for seats that pin none."""
    client: ClientConfig | None = None
    """Endpoint for seats that pin none."""
    sampling: SamplingConfig | None = None
    """Sampling for seats that pin none; a seat's own values merge on top."""
    interception: InterceptionConfig = ElasticInterceptionPoolConfig()
    """The interception shape, as in `EnvConfig`; tunneled when any seat's runtime is remote."""
    pools: PoolLimits = Field(default_factory=lambda: {"units": 4, "runtimes": 8})
    """How many task stages run at once (`units`) and how many boxes live at once
    (`runtimes`); a pipeline may hold any name it adds."""
    stay_alive: bool = False
    """Wait for new runnable work when idle; exit only on drain."""
