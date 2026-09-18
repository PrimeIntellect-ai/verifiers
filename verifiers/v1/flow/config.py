"""The flow's configuration: seats are `AgentConfig` fields declared on a subclass."""

from __future__ import annotations

from pydantic import Field, PositiveInt
from pydantic_config import BaseConfig

from verifiers.v1.clients import ClientConfig
from verifiers.v1.interception import ElasticInterceptionPoolConfig, InterceptionConfig
from verifiers.v1.types import SamplingConfig


class FlowConfig(BaseConfig):
    model: str | None = None
    """Model for seats that pin none."""
    client: ClientConfig | None = None
    """Endpoint for seats that pin none."""
    sampling: SamplingConfig | None = None
    """Sampling for seats that pin none; a seat's own values merge on top."""
    interception: InterceptionConfig = ElasticInterceptionPoolConfig()
    """The interception shape, as in `EnvConfig`; tunneled when any seat's runtime is remote."""
    pools: dict[str, PositiveInt] = Field(
        default_factory=lambda: {"units": 4, "runtimes": 8}
    )
    """How many task stages run at once (`units`) and how many boxes live at once
    (`runtimes`); a pipeline may hold any name it adds."""
    payload_cap: PositiveInt = 1_000_000
    """Largest call value a record keeps, in bytes of JSON; bulk belongs in traces or files."""
    attach_by_key: bool = False
    """A migration switch: a call whose record is not found under its identity attaches the
    unit's newest record with the same key and kind instead of running again. For a launch
    after a seat change that does not alter what a call asks (a budget, a runtime size) when
    the earlier identity carried it. Off, a changed identity reruns the call, which is right
    whenever the change could alter the answer (a model, a prompt, an input)."""
