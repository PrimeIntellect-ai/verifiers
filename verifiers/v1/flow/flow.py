"""`Flow`: a class whose attributes are nodes, compiled and validated at definition."""

from __future__ import annotations

from typing import TYPE_CHECKING, ClassVar, Generic

from pydantic import Field
from pydantic_config import BaseConfig
from typing_extensions import TypeVar

from verifiers.v1.clients import ClientConfig
from verifiers.v1.utils.generic import concrete_type

if TYPE_CHECKING:
    from verifiers.v1.flow.compile import Graph


class FlowConfig(BaseConfig):
    """Base config: seats are `AgentConfig` fields declared on the subclass."""

    model: str | None = None
    """Model for seats that pin none."""
    client: ClientConfig | None = None
    """Endpoint for seats that pin none."""
    max_concurrent_rows: int = 4
    pools: dict[str, int] = Field(default_factory=lambda: {"sandboxes": 8})
    """Named capacity pools: max concurrently running node instances per pool."""


ConfigT = TypeVar("ConfigT", bound=FlowConfig, default=FlowConfig)


class Flow(Generic[ConfigT]):
    """Subclass, declare nodes as class attributes, parameterize with a config class.
    The graph is compiled and validated when the class is defined."""

    graph: ClassVar[Graph]
    entry: ClassVar[str | None] = None
    """Entry node; the first declared node when None."""
    skip_checks: ClassVar[tuple[str, ...]] = ()

    def __init_subclass__(cls, **kwargs) -> None:
        super().__init_subclass__(**kwargs)
        from verifiers.v1.flow.compile import compile_flow

        cls.graph = compile_flow(cls)

    def __init__(self, config: ConfigT | None = None) -> None:
        self.config: ConfigT = config if config is not None else self.config_type()()  # type: ignore[assignment]

    @classmethod
    def config_type(cls) -> type[FlowConfig]:
        return concrete_type(cls, FlowConfig, origin=Flow) or FlowConfig

    @property
    def name(self) -> str:
        return type(self).__name__
