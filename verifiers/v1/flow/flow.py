"""`Flow`: a class whose attributes are nodes, compiled and validated at definition."""

from __future__ import annotations

from typing import TYPE_CHECKING, ClassVar, Generic

from pydantic import Field, field_validator
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
    max_concurrent_rows: int = Field(default=4, gt=0)
    """Rows that may run at once; zero or negative would hang the gate forever."""
    inference_concurrency: int | None = Field(default=None, gt=0)
    """Model requests (nested harness calls included) allowed upstream at once; None
    leaves each rollout its own unbounded client, byte-identical to before."""
    pools: dict[str, int] = Field(default_factory=lambda: {"runtimes": 8})
    """Named capacity pools: max concurrently running node instances per pool.
    Zero-capacity pools would deadlock node execution, so sizes must be positive."""

    @field_validator("pools")
    @classmethod
    def _pools_positive(cls, pools: dict[str, int]) -> dict[str, int]:
        empty = {name: size for name, size in pools.items() if size < 1}
        if empty:
            raise ValueError(
                f"pool sizes must be >= 1 (got {empty}); a zero-capacity pool "
                "deadlocks every node that holds it"
            )
        return pools


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
