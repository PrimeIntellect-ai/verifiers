from __future__ import annotations

from typing import Generic, TypeVar

from pydantic import ConfigDict
from pydantic_config import BaseConfig

from verifiers.v1.clients import ModelRuntime
from verifiers.v1.trace import Trace


class AlgorithmConfig(BaseConfig):
    """Configuration for a trace-training algorithm."""

    model_config = ConfigDict(extra="allow")

    id: str


ConfigT = TypeVar("ConfigT", bound=AlgorithmConfig)


class Algorithm(Generic[ConfigT]):
    """Trace transform that writes branch-level advantages and masks."""

    def __init__(self, config: ConfigT) -> None:
        self.config = config

    async def setup(self, models: dict[str, ModelRuntime]) -> None:
        """Bind any model runtimes the algorithm needs."""

    def loss(self) -> str:
        """Trainer loss channel produced by this algorithm."""
        return "rl"

    async def advantage(self, traces: list[Trace]) -> list[Trace]:
        """Mutate ``branch.advantages`` and ``branch.mask`` in place."""
        return traces
