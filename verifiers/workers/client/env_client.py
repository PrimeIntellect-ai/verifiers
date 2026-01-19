from abc import ABC, abstractmethod
from pathlib import Path

from verifiers.types import (
    ClientConfig,
    GenerateOutputs,
    RolloutInput,
    SamplingArgs,
    State,
)


class EnvClient(ABC):
    def __init__(self, address: str):
        self.address = address

    @abstractmethod
    async def health(self) -> bool: ...

    @abstractmethod
    async def run_rollout(
        self,
        input: RolloutInput,
        client_config: ClientConfig,
        model: str,
        sampling_args: SamplingArgs,
        score: bool = True,
    ) -> State:
        """Mirrors Environment.run_rollout"""
        ...

    @abstractmethod
    async def run_group(
        self,
        group_inputs: list[RolloutInput],
        client_config: ClientConfig,
        model: str,
        sampling_args: SamplingArgs,
        score: bool = True,
    ) -> list[State]:
        """Mirrors Environment.run_group"""
        ...

    @abstractmethod
    async def evaluate(
        self,
        client_config: ClientConfig,
        model: str,
        sampling_args: SamplingArgs,
        num_examples: int,
        rollouts_per_example: int,
        max_concurrent: int,
        results_path: Path | None,
        state_columns: list[str] | None,
        save_results: bool,
        save_every: int,
        independent_scoring: bool = False,
    ) -> GenerateOutputs:
        """Mirrors Environment.evaluate"""
        ...
