from abc import ABC, abstractmethod
from pathlib import Path

from pydantic import BaseModel

from verifiers.types import (
    ClientConfig,
    GenerateOutputs,
    RolloutInput,
    RolloutTiming,
    SamplingArgs,
    State,
    TrajectoryStep,
)


class HealthResponse(BaseModel):
    is_healthy: bool


class RolloutRequest(BaseModel):
    input: RolloutInput
    client_config: ClientConfig
    model: str
    sampling_args: SamplingArgs


class RolloutResponse(BaseModel):
    is_completed: bool
    is_truncated: bool
    stop_condition: str | None
    trajectory: list[TrajectoryStep]
    reward: float | None
    advantage: float | None
    metrics: dict[str, float] | None
    timing: RolloutTiming | None
    error: str | None


GroupRequest = list[RolloutRequest]
GroupResponse = list[RolloutResponse]


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
        # on_start: StartCallback | None = None,
        # on_progress: ProgressCallback | None = None,
        # on_log: LogCallback | None = None,
    ) -> GenerateOutputs:
        """Mirrors Environment.evaluate"""
        ...
