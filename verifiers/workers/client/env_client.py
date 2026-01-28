from abc import ABC, abstractmethod

from verifiers.types import (
    ClientConfig,
    RolloutInput,
    RolloutOutput,
    SamplingArgs,
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
    ) -> RolloutOutput:
        """Run a rollout on the remote environment server and return serializable output."""
        ...

    @abstractmethod
    async def run_group(
        self,
        group_inputs: list[RolloutInput],
        client_config: ClientConfig,
        model: str,
        sampling_args: SamplingArgs,
        score: bool = True,
    ) -> list[RolloutOutput]:
        """Run a group of rollouts on the remote environment server and return serializable outputs."""
        ...
