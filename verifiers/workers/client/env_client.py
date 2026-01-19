from abc import ABC, abstractmethod

from verifiers.types import ClientConfig, RolloutInput, SamplingArgs, State


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
    ) -> State: ...
