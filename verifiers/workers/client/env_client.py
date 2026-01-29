import logging
from abc import ABC, abstractmethod

from verifiers.types import (
    ClientConfig,
    RolloutInput,
    RolloutOutput,
    SamplingArgs,
)
from verifiers.workers.types import (
    HealthRequest,
    HealthResponse,
    RunGroupRequest,
    RunGroupResponse,
    RunRolloutRequest,
    RunRolloutResponse,
)


class EnvClient(ABC):
    """Base class for environment clients."""

    def __init__(self, address: str):
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        self.address = address

    async def health(self) -> bool:
        request = HealthRequest()
        response = await self.handle_health_request(request)
        return response.success

    async def run_rollout(
        self,
        input: RolloutInput,
        client_config: ClientConfig,
        model: str,
        sampling_args: SamplingArgs,
        max_retries: int = 0,
        state_columns: list[str] | None = None,
    ) -> RolloutOutput:
        request = RunRolloutRequest(
            input=input,
            client_config=client_config,
            model=model,
            sampling_args=sampling_args,
            max_retries=max_retries,
            state_columns=state_columns,
        )
        response = await self.handle_run_rollout_request(request)
        assert response.output is not None
        return response.output

    async def run_group(
        self,
        group_inputs: list[RolloutInput],
        client_config: ClientConfig,
        model: str,
        sampling_args: SamplingArgs,
        max_retries: int = 0,
        state_columns: list[str] | None = None,
    ) -> list[RolloutOutput]:
        request = RunGroupRequest(
            group_inputs=group_inputs,
            client_config=client_config,
            model=model,
            sampling_args=sampling_args,
            max_retries=max_retries,
            state_columns=state_columns,
        )
        response = await self.handle_run_group_request(request)
        assert response.outputs is not None
        return response.outputs

    @abstractmethod
    async def handle_health_request(self, request: HealthRequest) -> HealthResponse: ...

    @abstractmethod
    async def handle_run_rollout_request(
        self,
        request: RunRolloutRequest,
    ) -> RunRolloutResponse:
        """Run a rollout on the remote environment server."""
        ...

    @abstractmethod
    async def handle_run_group_request(
        self,
        request: RunGroupRequest,
    ) -> RunGroupResponse:
        """Run a group of rollouts on the remote environment server."""
        ...
