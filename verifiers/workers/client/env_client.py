import logging
from abc import ABC, abstractmethod

from verifiers.types import (
    ClientConfig,
    RolloutInput,
    RolloutOutput,
    SamplingArgs,
)
from verifiers.utils.client_utils import resolve_client_config
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

    @staticmethod
    def _request_timeout(client_config: ClientConfig) -> float:
        resolved_client_config = resolve_client_config(client_config)
        return max(1.0, resolved_client_config.timeout)

    async def health(self, timeout: float | None = 10) -> bool:
        request = HealthRequest()
        response = await self.handle_health_request(request, timeout=timeout)
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
        resolved_client_config = resolve_client_config(client_config)
        request = RunRolloutRequest(
            input=input,
            client_config=resolved_client_config,
            model=model,
            sampling_args=sampling_args,
            max_retries=max_retries,
            state_columns=state_columns,
        )
        response = await self.handle_run_rollout_request(
            request,
            timeout=self._request_timeout(resolved_client_config),
        )
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
        resolved_client_config = resolve_client_config(client_config)
        request = RunGroupRequest(
            group_inputs=group_inputs,
            client_config=resolved_client_config,
            model=model,
            sampling_args=sampling_args,
            max_retries=max_retries,
            state_columns=state_columns,
        )
        response = await self.handle_run_group_request(
            request,
            timeout=self._request_timeout(resolved_client_config),
        )
        assert response.outputs is not None
        return response.outputs

    @abstractmethod
    async def handle_health_request(
        self, request: HealthRequest, timeout: float | None
    ) -> HealthResponse: ...

    @abstractmethod
    async def handle_run_rollout_request(
        self, request: RunRolloutRequest, timeout: float | None
    ) -> RunRolloutResponse:
        """Run a rollout on the remote environment server."""
        ...

    @abstractmethod
    async def handle_run_group_request(
        self, request: RunGroupRequest, timeout: float | None
    ) -> RunGroupResponse:
        """Run a group of rollouts on the remote environment server."""
        ...

    @abstractmethod
    async def close(self) -> None:
        """Close the client and clean up resources."""
        ...
