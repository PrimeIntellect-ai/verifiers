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

    def __init__(
        self,
        address: str,
        name: str | None = None,
        health_check_interval: float = 1.0,  # 1s
        startup_timeout: float = 600.0,  # 10min
        recovery_timeout: float = 600.0,  # 10min
    ):
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        self.address = address
        self.name = f"{name} ({address})" if name is not None else address

        self.health_check_interval = health_check_interval
        self.startup_timeout = startup_timeout
        self.recovery_timeout = recovery_timeout

    async def health(self, timeout: float | None = 1) -> bool:
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
        retry_base_delay: float = 1.0,
        retry_max_backoff: float = 60.0,
        retry_jitter: bool = True,
        state_columns: list[str] | None = None,
    ) -> RolloutOutput:
        resolved_client_config = resolve_client_config(client_config)
        request = RunRolloutRequest(
            input=input,
            client_config=resolved_client_config,
            model=model,
            sampling_args=sampling_args,
            max_retries=max_retries,
            retry_base_delay=retry_base_delay,
            retry_max_backoff=retry_max_backoff,
            retry_jitter=retry_jitter,
            state_columns=state_columns,
        )
        response = await self.handle_run_rollout_request(request, timeout=None)
        assert response.output is not None
        return response.output

    async def run_group(
        self,
        group_inputs: list[RolloutInput],
        client_config: ClientConfig,
        model: str,
        sampling_args: SamplingArgs,
        max_retries: int = 0,
        retry_base_delay: float = 1.0,
        retry_max_backoff: float = 60.0,
        retry_jitter: bool = True,
        state_columns: list[str] | None = None,
    ) -> list[RolloutOutput]:
        resolved_client_config = resolve_client_config(client_config)
        request = RunGroupRequest(
            group_inputs=group_inputs,
            client_config=resolved_client_config,
            model=model,
            sampling_args=sampling_args,
            max_retries=max_retries,
            retry_base_delay=retry_base_delay,
            retry_max_backoff=retry_max_backoff,
            retry_jitter=retry_jitter,
            state_columns=state_columns,
        )
        response = await self.handle_run_group_request(request, timeout=None)
        assert response.outputs is not None
        return response.outputs

    @abstractmethod
    async def wait_for_server_startup(
        self,
        timeout: float | None = None,
    ) -> None:
        """Wait for server to become healthy on initial startup."""
        ...

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
