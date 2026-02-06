import pytest

from verifiers.types import ClientConfig, RolloutOutput
from verifiers.workers.client.env_client import EnvClient
from verifiers.workers.types import (
    HealthRequest,
    HealthResponse,
    RunGroupRequest,
    RunGroupResponse,
    RunRolloutRequest,
    RunRolloutResponse,
)


class RecordingEnvClient(EnvClient):
    def __init__(self, output: RolloutOutput):
        super().__init__(address="tcp://127.0.0.1:5000")
        self.output = output
        self.rollout_timeout: float | None = None
        self.group_timeout: float | None = None

    async def handle_health_request(
        self, request: HealthRequest, timeout: float | None
    ) -> HealthResponse:
        return HealthResponse()

    async def handle_run_rollout_request(
        self, request: RunRolloutRequest, timeout: float | None
    ) -> RunRolloutResponse:
        self.rollout_timeout = timeout
        return RunRolloutResponse(output=self.output)

    async def handle_run_group_request(
        self, request: RunGroupRequest, timeout: float | None
    ) -> RunGroupResponse:
        self.group_timeout = timeout
        return RunGroupResponse(outputs=[self.output] * len(request.group_inputs))

    async def close(self) -> None:
        return None


@pytest.mark.asyncio
async def test_run_rollout_uses_client_timeout(make_input, make_output):
    client = RecordingEnvClient(output=make_output())
    config = ClientConfig(api_base_url="http://localhost:8000/v1", timeout=12.5)

    await client.run_rollout(
        input=make_input(example_id=1),
        client_config=config,
        model="test-model",
        sampling_args={},
    )

    assert client.rollout_timeout == 12.5


@pytest.mark.asyncio
async def test_run_group_uses_max_timeout_across_client_configs(make_input, make_output):
    client = RecordingEnvClient(output=make_output())
    configs = [
        ClientConfig(api_base_url="http://localhost:8000/v1", timeout=8.0),
        ClientConfig(api_base_url="http://localhost:8001/v1", timeout=14.0),
        ClientConfig(api_base_url="http://localhost:8002/v1", timeout=3.0),
    ]

    await client.run_group(
        group_inputs=[make_input(example_id=1), make_input(example_id=2)],
        client_config=configs,
        model="test-model",
        sampling_args={},
    )

    assert client.group_timeout == 14.0
