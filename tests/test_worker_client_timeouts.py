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
        self.rollout_url: str | None = None
        self.group_url: str | None = None
        self.rollout_key_var: str | None = None
        self.group_key_var: str | None = None

    async def handle_health_request(
        self, request: HealthRequest, timeout: float | None
    ) -> HealthResponse:
        return HealthResponse()

    async def handle_run_rollout_request(
        self, request: RunRolloutRequest, timeout: float | None
    ) -> RunRolloutResponse:
        self.rollout_timeout = timeout
        self.rollout_url = request.client_config.api_base_url
        self.rollout_key_var = request.client_config.api_key_var
        return RunRolloutResponse(output=self.output)

    async def handle_run_group_request(
        self, request: RunGroupRequest, timeout: float | None
    ) -> RunGroupResponse:
        self.group_timeout = timeout
        self.group_url = request.client_config.api_base_url
        self.group_key_var = request.client_config.api_key_var
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
async def test_run_group_uses_client_timeout(make_input, make_output):
    client = RecordingEnvClient(output=make_output())
    config = ClientConfig(api_base_url="http://localhost:8000/v1", timeout=14.0)

    await client.run_group(
        group_inputs=[make_input(example_id=1), make_input(example_id=2)],
        client_config=config,
        model="test-model",
        sampling_args={},
    )

    assert client.group_timeout == 14.0


@pytest.mark.asyncio
async def test_run_rollout_resolves_endpoint_config_before_request(
    make_input, make_output
):
    client = RecordingEnvClient(output=make_output())
    config = ClientConfig(
        api_base_url="http://localhost:8000/v1",
        client_idx=1,
        endpoint_configs=[
            ClientConfig(api_base_url="http://localhost:8001/v1", timeout=8.0),
            ClientConfig(api_base_url="http://localhost:8002/v1", timeout=22.0),
        ],
    )

    await client.run_rollout(
        input=make_input(example_id=1),
        client_config=config,
        model="test-model",
        sampling_args={},
    )

    assert client.rollout_timeout == 22.0
    assert client.rollout_url == "http://localhost:8002/v1"


@pytest.mark.asyncio
async def test_run_group_resolves_endpoint_config_before_request(
    make_input, make_output
):
    client = RecordingEnvClient(output=make_output())
    config = ClientConfig(
        api_base_url="http://localhost:8000/v1",
        client_idx=1,
        endpoint_configs=[
            ClientConfig(api_base_url="http://localhost:8001/v1", timeout=8.0),
            ClientConfig(api_base_url="http://localhost:8002/v1", timeout=22.0),
        ],
    )

    await client.run_group(
        group_inputs=[make_input(example_id=1), make_input(example_id=2)],
        client_config=config,
        model="test-model",
        sampling_args={},
    )

    assert client.group_timeout == 22.0
    assert client.group_url == "http://localhost:8002/v1"


@pytest.mark.asyncio
async def test_resolved_endpoint_inherits_parent_fields(make_input, make_output):
    client = RecordingEnvClient(output=make_output())
    config = ClientConfig(
        api_key_var="PARENT_KEY",
        timeout=17.0,
        endpoint_configs=[ClientConfig(api_base_url="http://localhost:8001/v1")],
    )

    await client.run_group(
        group_inputs=[make_input(example_id=1)],
        client_config=config,
        model="test-model",
        sampling_args={},
    )

    assert client.group_timeout == 17.0
    assert client.group_url == "http://localhost:8001/v1"
    assert client.group_key_var == "PARENT_KEY"
