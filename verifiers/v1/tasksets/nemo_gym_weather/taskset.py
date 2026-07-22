"""Zero-config wrapper around NeMo Gym's example MCP weather task."""

from pathlib import Path

from verifiers.v1.tasksets.nemo_gym import (
    NeMoGymConfig,
    NeMoGymTaskset,
)


class NeMoGymWeatherConfig(NeMoGymConfig):
    dataset_path: Path = Path(__file__).with_name("example.jsonl")


class NeMoGymWeatherTaskset(NeMoGymTaskset[NeMoGymWeatherConfig]):
    resource_server = (
        "resources_servers.example_mcp_weather.app:ExampleMCPWeatherResourcesServer"
    )
