"""Upstream NeMo Gym ``example_mcp_weather`` as a zero-data-config taskset.

Start only that resources server with
``uv run verifiers/v1/tasksets/nemo_gym_weather/server.py``, then run this taskset with
any MCP-capable Verifiers V1 harness. Override ``task.resources_url`` when
``NEMO_GYM_PORT`` is not 8000.
"""

from pathlib import Path

from verifiers.v1.taskset import Taskset
from verifiers.v1.tasksets.nemo_gym import (
    NeMoGymConfig,
    NeMoGymTask,
    NeMoGymTaskset,
)


class NeMoGymWeatherConfig(NeMoGymConfig):
    dataset_path: Path = Path(__file__).with_name("example.jsonl")


class NeMoGymWeatherTaskset(NeMoGymTaskset, Taskset[NeMoGymTask, NeMoGymWeatherConfig]):
    pass
