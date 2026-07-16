"""Zero-config wrapper around NeMo Gym's example MCP weather task."""

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
