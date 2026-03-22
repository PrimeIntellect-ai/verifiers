from typing import Any

import verifiers as vf
from verifiers.envs.integrations.nemo_gym_agent import (
    NemoGymAgentEnv,
    _build_dataset,
    _resolve_gym_config,
)


def load_environment(
    dataset_split: str = "example",
    vllm_server_host: str = "127.0.0.1",
    vllm_server_port: int = 8000,
    head_server_host: str = "0.0.0.0",
    head_server_port: int = 11000,
    **kwargs: Any,
) -> vf.Environment:
    dataset, _ = _build_dataset(resource_server="arc_agi", dataset_split=dataset_split)
    return NemoGymAgentEnv(
        gym_configs=[_resolve_gym_config("arc_agi")],
        dataset=dataset,
        vllm_server_host=vllm_server_host,
        vllm_server_port=vllm_server_port,
        head_server_host=head_server_host,
        head_server_port=head_server_port,
        **kwargs,
    )
