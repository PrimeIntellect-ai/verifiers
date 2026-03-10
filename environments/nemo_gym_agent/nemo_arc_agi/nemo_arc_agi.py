from typing import Any

import verifiers as vf
from verifiers.envs.integrations.nemo_gym_agent import NemoGymAgentEnv, _build_dataset


def load_environment(
    gym_config: str,
    dataset_split: str = "example",
    vllm_server_host: str = "127.0.0.1",
    vllm_server_port: int = 8000,
    head_server_host: str = "0.0.0.0",
    head_server_port: int = 11000,
    **kwargs: Any,
) -> vf.Environment:
    """NeMo Gym arc_agi environment using the agent server.

    arc_agi is a single-turn visual grid pattern-matching task.  The agent
    server manages the single inference call and scores it via the resource
    server's /verify endpoint.

    Args:
        gym_config: Path to the NeMo Gym resource-server YAML config for
            arc_agi.
        dataset_split: One of "example", "train", "validation".
        vllm_server_host / vllm_server_port: Address of the running vLLM
            policy server that the agent server will call.
        head_server_host / head_server_port: Address the NeMo Gym head
            server will bind to.
    """
    dataset, _ = _build_dataset(
        resource_server="arc_agi", dataset_split=dataset_split
    )
    return NemoGymAgentEnv(
        gym_configs=[gym_config],
        dataset=dataset,
        vllm_server_host=vllm_server_host,
        vllm_server_port=vllm_server_port,
        head_server_host=head_server_host,
        head_server_port=head_server_port,
        **kwargs,
    )
