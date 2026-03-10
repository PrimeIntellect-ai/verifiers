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
    """NeMo Gym workplace_assistant environment using the agent server.

    workplace_assistant is a multi-turn, tool-rich environment.  The agent
    server handles session seeding, the full multi-turn tool-calling loop, and
    final reward via the resource server's /verify endpoint — no sandbox or
    explicit /seed_session calls needed from this side.

    Args:
        gym_config: Path to the NeMo Gym resource-server YAML config for
            workplace_assistant.
        dataset_split: One of "example", "train", "validation".
        vllm_server_host / vllm_server_port: Address of the running vLLM
            policy server that the agent server will call.
        head_server_host / head_server_port: Address the NeMo Gym head
            server will bind to.
    """
    dataset, _ = _build_dataset(
        resource_server="workplace_assistant", dataset_split=dataset_split
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
