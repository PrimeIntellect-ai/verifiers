from typing import Any

import verifiers as vf
from verifiers.envs.integrations.nemo_gym import (
    NemoGymEnv,
    _build_dataset,
    _resolve_gym_config,
)


def load_environment(
    dataset_path: str | None = None,
    policy_base_url: str | None = None,
    policy_api_key: str | None = None,
    **kwargs: Any,
) -> vf.Environment:
    dataset, _ = _build_dataset("mcqa", "example", dataset_path=dataset_path)
    return NemoGymEnv(
        gym_configs=[_resolve_gym_config("mcqa")],
        dataset=dataset,
        policy_base_url=policy_base_url,
        policy_api_key=policy_api_key,
        **kwargs,
    )
