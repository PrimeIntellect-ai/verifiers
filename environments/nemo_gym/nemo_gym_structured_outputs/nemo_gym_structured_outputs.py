from typing import Any

import verifiers as vf
from verifiers.envs.integrations.nemo_gym import (
    NemoGymEnv,
    _build_dataset,
    _resolve_gym_config,
)


def load_environment(
    dataset_path: str | None = None,
    **kwargs: Any,
) -> vf.Environment:
    dataset, _ = _build_dataset(
        "structured_outputs", "example", dataset_path=dataset_path
    )
    return NemoGymEnv(
        gym_configs=[
            _resolve_gym_config("structured_outputs", "structured_outputs_json")
        ],
        dataset=dataset,
        **kwargs,
    )
