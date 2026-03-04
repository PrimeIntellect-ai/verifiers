from typing import Any

import verifiers as vf
from verifiers.envs.integrations.nemo_gym_env import (
    NemoGymEnv,
    _build_dataset,
    _reward_from_verify,
)


def load_environment(
    dataset_split: str = "example",
    max_turns: int = 8,
    **kwargs: Any,
) -> vf.Environment:
    dataset, _ = _build_dataset(
        resource_server="math_with_code", dataset_split=dataset_split
    )
    rubric = vf.Rubric(funcs=[_reward_from_verify], weights=[1.0])
    return NemoGymEnv(
        resource_server="math_with_code",
        dataset=dataset,
        rubric=rubric,
        max_turns=max_turns,
        config_overrides={"max_execution_time": 10},
        extra_pip_packages=["numpy", "scipy", "pandas"],
        **kwargs,
    )
