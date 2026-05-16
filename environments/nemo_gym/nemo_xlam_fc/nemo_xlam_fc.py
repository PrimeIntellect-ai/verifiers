from typing import Any

import verifiers as vf
from verifiers.envs.integrations.nemo_gym import (
    NemoGymEnv,
    _build_dataset,
    _reward_from_verify,
)


def load_environment(
    dataset_split: str = "example",
    **kwargs: Any,
) -> vf.Environment:
    dataset, _ = _build_dataset(resource_server="xlam_fc", dataset_split=dataset_split)
    rubric = vf.Rubric(funcs=[_reward_from_verify], weights=[1.0])
    return NemoGymEnv(
        resource_server="xlam_fc",
        dataset=dataset,
        rubric=rubric,
        max_turns=1,
        **kwargs,
    )
