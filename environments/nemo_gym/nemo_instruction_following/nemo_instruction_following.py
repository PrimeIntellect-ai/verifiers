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
    dataset, _ = _build_dataset(
        resource_server="instruction_following", dataset_split=dataset_split
    )
    rubric = vf.Rubric(funcs=[_reward_from_verify], weights=[1.0])
    return NemoGymEnv(
        resource_server="instruction_following",
        dataset=dataset,
        rubric=rubric,
        max_turns=1,
        extra_pip_packages=[
            "git+https://github.com/abukharin-nv/verifiable-instructions.git",
            "https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-3.8.0/en_core_web_sm-3.8.0-py3-none-any.whl",
        ],
        **kwargs,
    )
