from typing import Any

import verifiers as vf
from verifiers.envs.integrations.nemo_gym_env import (
    load_environment as load_nemo_gym_environment,
)

DEFAULT_NEMO_GYM_WHEEL = "https://test-files.pythonhosted.org/packages/c0/58/451a826009a0b206c932e1ebde3dcff2a8b31152c77133fdde7e5f7ccd90/nemo_gym-0.2.9892rc0-py3-none-any.whl"


def load_environment(
    dataset_split: str = "example",
    dataset_path: str | None = None,
    dataset_limit: int | None = None,
    max_turns: int = 16,
    nemo_package: str = DEFAULT_NEMO_GYM_WHEEL,
    nemo_package_version: str | None = None,
    **kwargs: Any,
) -> vf.Environment:
    """Workplace Assistant environment via the NeMo Gym sandbox adapter."""

    return load_nemo_gym_environment(
        resource_server="workplace_assistant",
        dataset_split=dataset_split,
        dataset_path=dataset_path,
        dataset_limit=dataset_limit,
        max_turns=max_turns,
        nemo_package=nemo_package,
        nemo_package_version=nemo_package_version,
        seed_session_on_start=True,
        **kwargs,
    )
