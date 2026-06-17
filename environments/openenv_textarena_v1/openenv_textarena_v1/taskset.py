"""OpenEnv TextArena: the reusable OpenEnv adapter pinned to Wordle."""

from typing import Literal

import verifiers.v1 as vf
from tasksets.openenv_v1 import (
    OpenEnvConfig,
    OpenEnvState,
    OpenEnvTask,
    OpenEnvTaskset,
)


class OpenEnvTextArenaConfig(OpenEnvConfig):
    project: Literal["proj"] = "proj"
    system_prompt: str = (
        "Play the game carefully. Return one valid move in the exact format requested."
    )


class OpenEnvTextArenaTaskset(
    OpenEnvTaskset, vf.Taskset[OpenEnvTask, OpenEnvTextArenaConfig, OpenEnvState]
):
    pass
