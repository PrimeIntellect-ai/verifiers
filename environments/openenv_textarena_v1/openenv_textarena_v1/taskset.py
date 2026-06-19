"""OpenEnv TextArena: the reusable OpenEnv adapter pinned to the official image."""

from typing import Literal

import verifiers.v1 as vf
from tasksets.openenv_v1 import (
    OpenEnvConfig,
    OpenEnvState,
    OpenEnvTask,
    OpenEnvTaskset,
)


class OpenEnvTextArenaConfig(OpenEnvConfig):
    image: Literal[
        "ghcr.io/meta-pytorch/openenv-textarena-env@sha256:c4ba5acf578e77a721c4bb009933ee54e1c8893d290c94362d95be7527f2f079"
    ] = "ghcr.io/meta-pytorch/openenv-textarena-env@sha256:c4ba5acf578e77a721c4bb009933ee54e1c8893d290c94362d95be7527f2f079"
    contract: Literal["gym"] = "gym"
    system_prompt: str = (
        "Play the game carefully. Return one valid move in the exact format requested."
    )


class OpenEnvTextArenaTaskset(
    OpenEnvTaskset, vf.Taskset[OpenEnvTask, OpenEnvTextArenaConfig, OpenEnvState]
):
    pass
