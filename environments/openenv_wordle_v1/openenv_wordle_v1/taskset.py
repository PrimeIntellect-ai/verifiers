"""OpenEnv's official Wordle environment.

The Space serves its ASGI app from ``textarena_env.server.app`` rather than
OpenEnv's default ``server.app``, so this wrapper supplies that provider option.
"""

from typing import Literal

import verifiers.v1 as vf
from verifiers.v1.tasksets.openenv import (
    OpenEnvConfig,
    OpenEnvTask,
    OpenEnvTaskConfig,
    OpenEnvTaskset,
    OpenEnvUserConfig,
)


class OpenEnvWordleConfig(OpenEnvConfig):
    env: Literal["openenv/wordle"] = "openenv/wordle"
    # Taskset-level provider options extend this required OpenEnv default.
    task: OpenEnvTaskConfig = OpenEnvTaskConfig(
        user=OpenEnvUserConfig(provider_kwargs={"app": "textarena_env.server.app:app"})
    )


class OpenEnvWordleTaskset(
    OpenEnvTaskset, vf.Taskset[OpenEnvTask, OpenEnvWordleConfig]
):
    pass
