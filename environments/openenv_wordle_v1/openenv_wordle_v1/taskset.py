"""OpenEnv's official Wordle environment.

The Space serves its ASGI app from ``textarena_env.server.app`` rather than
OpenEnv's default ``server.app``, so this wrapper supplies that provider option.
"""

from typing import Any, Literal

import verifiers.v1 as vf
from verifiers.v1.tasksets.openenv import (
    OpenEnvConfig,
    OpenEnvTask,
    OpenEnvTaskset,
)


class OpenEnvWordleConfig(OpenEnvConfig):
    env: Literal["openenv/wordle"] = "openenv/wordle"
    provider_kwargs: dict[str, Any] = {"app": "textarena_env.server.app:app"}


class OpenEnvWordleTaskset(
    OpenEnvTaskset, vf.Taskset[OpenEnvTask, OpenEnvWordleConfig]
):
    pass
