"""OpenEnv's official Wordle environment."""

from typing import Any, Literal

import verifiers.v1 as vf
from verifiers.v1.tasksets import (
    OpenEnvConfig,
    OpenEnvTask,
    OpenEnvTaskset,
)


class OpenEnvWordleConfig(OpenEnvConfig):
    env: Literal["openenv/wordle"] = "openenv/wordle"

    def model_post_init(self, __context: Any) -> None:
        # Wordle uses a non-default ASGI app in its Space repository.
        self.task.user.provider_kwargs.setdefault("app", "textarena_env.server.app:app")


class OpenEnvWordleTaskset(
    OpenEnvTaskset, vf.Taskset[OpenEnvTask, OpenEnvWordleConfig]
):
    pass
