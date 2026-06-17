"""OpenEnv Echo: the reusable OpenEnv adapter pinned to the bundled MCP project."""

from typing import Literal

import verifiers.v1 as vf
from tasksets.openenv_v1 import (
    OpenEnvConfig,
    OpenEnvState,
    OpenEnvTask,
    OpenEnvTaskset,
)


class OpenEnvEchoConfig(OpenEnvConfig):
    project: Literal["proj"] = "proj"
    instruction: str = (
        "Call at least one OpenEnv echo tool, then summarize the returned result."
    )


class OpenEnvEchoTaskset(
    OpenEnvTaskset, vf.Taskset[OpenEnvTask, OpenEnvEchoConfig, OpenEnvState]
):
    pass
