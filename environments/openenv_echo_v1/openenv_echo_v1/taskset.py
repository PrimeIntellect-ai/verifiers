"""OpenEnv Echo: the reusable OpenEnv adapter pinned to the official MCP image."""

from typing import Literal

import verifiers.v1 as vf
from tasksets.openenv_v1 import (
    OpenEnvConfig,
    OpenEnvState,
    OpenEnvTask,
    OpenEnvTaskset,
)


class OpenEnvEchoConfig(OpenEnvConfig):
    image: Literal[
        "ghcr.io/meta-pytorch/openenv-echo-env@sha256:56c55669c00b23a6af6adbcd8dd1fb5da3a276aec186b5c46cb4abeb708afa9c"
    ] = "ghcr.io/meta-pytorch/openenv-echo-env@sha256:56c55669c00b23a6af6adbcd8dd1fb5da3a276aec186b5c46cb4abeb708afa9c"
    contract: Literal["mcp"] = "mcp"
    prompt: str = (
        "Call at least one OpenEnv echo tool, then summarize the returned result."
    )


class OpenEnvEchoTaskset(
    OpenEnvTaskset, vf.Taskset[OpenEnvTask, OpenEnvEchoConfig, OpenEnvState]
):
    pass
