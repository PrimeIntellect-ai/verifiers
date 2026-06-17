from typing import Literal

import verifiers.v1 as vf


class OpenEnvTask(vf.Task):
    contract: Literal["gym", "mcp"]
    port: int
    start_command: str
    seed: int


class OpenEnvState(vf.State):
    reward: float = 0.0
    done: bool = False
