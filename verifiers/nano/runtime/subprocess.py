"""Local subprocess runtime: run the program on the host; server on localhost."""

import asyncio
import os
from typing import Literal

from pydantic_config import BaseConfig

from verifiers.nano.runtime.base import ProgramResult, Runtime

# A local subprocess inherits only these host vars (a whitelist) plus the
# interception endpoint we inject — so it can never reach a real provider with
# the host's credentials. (A container/sandbox is isolated and inherits nothing.)
INHERITED_ENV_VARS = (
    "PATH",
    "HOME",
    "USER",
    "LOGNAME",
    "SHELL",
    "LANG",
    "LC_ALL",
    "TERM",
    "TMPDIR",
)


class SubprocessConfig(BaseConfig):
    kind: Literal["subprocess"] = "subprocess"
    cwd: str | None = None


class SubprocessRuntime(Runtime):
    """Runs the program as a local subprocess; the server is on localhost."""

    def __init__(self, config: SubprocessConfig) -> None:
        self.config = config

    async def start(self, port: int) -> str:
        return f"http://127.0.0.1:{port}/v1"

    async def run(self, argv: list[str], env: dict[str, str]) -> ProgramResult:
        full_env = {k: os.environ[k] for k in INHERITED_ENV_VARS if k in os.environ}
        full_env.update(env)
        proc = await asyncio.create_subprocess_exec(
            *argv,
            env=full_env,
            cwd=self.config.cwd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, stderr = await proc.communicate()
        return ProgramResult(
            exit_code=proc.returncode or 0,
            stdout=stdout.decode(errors="replace"),
            stderr=stderr.decode(errors="replace"),
        )
