"""The built-in default agent: stages a small openai chat-loop script and runs it."""

import base64
import secrets
from pathlib import Path
from typing import Literal

from pydantic import Field
from pydantic_config import BaseConfig

from verifiers.nano.agent.base import Agent
from verifiers.nano.runtime import Runtime

PROGRAM_SOURCE = (Path(__file__).resolve().parent / "program.py").read_text()


class DefaultAgentConfig(BaseConfig):
    """The built-in agent. Stdlib + `openai` only, so it stages into any runtime."""

    kind: Literal["default"] = "default"
    env: dict[str, str] = Field(default_factory=dict)
    """Extra environment variables for the program."""


class DefaultAgent(Agent):
    config: DefaultAgentConfig

    async def prepare(self, runtime: Runtime, env: dict[str, str]) -> list[str]:
        if runtime.config.kind != "subprocess":  # the host venv already has openai
            await runtime.run(["sh", "-c", "pip install -q openai"], env)
        path = f"/tmp/vf_agent_{secrets.token_hex(6)}.py"
        source = base64.b64encode(PROGRAM_SOURCE.encode()).decode()
        await runtime.run(["sh", "-c", f"printf %s {source} | base64 -d > {path}"], env)
        return ["python3", path]
