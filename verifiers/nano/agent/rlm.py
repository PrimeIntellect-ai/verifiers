"""The rlm agent: installs the rlm CLI into the runtime and runs the binary.

`RLMAgentConfig` carries both how to install rlm (repo/branch/token/path) and its
runtime knobs (`max_depth`, `tools`), which rlm reads from `RLM_*` env vars.
"""

import os
from typing import Literal

from pydantic import Field
from pydantic_config import BaseConfig

from verifiers.nano.agent.base import Agent
from verifiers.nano.runtime import Runtime

CONTAINER_PATH = "/root/.local/bin:/usr/local/bin:/usr/bin:/bin"


class RLMAgentConfig(BaseConfig):
    """The rlm CLI agent — describes how to install rlm and how it should run."""

    kind: Literal["rlm"] = "rlm"
    env: dict[str, str] = Field(default_factory=dict)
    # install
    path: str = "/root/.local/bin/rlm"
    """Path to the rlm binary (the in-container install location by default; set to
    the host binary when running on the subprocess runtime)."""
    repo: str = "github.com/PrimeIntellect-ai/rlm.git"
    branch: str = "main"
    gh_token_var: str = "GH_TOKEN"
    """Env var holding a GitHub token for cloning the (private) rlm repo."""
    # runtime knobs (passed to rlm via RLM_* env vars)
    max_depth: int = 0
    """Recursion depth rlm may spawn sub-agents to (RLM_MAX_DEPTH)."""
    tools: list[str] | None = None
    """Built-in rlm tools to enable (RLM_TOOLS); None uses rlm's default set."""


class RLMAgent(Agent):
    config: RLMAgentConfig

    async def prepare(self, runtime: Runtime, env: dict[str, str]) -> list[str]:
        env["RLM_MAX_DEPTH"] = str(self.config.max_depth)
        if self.config.tools is not None:
            env["RLM_TOOLS"] = ",".join(self.config.tools)
        if runtime.config.kind != "subprocess":  # install rlm into the container
            token = os.environ.get(self.config.gh_token_var, "")
            auth = f"{token}@" if token else ""
            await runtime.run(
                [
                    "sh",
                    "-c",
                    "apt-get update -qq && apt-get install -y -qq git curl && "
                    f"git clone --depth 1 --branch {self.config.branch} "
                    f"https://{auth}{self.config.repo} /opt/rlm && "
                    f"GH_TOKEN={token} RLM_CHECKOUT_PATH=/opt/rlm bash /opt/rlm/install.sh",
                ],
                env,
            )
            env["PATH"] = CONTAINER_PATH
        return [self.config.path]
