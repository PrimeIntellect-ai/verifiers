"""The harness plugin's config: which program plays a seat, and its knobs."""

from __future__ import annotations

import os
from pathlib import Path

from pydantic import Field
from pydantic_config import BaseConfig

from verifiers.v1.types import ID
from verifiers.v1.utils.install import env_name


class HarnessConfig(BaseConfig):
    id: ID = "bash"
    """Local package or Hub `org/name[@version]`, set through the seat's
    `--env.<role>.harness.id` (`--env.agent.harness.id` on the single-agent env)."""
    env: dict[str, str] = Field(default_factory=dict)
    """Extra program variables; harness-owned variables take precedence."""
    forward_env: list[str] = Field(default_factory=list)
    """Host variables to forward without writing secrets into config; explicit `env` wins."""
    disabled_tools: list[str] | None = None
    skills: list[Path] = Field(default_factory=list)
    """Skill folders to upload into the program's skill discovery directory — each
    lands at `<skills dir>/<folder name>`. Only harnesses whose program discovers
    skills natively (`SUPPORTS_SKILLS`) accept them."""

    @property
    def name(self) -> str:
        return env_name(self.id)

    @property
    def resolved_env(self) -> dict[str, str]:
        forwarded = {k: os.environ[k] for k in self.forward_env if k in os.environ}
        return {**forwarded, **self.env}
