"""The harness plugin's config: which program plays a seat, and its knobs."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Annotated

from pydantic import BaseModel, ConfigDict, Field, FiniteFloat
from pydantic_config import BaseConfig

from verifiers.v1.types import ID

PinnedVersion = Annotated[str, Field(pattern=r"^[A-Za-z0-9._+-]+$")]
"""A release/tag a harness pins its program install to."""


class RuntimeSkills(BaseModel):
    """A directory of skill folders already present inside the agent runtime."""

    runtime: str


SkillSource = Path | RuntimeSkills


def skill_destination(skill: SkillSource, dest: str) -> str:
    """Runtime roots merge into `dest`; host folders keep their resolved name."""
    return (
        dest if isinstance(skill, RuntimeSkills) else f"{dest}/{skill.resolve().name}"
    )


class HarnessConfig(BaseConfig):
    id: ID = "bash"
    """Installed harness package, set through the seat's
    `--env.<role>.harness.id` (`--env.agent.harness.id` on the single-agent env)."""
    env: dict[str, str] = Field(default_factory=dict)
    """Extra program variables; harness-owned variables take precedence."""
    forward_env: list[str] = Field(default_factory=list)
    """Host variables to forward without writing secrets into config; explicit `env` wins."""
    tool_timeout: FiniteFloat = Field(600.0, gt=0)
    """Seconds a single MCP tool call may take; raise it for tools that boot a VM."""
    disabled_tools: list[str] | None = None
    skills: list[SkillSource] = Field(default_factory=list)
    """Host skill folders or `{runtime: path}` roots of skills inside the runtime.
    Sources are installed in order; later files override matching earlier files.
    Only harnesses with native skill support (`SUPPORTS_SKILLS`) accept them."""

    @property
    def name(self) -> str:
        return self.id

    @property
    def resolved_env(self) -> dict[str, str]:
        forwarded = {k: os.environ[k] for k in self.forward_env if k in os.environ}
        return {**forwarded, **self.env}


class WireHarnessConfig(HarnessConfig):
    """Wire form that preserves harness-specific knobs without importing the harness."""

    model_config = ConfigDict(extra="allow")
