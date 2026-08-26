"""The harness plugin's config: which program plays a seat, and its knobs."""

from __future__ import annotations

import os
import random
from pathlib import Path

from pydantic import ConfigDict, Field, FiniteFloat, PositiveInt, model_validator
from pydantic_config import BaseConfig

from verifiers.v1.types import ID


class CompactionConfig(BaseConfig):
    """Optional context compaction policy for in-house agent loops."""

    summarize_at_tokens: PositiveInt | tuple[PositiveInt, PositiveInt] | None = None
    """Compact at this token count. A pair draws a task-seeded threshold. When unset, use
    90% of the model context window when the provider advertises it."""

    @model_validator(mode="after")
    def validate_range(self) -> CompactionConfig:
        value = self.summarize_at_tokens
        if isinstance(value, tuple) and value[0] > value[1]:
            raise ValueError(
                "`summarize_at_tokens` range must be (lo, hi) with lo <= hi."
            )
        return self

    def summarize_threshold(self, task_idx: int | None) -> int | None:
        value = self.summarize_at_tokens
        if isinstance(value, tuple):
            lo, hi = value
            return random.Random(task_idx or 0).randint(lo, hi)
        return value


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
    skills: list[Path] = Field(default_factory=list)
    """Skill folders to upload into the program's skill discovery directory — each
    lands at `<skills dir>/<folder name>`. Only harnesses whose program discovers
    skills natively (`SUPPORTS_SKILLS`) accept them."""

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
