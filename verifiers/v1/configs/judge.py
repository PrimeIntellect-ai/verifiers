"""Shared judge identity/weight, HTTP model settings, and plugged judge configs."""

from collections.abc import Sequence
from pathlib import Path
from typing import Any

from pydantic import BaseModel, FiniteFloat, SerializeAsAny
from pydantic_config import BaseConfig

from verifiers.v1.configs.client import BaseClientConfig
from verifiers.v1.configs.harness import ID
from verifiers.v1.configs.sampling import SamplingConfig


class JudgePluginConfig(BaseConfig):
    """Identity and reward weight shared by HTTP and runtime-backed judges."""

    id: ID = ""
    """Plugin id; empty for a judge called directly by task code."""
    name: str = ""
    """Reward key override for a plugged judge."""
    weight: FiniteFloat = 1.0


class JudgeConfig(BaseClientConfig, JudgePluginConfig):
    model: str = "openai/gpt-6-luna"
    sampling: SamplingConfig = SamplingConfig()
    prompt: Path | None = None
    """File whose text overrides the judge's default prompt template."""


Judges = list[SerializeAsAny[JudgePluginConfig]]
"""Config-plugged judges, resolved by id and serialized as their concrete types."""


def judge_key(config: JudgePluginConfig) -> str:
    return config.name or config.id


def resolve_judges(entries: Sequence[Any]) -> list[JudgePluginConfig]:
    from verifiers.v1.utils.loaders import judge_config_type

    resolved = []
    for entry in entries:
        raw = entry.model_dump() if isinstance(entry, BaseModel) else dict(entry)
        if not raw.get("id"):
            raise ValueError(
                "each `judges` entry needs an `id` (a judge plugin: `reference`, "
                "`rubric`, `decision`, `rewardkit`, or an installed package)"
            )
        resolved.append(judge_config_type(raw["id"]).model_validate(raw))
    return resolved


def check_judges(entries: Sequence[JudgePluginConfig]) -> None:
    for entry in entries:
        if not entry.id:
            raise ValueError(
                "each `judges` entry needs an `id` (a judge plugin: `reference`, "
                "`rubric`, `decision`, or an installed package)"
            )
    keys = [judge_key(entry) for entry in entries]
    if duplicates := {key for key in keys if keys.count(key) > 1}:
        raise ValueError(
            f"`judges` entries share a reward key {sorted(duplicates)}; set a "
            "distinct `name` on each to keep both verdicts"
        )
