"""The judge plugin's config: an endpoint plus prompt/scoring knobs, and the
`judges` entries plugged into a task config."""

from collections.abc import Sequence
from pathlib import Path
from typing import Any

from pydantic import BaseModel, SerializeAsAny, model_validator

from verifiers.v1.clients import BaseClientConfig
from verifiers.v1.types import ID, SamplingConfig
from verifiers.v1.utils.install import env_name


class JudgeSamplingConfig(SamplingConfig):
    pass


class JudgeConfig(BaseClientConfig):
    id: ID = ""
    """Plugin id; empty for a judge called directly by task code."""
    name: str = ""
    """Reward key override for a plugged judge."""
    weight: float = 1.0
    model: str = "openai/gpt-5.4-nano"
    sampling: JudgeSamplingConfig = JudgeSamplingConfig()
    prompt: str | None = None
    prompt_file: Path | None = None
    """Prompt file override, mutually exclusive with `prompt`."""

    @model_validator(mode="after")
    def check_prompt_source(self) -> "JudgeConfig":
        if self.prompt is not None and self.prompt_file is not None:
            raise ValueError("set `prompt` or `prompt_file`, not both")
        return self


Judges = list[SerializeAsAny[JudgeConfig]]
"""Config-plugged judges, resolved by id and serialized as their concrete types."""


def judge_key(config: JudgeConfig) -> str:
    return config.name or env_name(config.id)


def resolve_judges(entries: Sequence[Any]) -> list[JudgeConfig]:
    from verifiers.v1.loaders import judge_config_type

    resolved = []
    for entry in entries:
        raw = entry.model_dump() if isinstance(entry, BaseModel) else dict(entry)
        if not raw.get("id"):
            raise ValueError(
                "each `judges` entry needs an `id` (a judge plugin: `reference`, "
                "`rubric`, a local package, or a hub `org/name` package)"
            )
        resolved.append(judge_config_type(raw["id"]).model_validate(raw))
    return resolved


def check_judges(entries: Sequence[JudgeConfig]) -> None:
    for entry in entries:
        if not entry.id:
            raise ValueError(
                "each `judges` entry needs an `id` (a judge plugin: `reference`, "
                "`rubric`, a local package, or a hub `org/name` package)"
            )
    keys = [judge_key(entry) for entry in entries]
    if duplicates := {key for key in keys if keys.count(key) > 1}:
        raise ValueError(
            f"`judges` entries share a reward key {sorted(duplicates)}; set a "
            "distinct `name` on each to keep both verdicts"
        )
