"""The judge plugin's config: an endpoint plus prompt/scoring knobs, and the
`judges` entries plugged into a task config."""

from collections.abc import Sequence
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field, FiniteFloat, SerializeAsAny

from verifiers.v1.configs.client import BaseClientConfig
from verifiers.v1.configs.harness import ID
from verifiers.v1.configs.sampling import SamplingConfig


class JudgeConfig(BaseClientConfig):
    id: ID = ""
    """Plugin id; empty for a judge called directly by task code."""
    name: str = ""
    """Reward key override for a plugged judge."""
    weight: FiniteFloat = 1.0
    model: str = "openai/gpt-6-luna"
    sampling: SamplingConfig = SamplingConfig()
    prompt: Path | None = None
    """File whose text overrides the judge's default prompt template."""
    max_input_tokens: int | None = Field(None, ge=1)
    """The input tokens at which the judge endpoint may cut its input. A call whose
    reported input count reaches it raises `JudgeInputTooLarge` instead of grading; a
    request far over it (`judge.input_token_floor`) raises before the call. None checks
    nothing."""


Judges = list[SerializeAsAny[JudgeConfig]]
"""Config-plugged judges, resolved by id and serialized as their concrete types."""


def judge_key(config: JudgeConfig) -> str:
    return config.name or config.id


def resolve_judges(entries: Sequence[Any]) -> list[JudgeConfig]:
    from verifiers.v1.utils.loaders import judge_config_type

    resolved = []
    for entry in entries:
        raw = entry.model_dump() if isinstance(entry, BaseModel) else dict(entry)
        if not raw.get("id"):
            raise ValueError(
                "each `judges` entry needs an `id` (a judge plugin: `reference`, "
                "`rubric`, `decision`, or an installed package)"
            )
        resolved.append(judge_config_type(raw["id"]).model_validate(raw))
    return resolved


def check_judges(entries: Sequence[JudgeConfig]) -> None:
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
