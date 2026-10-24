"""Sampling knobs for model requests."""

from typing import Any

from pydantic import AliasChoices, BaseModel, ConfigDict, Field, model_validator


class SamplingConfig(BaseModel):
    """Typed sampling knobs; provider-specific keys pass through (extra='allow')."""

    model_config = ConfigDict(extra="allow")
    temperature: float | None = None
    top_p: float | None = None
    reasoning_effort: str | None = None
    max_tokens: int | None = Field(
        None, validation_alias=AliasChoices("max_tokens", "max_completion_tokens")
    )

    @model_validator(mode="before")
    @classmethod
    def normalize(cls, data: Any) -> Any:
        """Use one key per parameter before merging overrides, retaining null clears."""
        if not isinstance(data, dict):
            return data
        normalized = {}
        for layer in (data.get("extra_body") or {}, data):
            layer = dict(layer)
            if "max_completion_tokens" in layer:
                layer.setdefault("max_tokens", layer.pop("max_completion_tokens"))
            layer.pop("extra_body", None)
            normalized.update(layer)
        return normalized


Sampling = SamplingConfig
