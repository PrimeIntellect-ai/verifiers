"""The plain types the config tree shares with the runtime. `verifiers.v1.types` re-exports
them; config modules import them from here, so the config tree loads without numpy."""

from typing import Any, Literal

from pydantic import AliasChoices, BaseModel, ConfigDict, Field, model_validator

DialectName = Literal["chat_completions", "responses", "anthropic_messages"]
"""A model API wire format: a `Dialect.name`, and the values of the harness `dialect`
setting."""


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


ID = str
"""Plugin id: the name of an installed package exporting the plugin."""
