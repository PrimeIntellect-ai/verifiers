from dataclasses import dataclass, fields
from typing import Any, ClassVar


@dataclass
class SamplingArgs:
    """Container for supported inference sampling arguments."""

    # Standard OpenAI parameters
    temperature: float | None = None
    top_p: float | None = None
    max_tokens: int | None = None
    presence_penalty: float | None = None
    frequency_penalty: float | None = None

    # Extra vLLM parameters that belong under extra_body
    _EXTRA_BODY_FIELDS: ClassVar[set[str]] = {"top_k", "min_p", "repetition_penalty"}

    top_k: int | None = None
    min_p: float | None = None
    repetition_penalty: float | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert parameters to an OpenAI-compatible payload."""
        payload: dict[str, Any] = {}
        extra_body: dict[str, Any] = {}

        for field in fields(self):
            value = getattr(self, field.name)
            if value is not None:
                if field.name in self._EXTRA_BODY_FIELDS:
                    extra_body[field.name] = value
                else:
                    payload[field.name] = value

        if extra_body:
            payload["extra_body"] = extra_body

        return payload


def merge_sampling_args(
    args: SamplingArgs,
    extra_dict: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Merge serialized args with an optional extra (overriding) dictionary."""
    merged = args.to_dict()
    if not extra_dict:
        return merged

    for key, value in extra_dict.items():
        if key == "extra_body":
            existing_extra = merged.setdefault("extra_body", {})
            existing_extra.update(value)
        else:
            merged[key] = value

    return merged
