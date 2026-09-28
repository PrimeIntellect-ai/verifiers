"""Placement and retry policy for deterministic grading in a fresh runtime."""

from pydantic import Field, SerializationInfo, field_serializer
from pydantic_config import BaseConfig

from verifiers.v1.runtimes import RuntimeConfig


class VerifierConfig(BaseConfig):
    runtime: RuntimeConfig | None = None
    """Independent verifier placement and policy. None provisions a fresh runtime
    equivalent to the solver's resolved task runtime."""
    env: dict[str, str] | None = None
    """Process environment for verifier setup and scoring. None uses the task's
    normal runtime environment."""
    retries: int = Field(2, ge=0)
    """Extra fresh-runtime attempts after setup, restoration, staging, or scoring
    failures."""

    @field_serializer("runtime")
    def serialize_runtime(
        self, runtime: RuntimeConfig | None, info: SerializationInfo
    ) -> dict | None:
        """Keep an omitted image omitted across a resolved-config round trip."""
        if runtime is None:
            return None
        values = runtime.model_dump(mode=info.mode)
        if "image" not in runtime.model_fields_set:
            values.pop("image", None)
        return values
