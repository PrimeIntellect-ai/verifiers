"""Placement and retry policy for task scoring, independent of the agent program."""

from typing import Literal

from pydantic import Field, SerializationInfo, field_serializer
from pydantic_config import BaseConfig

from verifiers.v1.runtimes import RuntimeConfig


class VerifierConfig(BaseConfig):
    mode: Literal["task", "shared", "isolated"] = "task"
    """Honor the task default, score in the workspace, or restore into a fresh runtime."""
    runtime: RuntimeConfig | None = None
    """Fresh verifier provider/configuration. None uses the agent's provider policy."""
    env: dict[str, str] | None = None
    """Fresh verifier environment. None uses the scoring task's environment."""
    retries: int = Field(2, ge=0)
    """Additional fresh attempts after setup, restoration, or scoring failure."""

    @field_serializer("runtime")
    def serialize_runtime(
        self, runtime: RuntimeConfig | None, info: SerializationInfo
    ) -> dict | None:
        if runtime is None:
            return None
        values = runtime.model_dump(mode=info.mode)
        if "image" not in runtime.model_fields_set:
            values.pop("image", None)
        return values
