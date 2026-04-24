from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from verifiers.types import Messages, RolloutInput

from .channels import ChannelMap


class Task(BaseModel):
    """Immutable taskset row passed into a harness."""

    model_config = ConfigDict(frozen=True)

    prompt: Messages
    example_id: int = 0
    answer: Any = ""
    info: dict[str, Any] = Field(default_factory=dict)
    inputs: dict[str, Any] = Field(default_factory=dict)
    channels: ChannelMap = Field(default_factory=dict)

    def to_input(self) -> RolloutInput:
        input_data: RolloutInput = {
            "prompt": self.prompt,
            "example_id": self.example_id,
        }
        if self.answer != "":
            input_data["answer"] = self.answer
        if self.info:
            input_data["info"] = self.info
        if self.channels:
            input_data["channels"] = self.channels
        input_data.update(self.inputs)
        return input_data
