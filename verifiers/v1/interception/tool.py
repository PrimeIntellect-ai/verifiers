"""Typed wire payload for native harness tool hooks."""

from typing import Literal

from pydantic import BaseModel

from verifiers.v1.types import ToolMessage


class ToolHookRequest(BaseModel):
    phase: Literal["before", "after"]
    can_rewrite: bool
    message: ToolMessage
