"""Explicit termination returned by request and response hooks."""

from typing import Literal

from pydantic import BaseModel, ConfigDict


class Terminate(BaseModel):
    """End the rollout without delivering the intercepted request or response."""

    model_config = ConfigDict(frozen=True)

    reason: str = "intercepted"
    reward: float = 0.0


class Termination(Terminate):
    """The hook and boundary responsible for a trace's termination."""

    handler: str
    boundary: Literal["request", "response"]
