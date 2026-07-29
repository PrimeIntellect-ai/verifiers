"""Types for task-authored model-exchange interception."""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Literal

from verifiers.v1.types import AssistantMessage, StrictBaseModel, ToolMessage

Direction = Literal["request", "response"]


class Terminate(StrictBaseModel):
    """End the rollout immediately with a final reward."""

    reason: str = "intercepted"
    reward: float = 0.0


InterceptResult = AssistantMessage | ToolMessage | dict | Terminate | None
Interceptor = Callable[..., InterceptResult | Awaitable[InterceptResult]]


class InterceptRecord(StrictBaseModel):
    """One action an interceptor took on a model exchange."""

    direction: Direction
    handler: str
    action: Literal["rewrite", "terminate"]


@dataclass(frozen=True)
class InterceptOutcome:
    """Internal summary of one direction's handler chain."""

    rewritten: bool = False
    termination: tuple[str, Terminate] | None = None


__all__ = [
    "Direction",
    "InterceptRecord",
    "InterceptResult",
    "Interceptor",
    "Terminate",
]
