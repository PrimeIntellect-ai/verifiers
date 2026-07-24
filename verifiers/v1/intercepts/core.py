"""Types for task-authored model-exchange interception."""

from __future__ import annotations

import json
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Any, Literal

from verifiers.v1.types import StrictBaseModel

Direction = Literal["request", "response"]


class Terminate(StrictBaseModel):
    """End the rollout immediately with an ordinary reward."""

    reason: str = "intercepted"
    reward: float = 0.0


InterceptResult = str | Terminate | None
Interceptor = Callable[..., InterceptResult | Awaitable[InterceptResult]]


class InterceptRecord(StrictBaseModel):
    """One action an interceptor took on a model exchange."""

    direction: Direction
    handler: str
    action: Literal["rewrite", "terminate"]
    target: str = ""
    reason: str = ""
    before: str = ""
    reward: float | None = None


@dataclass(frozen=True)
class PendingTermination:
    """A terminal result waiting for the server to preserve the sampled turn."""

    handler: str
    result: Terminate


@dataclass(frozen=True)
class InterceptOutcome:
    """Internal summary of one direction's handler chain."""

    rewritten: bool = False
    termination: PendingTermination | None = None


def snippet(value: Any) -> str:
    """A bounded JSON preview for trace records."""
    return json.dumps(value, separators=(",", ":"), default=str)[:500]


__all__ = [
    "Direction",
    "InterceptRecord",
    "InterceptResult",
    "Interceptor",
    "Terminate",
]
