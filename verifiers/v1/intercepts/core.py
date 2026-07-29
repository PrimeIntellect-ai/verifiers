"""Types for task-authored model-exchange interception."""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from typing import Literal

from verifiers.v1.types import StrictBaseModel

Direction = Literal["request", "response"]


InterceptResult = str | None
Interceptor = Callable[..., InterceptResult | Awaitable[InterceptResult]]


class InterceptRecord(StrictBaseModel):
    """One action an interceptor took on a model exchange."""

    direction: Direction
    handler: str
    action: Literal["rewrite"]


__all__ = [
    "Direction",
    "InterceptRecord",
    "InterceptResult",
    "Interceptor",
]
