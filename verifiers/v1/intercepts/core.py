"""Types shared by task-authored interception."""

from collections.abc import Awaitable, Callable
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from verifiers.v1.types import Message

Direction = Literal["request", "response"]


class Terminate(BaseModel):
    """End the rollout instead of delivering the intercepted exchange."""

    model_config = ConfigDict(frozen=True)

    reason: str = "intercepted"
    reward: float = 0.0


InterceptResult = Message | Terminate | None
Interceptor = Callable[..., InterceptResult | Awaitable[InterceptResult]]


class InterceptRecord(BaseModel):
    """One rewrite or termination recorded on a trace."""

    direction: Direction
    handler: str
    action: Literal["rewrite", "terminate"]
    reason: str = ""
    reward: float | None = None


class InterceptDecision(BaseModel):
    """The records produced by sequential hooks."""

    records: list[InterceptRecord] = Field(default_factory=list)
    termination: InterceptRecord | None = None
