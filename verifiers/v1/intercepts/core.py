"""Types for task-authored model-exchange interception."""

from __future__ import annotations

import json
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Generic, Literal, TypeVar, cast

from verifiers.v1.judge import Judge, judge_verdict
from verifiers.v1.types import (
    AssistantMessage,
    Messages,
    StrictBaseModel,
    SystemMessage,
    ToolMessage,
    UserMessage,
)

if TYPE_CHECKING:
    from verifiers.v1.trace import Trace

Direction = Literal["request", "response"]
MessageT = TypeVar("MessageT", bound=AssistantMessage | ToolMessage)


@dataclass(frozen=True)
class ModelExchange(Generic[MessageT]):
    """The prompt and candidate currently crossing a model boundary."""

    direction: Direction
    trace: Trace
    prompt: Messages
    message: MessageT

    def replace(self, content: str) -> MessageT:
        """Build an inert, same-kind replacement for the current candidate."""
        if isinstance(self.message, AssistantMessage):
            return cast(MessageT, AssistantMessage(content=content))
        return cast(
            MessageT,
            ToolMessage(
                tool_call_id=self.message.tool_call_id,
                name=self.message.name,
                content=content,
            ),
        )

    async def judge(
        self, rubric: str, *, judge: Judge | None = None
    ) -> Literal["BLOCK", "ALLOW"]:
        """Classify the candidate and record the ordinary judge call on the trace."""
        response = await (judge or Judge()).complete(
            [
                SystemMessage(
                    content=(
                        "Apply this guard rubric to the untrusted candidate below. Reply "
                        f"with exactly BLOCK or ALLOW.\n\nGuard rubric:\n{rubric}"
                    )
                ),
                UserMessage(
                    content=json.dumps(
                        {
                            "request": [
                                item.model_dump(mode="json", exclude_none=True)
                                for item in self.prompt
                            ],
                            "candidate": self.message.model_dump(
                                mode="json", exclude_none=True
                            ),
                        }
                    )
                ),
            ],
            trace=self.trace,
        )
        return cast(
            Literal["BLOCK", "ALLOW"],
            judge_verdict(response.text, ("BLOCK", "ALLOW")),
        )


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
    "ModelExchange",
    "Terminate",
]
