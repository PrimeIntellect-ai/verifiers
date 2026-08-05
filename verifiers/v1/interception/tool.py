"""Typed wire contract for harness-owned tool interception."""

from typing import Literal

from pydantic import BaseModel

from verifiers.v1.types import MessageContent


class ToolInterceptionRequest(BaseModel):
    """One harness tool call, before execution or after producing its result."""

    tool_call_id: str
    name: str
    can_rewrite: bool
    phase: Literal["before", "after"] = "after"
    content: MessageContent | None = None


class ToolInterceptionDecision(BaseModel):
    """Execute, synthesize a result, or terminate the rollout."""

    action: Literal["allow", "rewrite", "terminate"]
    content: str | None = None
    reason: str | None = None
