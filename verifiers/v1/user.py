from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field
from verifiers.types import Messages

from .config import Config
from .toolset import MCPServerSpec
from .types import JsonData


class User(Config):
    """MCP server used by the harness as the environment/user simulator."""

    class TurnRequest(BaseModel):
        task: JsonData
        state: JsonData
        transcript: list[JsonData]

    class TurnResult(BaseModel):
        messages: Messages = Field(default_factory=list)
        scratch: JsonData = Field(default_factory=dict)
        metrics: dict[str, float] = Field(default_factory=dict)
        artifacts: JsonData = Field(default_factory=dict)
        reward_delta: float = 0.0
        stop_condition: str | None = None
        is_completed: bool | None = None
        is_truncated: bool | None = None

    name: str = "user"
    server: MCPServerSpec
    scope: Literal["rollout", "env"] = "rollout"
