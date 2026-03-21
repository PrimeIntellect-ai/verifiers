"""
Agent: A concrete entity that can respond. Model + config.

Agent + Task = Rollout.

Agent duck-types as the old Actor — same fields + merge_sampling_args().
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from openai import AsyncOpenAI

    from verifiers.types import Messages


@dataclass
class Agent:
    """
    A concrete entity that can respond.

    Fields:
        id: Unique identifier (e.g., "guesser", "player1")
        system_prompt: The agent's persona/instructions
        max_tokens: Max response length
        is_trainable: Whether to compute GRPO advantages (False for frozen agents)
        model: Model name override. None = use default from trainer.
        client: AsyncOpenAI client override. None = use default.
        sampling_args: Per-agent model settings (temperature, etc.)
    """

    id: str
    system_prompt: str = ""
    max_tokens: int = 4096
    is_trainable: bool = True
    model: str | None = None
    client: "AsyncOpenAI | None" = None
    sampling_args: dict[str, Any] = field(default_factory=dict)

    def merge_sampling_args(self, base_args: dict[str, Any]) -> dict[str, Any]:
        """Merge agent's sampling args with base args (agent takes precedence).

        Duck-types with Actor.merge_sampling_args() so Agent works directly
        in the MultiAgentEnv rollout loop without conversion.
        """
        merged = dict(base_args)
        merged.update(self.sampling_args)
        if self.max_tokens:
            merged["max_tokens"] = self.max_tokens
        return merged

    def __hash__(self) -> int:
        return hash(self.id)

    def __eq__(self, other: object) -> bool:
        if isinstance(other, Agent):
            return self.id == other.id
        return False

    def __repr__(self) -> str:
        trainable = "trainable" if self.is_trainable else "frozen"
        return f"Agent(id={self.id!r}, {trainable})"
