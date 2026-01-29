"""
Actor: A trainable entity with distinct identity (system prompt) in multi-agent environments.
"""

from dataclasses import dataclass, field
from typing import Any


@dataclass
class Actor:
    """
    A trainable actor with distinct system prompt.

    Fields:
        id: Unique identifier for this actor (e.g., "player1", "guesser")
        system_prompt: The actor's persona/instructions (used in build_actor_prompt)
        max_tokens: Max response length for this actor
        is_trainable: Whether to compute GRPO advantages (False for frozen actors)
        sampling_args: Per-actor model settings (temperature, etc.)
    """

    id: str
    system_prompt: str = ""
    max_tokens: int = 4096
    is_trainable: bool = True
    sampling_args: dict[str, Any] = field(default_factory=dict)

    def __hash__(self) -> int:
        return hash(self.id)

    def __eq__(self, other: object) -> bool:
        if isinstance(other, Actor):
            return self.id == other.id
        return False

    def __repr__(self) -> str:
        trainable_str = "trainable" if self.is_trainable else "frozen"
        return f"Actor(id={self.id!r}, {trainable_str})"

    def merge_sampling_args(self, base_args: dict[str, Any]) -> dict[str, Any]:
        """Merge actor's sampling args with base args (actor takes precedence)."""
        merged = dict(base_args)
        merged.update(self.sampling_args)
        if self.max_tokens:
            merged["max_tokens"] = self.max_tokens
        return merged
