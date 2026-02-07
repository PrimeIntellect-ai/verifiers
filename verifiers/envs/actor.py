"""
Actor: A trainable entity with distinct identity (system prompt) in multi-agent environments.
"""

from dataclasses import dataclass


@dataclass
class Actor:
    """
    A trainable actor with distinct system prompt.

    Fields:
        id: Unique identifier for this actor (e.g., "player1", "guesser")
        system_prompt: The actor's persona/instructions (used in build_actor_prompt)
        is_trainable: Whether to compute GRPO advantages (False for frozen actors)
    """

    id: str
    system_prompt: str = ""
    is_trainable: bool = True

    def __hash__(self) -> int:
        return hash(self.id)

    def __eq__(self, other: object) -> bool:
        if isinstance(other, Actor):
            return self.id == other.id
        return False

    def __repr__(self) -> str:
        trainable_str = "trainable" if self.is_trainable else "frozen"
        return f"Actor(id={self.id!r}, {trainable_str})"
