"""
Agent: A participant in multi-agent environments.

Contains agent metadata (id, system prompt, trainability).
"""

from dataclasses import dataclass


@dataclass
class Agent:
    """
    An agent in a multi-agent environment.

    Fields:
        id: Unique identifier for this agent (e.g., "player_0", "guesser")
        system_prompt: The agent's specific instructions
        is_trainable: Whether to compute gradients for this agent's actions
    """

    id: str
    system_prompt: str = ""
    is_trainable: bool = True

    def __hash__(self) -> int:
        return hash(self.id)

    def __eq__(self, other: object) -> bool:
        if isinstance(other, Agent):
            return self.id == other.id
        return False

    def __repr__(self) -> str:
        trainable_str = "trainable" if self.is_trainable else "frozen"
        return f"Agent(id={self.id!r}, {trainable_str})"
