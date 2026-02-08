"""
Agent: A participant in multi-agent environments.

Currently contains agent metadata (id, system prompt, trainability).
In the future, when Harness is introduced, Agent will be extended to
compose with Harness and Model: Agent = Harness + Model.
"""

from dataclasses import dataclass


@dataclass
class Agent:
    """
    An agent in a multi-agent environment.

    Fields:
        id: Unique identifier for this agent (e.g., "player_0", "guesser")
        system_prompt: The agent's persona/instructions
        is_trainable: Whether to compute gradients for this agent's actions

    Future:
        When Harness is introduced, Agent will be extended to include
        rollout logic and model binding: Agent = Harness + Model.
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
