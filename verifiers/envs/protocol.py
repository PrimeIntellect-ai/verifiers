"""
Protocol: Defines how multiple agents interact in a multi-agent environment.

A Protocol specifies turn order and agent interaction patterns, separate from
the task/environment logic. This allows the same protocol (e.g., round-robin)
to be reused across different tasks.

Example protocols:
- RoundRobinProtocol: Agents take turns in order (0 → 1 → ...)
- SimultaneousProtocol: All agents act at once (future)
- HierarchicalProtocol: Some agents coordinate others (future)
"""

from abc import ABC, abstractmethod

from verifiers.types import State


class Protocol(ABC):
    """
    Abstract base class for multi-agent interaction protocols.

    A Protocol defines:
    - Turn order (who goes first, who goes next)
    - Agent interaction patterns

    Protocols are independent of:
    - Task/environment logic (game rules, rewards)
    - Model/harness details (how agents generate responses)
    """

    @abstractmethod
    def get_initial_agent(self, state: State) -> str:
        """
        Return the agent ID that starts the rollout.

        Args:
            state: Initial game state

        Returns:
            Agent ID (e.g., "player_0")
        """
        pass

    @abstractmethod
    def get_next_agent(self, state: State) -> str:
        """
        Return the agent ID for the next turn.

        Args:
            state: Current game state (use to determine whose turn it is)

        Returns:
            Agent ID for the next turn
        """
        pass


class RoundRobinProtocol(Protocol):
    """
    Simple round-robin turn order: agents take turns in sequence.

    Example with 3 agents:
        0 → 1 → 2 → 0 → ...
    """

    def __init__(self, agent_ids: list[str]):
        """
        Initialize round-robin protocol.

        Args:
            agent_ids: List of agent IDs in turn order
        """
        if not agent_ids:
            raise ValueError("agent_ids must not be empty")
        self.agent_ids = agent_ids

    def get_initial_agent(self, state: State) -> str:
        """First agent in the list starts."""
        return self.agent_ids[0]

    def get_next_agent(self, state: State) -> str:
        """Cycle through agents in order."""
        current = state["extras"].get("current_agent_id", self.agent_ids[0])
        try:
            current_idx = self.agent_ids.index(current)
        except ValueError:
            current_idx = -1
        next_idx = (current_idx + 1) % len(self.agent_ids)
        return self.agent_ids[next_idx]
