"""
Protocol: Defines how multiple agents interact in a multi-agent environment.

A Protocol specifies turn order and agent interaction patterns, separate from
the task/environment logic. This allows the same protocol (e.g., round-robin)
to be reused across different tasks.

Example protocols:
- RoundRobinProtocol: Agents take turns in order (0 → 1 → ...)
- SpawningProtocol: One agent can spawn sub-rollouts of another (e.g.,
  proposer-solver where the proposer creates problems that k solvers attempt)
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from verifiers.types import State

if TYPE_CHECKING:
    from verifiers.envs.environment import Environment


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


@dataclass
class SpawnSpec:
    """A spawn request: one or more sub-rollouts of ``child_env`` for ``agent_id``.

    Fields:
        agent_id: Role tag for the spawned children. Their trajectory steps
            are embedded in the parent's trajectory with this id so existing
            per-agent advantage/credit-assignment machinery
            (Rubric.score_group, interleave_rollout(split_by_agent=True),
            MicroBatch.actor_ids) picks them up without modification.
        child_env: The Environment to roll out for each child. Any
            ``Environment`` with a ``rollout()`` method works — does not
            have to be a MultiAgentEnv.
        inputs: One dataset row per child rollout. Length determines how many
            children are spawned.
        is_trainable: When False, the spawned tokens are still generated and
            included in the parent's context but their completion masks are
            zeroed out at training time (mirrors ``Agent.is_trainable``).
    """

    agent_id: str
    child_env: "Environment"
    inputs: list[Any]
    is_trainable: bool = True


@dataclass
class SpawnResult:
    """A SpawnSpec paired with the resulting child states.

    Stored in ``state["extras"]["spawns"]`` so MultiAgentRewardFunc
    implementations can compute parent rewards from child outcomes
    (e.g. goldilocks scoring on per-child verify scores).
    """

    spec: SpawnSpec
    states: list[State] = field(default_factory=list)


class SpawningProtocol(Protocol):
    """Protocol extension for hierarchical / one-to-many agent interactions.

    The base ``Protocol`` only sequences turns of a fixed agent set. A
    ``SpawningProtocol`` additionally lets the current agent's turn fan out
    into N sub-rollouts of another agent (or env). The canonical example
    is proposer-solver: the proposer designs a problem, then k solver
    children attempt it, and the proposer's reward depends on how the
    solvers fared.

    Implementers provide:
        - ``should_spawn(state)``: was the previous turn a spawn trigger?
        - ``get_spawn_specs(state)``: what to spawn (one or more SpawnSpecs)

    MultiAgentEnv.rollout() handles the actual sub-rollout execution and
    weaves children's trajectory steps into the parent's trajectory, tagged
    with the child agent_id from the spec. Reward functions read the
    completed children from ``state["extras"]["spawns"]`` to compute
    per-agent rewards.
    """

    @abstractmethod
    def should_spawn(self, state: State) -> bool:
        """Return True if the most recent turn should trigger sub-rollouts.

        Called after ``MultiAgentEnv.on_turn_complete`` and before
        ``get_next_agent``. Subclasses typically inspect the last trajectory
        step's content (e.g. did the proposer emit a valid problem?).
        """
        ...

    @abstractmethod
    def get_spawn_specs(self, state: State) -> list[SpawnSpec]:
        """Return the spawn requests for the current state.

        Called only when ``should_spawn`` is True. Each ``SpawnSpec`` may
        request multiple child rollouts via its ``inputs`` list, and
        multiple specs may be returned (e.g. spawn two solver groups for
        two different problems the proposer emitted in one turn).
        """
        ...
