"""
TaskSet: What to solve + the environment + how to evaluate.

Generalizes HuggingFace Dataset to be agent-native. A TaskSet bundles:
- Dataset (examples to solve)
- Rubric (evaluation criteria)
- Game rules (for interactive/multi-agent tasks)

Tasks do NOT specify how the agent works — just what should be done,
the environment, and how it'll be evaluated.

Simple tasks are just data + rubric (like HF datasets).
Interactive tasks add game hooks for multi-agent environments.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from datasets import Dataset

    import verifiers as vf
    from verifiers.types import Messages, State


class TaskSet:
    """
    A task definition: dataset + rubric + environment rules.

    For simple tasks (math, code, QA), just provide name/dataset/rubric.
    For interactive tasks (games, multi-agent), subclass and override
    the game hooks.

    Args:
        name: Unique task name (used for routing and identification)
        dataset: HuggingFace Dataset with prompt/answer/info/example_id/task columns
        rubric: Scoring rubric (Rubric or MultiAgentRubric)
        roles: Ordered list of role IDs for multi-agent (e.g., ["guesser", "thinker"])
    """

    def __init__(
        self,
        name: str,
        dataset: "Dataset",
        rubric: "vf.Rubric | None" = None,
        roles: list[str] | None = None,
    ):
        self.name = name
        self.dataset = dataset
        self.rubric = rubric
        self.roles: list[str] = roles or []

    # -------------------------------------------------------------------------
    # Dataset-like interface
    # -------------------------------------------------------------------------

    def get_examples(self, num_examples: int = -1) -> "Dataset":
        """Return dataset, optionally truncated."""
        if num_examples > 0:
            n = min(num_examples, len(self.dataset))
            return self.dataset.select(range(n))
        return self.dataset

    # -------------------------------------------------------------------------
    # Skills: task-specific tools exported to agents
    # -------------------------------------------------------------------------

    def get_skills(self) -> list[Any]:
        """Return task-specific tools/skills that agents should have.

        Override to export domain-specific tools (e.g., submit_grid for ARC,
        run_tests for code tasks). The env merges these with its own tools.
        """
        return []

    # -------------------------------------------------------------------------
    # Game hooks (override for interactive/multi-agent tasks)
    # -------------------------------------------------------------------------

    async def setup_state(self, state: "State") -> "State":
        """Initialize task-specific state in state['extras'].

        Called once at the start of each episode.
        """
        return state

    async def build_prompt(self, role: str, state: "State") -> "Messages":
        """Build the prompt for this role's turn.

        Args:
            role: Which role will respond (e.g., "guesser", "player1")
            state: Current game state with trajectory and extras

        Returns:
            Messages list (system + user messages)
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} is an interactive task but doesn't implement build_prompt(). "
            f"Override this method to define what each role sees."
        )

    async def on_turn_complete(self, state: "State") -> None:
        """Game logic after each turn completes.

        The last turn's info is in state['trajectory'][-1]:
        - ['completion'][-1]['content']: The model's response text
        - ['extras']['actor_id']: Which role just responded
        """
        pass

    async def should_stop(self, state: "State") -> bool:
        """Return True if the task/game should end."""
        return False

    async def on_game_end(self, state: "State") -> None:
        """Compute final metrics after the game loop exits."""
        pass

    # -------------------------------------------------------------------------
    # Spawn (for hierarchical tasks that fan out to child environments)
    # -------------------------------------------------------------------------

    async def spawn(self, state: "State", inputs: list, score: bool = False, **kwargs) -> list:
        """Spawn child rollouts via Registry.

        Requires the parent env to have Registry wired (via Registry(agents, envs)).
        Registry reference is stored in state["registry"] by MultiAgentEnv.setup_state().

        Args:
            state: Current game state (contains registry reference)
            inputs: List of rollout inputs, each with "task" field for routing
            score: Whether to score child rollouts (default False)

        Returns:
            List of completed child states
        """
        registry = state.get("registry")
        if registry is None:
            raise RuntimeError(
                "spawn() requires Registry. Wire with: Registry(agents=..., envs=...)"
            )
        return await registry.spawn(
            inputs,
            client=state["client"],
            model=state["model"],
            sampling_args=state.get("sampling_args"),
            score=score,
            **kwargs,
        )

    # -------------------------------------------------------------------------
    # Turn management (override for custom turn ordering)
    # -------------------------------------------------------------------------

    def get_initial_role(self, state: "State") -> str:
        """Which role goes first. Default: first in self.roles."""
        if not self.roles:
            raise ValueError("TaskSet.roles is empty — set roles for multi-agent tasks")
        return self.roles[0]

    def get_next_role(self, state: "State") -> str:
        """Which role goes next. Default: cycle through self.roles."""
        if not self.roles:
            raise ValueError("TaskSet.roles is empty — set roles for multi-agent tasks")
        current = state["extras"].get("current_actor_id")
        if current is None:
            return self.roles[0]
        try:
            idx = self.roles.index(current)
        except ValueError:
            return self.roles[0]
        return self.roles[(idx + 1) % len(self.roles)]

    def get_active_roles(self, state: "State") -> list[str]:
        """Which roles act this turn. Override for simultaneous moves.

        Default: single role (alternating turns).
        """
        current = state["extras"].get("current_actor_id")
        if current is None:
            return [self.get_initial_role(state)]
        return [self.get_next_role(state)]

    # -------------------------------------------------------------------------
    # Repr
    # -------------------------------------------------------------------------

    def __repr__(self) -> str:
        parts = [f"name={self.name!r}", f"examples={len(self.dataset)}"]
        if self.roles:
            parts.append(f"roles={self.roles}")
        return f"TaskSet({', '.join(parts)})"
