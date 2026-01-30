"""
Multi-agent environment with turn order management and hierarchical spawning.

This module provides the base class for multi-agent RL environments, extending
MultiTurnEnv with support for:
- Multiple actors with distinct system prompts and sampling args
- Turn order management via get_initial_actor() / get_next_actor() / get_active_actors()
- Per-actor trajectory tagging for credit assignment
- Per-actor state splitting for individual reward computation
- Hierarchical episode spawning via Protocol.spawn()

Key concepts:
- Actor: A trainable entity with its own system prompt (defined in actor.py)
- Protocol: Wires actors to environments, enables spawning (defined in protocol.py)
- State splitting: One game state -> multiple actor states for per-actor rewards

Game Implementation:
- Subclasses implement these main hooks:
  - build_actor_prompt(actor_id, state): Build fresh prompt for this actor
  - on_turn_complete(state): Update game state after each turn
  - on_game_end(state): Compute final metrics after game loop exits (optional)
- Framework handles timing automatically

"""

from __future__ import annotations

import asyncio
import uuid
from abc import abstractmethod
from typing import TYPE_CHECKING

from datasets import Dataset
from openai import AsyncOpenAI

import verifiers as vf
from verifiers.envs.multiturn_env import MultiTurnEnv
from verifiers.types import (
    Messages,
    RolloutInput,
    SamplingArgs,
    State,
    TrajectoryStep,
)

if TYPE_CHECKING:
    from verifiers.envs.actor import Actor
    from verifiers.envs.protocol import Protocol


def _dummy_dataset() -> Dataset:
    """
    Create a placeholder dataset for environments that don't specify one.

    The real dataset is typically owned by Protocol. This prevents errors
    when MultiTurnEnv requires a dataset but one isn't provided.
    """
    return Dataset.from_dict({
        "example_id": [0],
        "prompt": [[{"role": "user", "content": "dummy"}]],
        "answer": [""],
    })


# =============================================================================
# MultiAgentEnv Base Class
# =============================================================================

class MultiAgentEnv(MultiTurnEnv):
    """
    Base class for multi-agent environments.

    Subclasses must implement:
    - get_initial_actor(): Who goes first
    - get_next_actor(): Who goes next (for alternating turns)
    - build_actor_prompt(): Build prompt for current actor
    - on_turn_complete(): Game logic after each turn

    Subclasses may optionally override:
    - on_game_end(): Compute final metrics after game loop exits

    The Protocol reference is injected by Protocol.__init__ when wiring
    actors to environments.
    """

    # -------------------------------------------------------------------------
    # Class Attributes
    # -------------------------------------------------------------------------

    # List of actor IDs this environment uses (e.g., ["player1", "player2"])
    # Subclasses should override this
    actors: list[str] = []

    # Injected by Protocol.__init__ - provides actor lookup and spawning
    protocol: "Protocol | None" = None

    def __init__(self, **kwargs):
        """
        Initialize with dummy dataset if none provided.

        The parent class (MultiTurnEnv) requires a dataset, but for multi-agent
        environments the Protocol often owns the real dataset.
        """
        if "dataset" not in kwargs and "eval_dataset" not in kwargs:
            kwargs["dataset"] = _dummy_dataset()
        super().__init__(**kwargs)

    # -------------------------------------------------------------------------
    # Turn Management
    # -------------------------------------------------------------------------

    @abstractmethod
    def get_initial_actor(self, state: State) -> str:
        """
        Return the actor ID that starts the rollout.

        Example: return "guesser" for Twenty Questions
        """
        pass

    @abstractmethod
    def get_next_actor(self, state: State) -> str:
        """
        Return the actor ID for the next turn.

        Example: return "thinker" if current == "guesser" else "guesser"
        """
        pass

    def get_active_actors(self, state: State) -> list[str]:
        """
        Return actor IDs that act this turn.

        Default: Single actor (standard alternating turns).
        Override for simultaneous moves (e.g., RPS returns ["player1", "player2"]).
        """
        current = state["extras"].get("current_actor_id")
        if current is None:
            return [self.get_initial_actor(state)]
        return [self.get_next_actor(state)]

    def get_actor(self, actor_id: str) -> "Actor":
        """Get an actor by ID from Protocol."""
        if self.protocol is None:
            raise RuntimeError(
                f"Cannot get_actor('{actor_id}') before Protocol is initialized. "
                f"Ensure this environment is passed to Protocol(envs=[...])."
            )
        return self.protocol.get_actor(actor_id)

    # -------------------------------------------------------------------------
    # State Setup
    # -------------------------------------------------------------------------

    async def setup_state(self, state: State) -> State:
        """
        Initialize multi-agent state fields.

        Sets up state["extras"] with:
        - current_actor_id: Who is currently speaking (set in rollout)
        - actor_history: List of (actor_id, turn_index) for credit assignment
        - episode_id: Unique ID for this rollout
        - parent_episode_id: Links to parent if this is a spawned child

        Also initializes state["child_states"] for per-actor state splitting.
        """
        state = await super().setup_state(state)

        state["child_states"] = []
        state["extras"] = {
            "current_actor_id": None,  # Set in rollout() after setup
            "actor_history": [],       # Tracks who spoke at each turn
            "episode_id": state.get("trajectory_id", uuid.uuid4().hex),
            "parent_episode_id": None, # Set if spawned from parent episode
        }

        return state

    # -------------------------------------------------------------------------
    # Game Hooks (Subclasses Implement These)
    # -------------------------------------------------------------------------

    @abstractmethod
    async def build_actor_prompt(self, actor_id: str, state: State) -> Messages:
        """
        Build the prompt for the given actor's turn.

        This is called BEFORE the model generates a response.
        Build a fresh prompt with whatever context this actor needs.

        Args:
            actor_id: The actor who will respond (e.g., "guesser", "player1")
            state: Current game state with trajectory and extras

        Returns:
            Messages list with system prompt and user content
        """
        pass

    @abstractmethod
    async def on_turn_complete(self, state: State) -> None:
        """
        Update game state after a turn completes.

        This is called AFTER the model response is stored in trajectory.
        Use this for game logic:
        - Update scores, counters, flags
        - Check win conditions (set state["extras"]["won"] = True, etc.)
        - Store choices for later resolution (simultaneous moves)

        The last turn's info is in state["trajectory"][-1]:
        - ["completion"][-1]["content"]: The model's response text
        - ["extras"]["actor_id"]: Which actor just responded

        Args:
            state: Current game state (mutate extras as needed)
        """
        pass

    async def on_game_end(self, state: State) -> None:
        """
        Finalize game state after the game loop exits.

        This is called ONCE after the game loop completes (stop condition met),
        BEFORE render_completion() and BEFORE the rubric scores the state.

        Use this for:
        - Computing final metrics (win rates, efficiency scores, etc.)
        - Declaring the winner
        - Preparing data that the rubric will read from state["extras"]

        Unlike on_turn_complete() which is called after each turn, this is
        called exactly once when the game is definitely over.

        Args:
            state: Final game state (mutate extras as needed for scoring)
        """
        pass

    # -------------------------------------------------------------------------
    # Parent Class Requirement (env_response)
    # -------------------------------------------------------------------------

    async def env_response(
        self, messages: Messages, state: State, **kwargs
    ) -> Messages:
        """
        Satisfy MultiTurnEnv's abstract requirement.

        MultiAgentEnv uses on_turn_complete() instead, which is called
        explicitly in our rollout() after storing the response.
        This method is not used in the multi-agent flow.
        """
        return []

    # -------------------------------------------------------------------------
    # Trajectory Management
    # -------------------------------------------------------------------------

    async def add_trajectory_step(
        self, state: State, trajectory_step: TrajectoryStep
    ) -> None:
        """Tag trajectory step with actor_id and record actor history."""
        current_actor_id = state["extras"]["current_actor_id"]
        if current_actor_id:
            # Tag step with actor_id for credit assignment
            trajectory_step["extras"]["actor_id"] = current_actor_id
            # Record history
            turn_index = len(state["trajectory"])
            state["extras"]["actor_history"].append((current_actor_id, turn_index))
        await super().add_trajectory_step(state, trajectory_step)

    # -------------------------------------------------------------------------
    # Main Rollout Loop
    # -------------------------------------------------------------------------

    async def rollout(
        self,
        input,
        client,
        model,
        sampling_args=None,
    ) -> State:
        """
        Run a multi-agent episode.

        Flow:
        1. Setup state
        2. Loop until game ends:
           a. Get active actors (1 for alternating, multiple for simultaneous)
           b. For each actor:
              - Build prompt via build_actor_prompt()
              - Get model response
              - Store in trajectory
              - Process via on_turn_complete()
        3. Call on_game_end() for final metrics
        4. Return final state
        """
        state = await self.init_state(input, client, model, sampling_args)
        try:
            state = await self.setup_state(state)
        except vf.Error as e:
            state["error"] = e
            return state

        while not await self.is_completed(state):
            active_actors = self.get_active_actors(state)

            for actor_id in active_actors:
                state["extras"]["current_actor_id"] = actor_id

                try:
                    # 1. Build prompt for this actor
                    prompt_messages = await self.build_actor_prompt(actor_id, state)

                    # 2. Get model response with actor's sampling args and optional model/client override
                    actor = self.get_actor(actor_id)
                    merged_args = actor.merge_sampling_args(sampling_args or {})

                    # Log which model is being used for this actor
                    used_model = actor.model or state.get("model", "default")
                    self.logger.info(f"[{actor_id}] using model: {used_model}")

                    response = await self.get_model_response(
                        state,
                        prompt_messages,
                        client=actor.client,  # None = use default from state
                        model=actor.model,    # None = use default from state
                        sampling_args=merged_args,
                    )

                    # 3. Store in trajectory
                    await self.add_model_response(state, prompt_messages, response)

                    # 4. Process turn (game logic)
                    await self.on_turn_complete(state)

                except vf.OverlongPromptError:
                    state["prompt_too_long"] = True
                    state["is_truncated"] = True
                    break
                except vf.Error as e:
                    state["error"] = e
                    break

            # Check if we should stop after processing all active actors
            if await self.is_completed(state):
                break

        await self.on_game_end(state)
        await self.render_completion(state)
        return state

    # -------------------------------------------------------------------------
    # Per-Actor State Creation
    # -------------------------------------------------------------------------
    #
    # After a game completes, we split the single game state into per-actor
    # states for individual reward computation and GRPO advantage calculation.
    #
    # Example: RPS game with 6 turns
    #   Full trajectory: [p1, p2, p1, p2, p1, p2]
    #   Player1 state: trajectory=[p1, p1, p1], prompt="You are Player 1..."
    #   Player2 state: trajectory=[p2, p2, p2], prompt="You are Player 2..."
    # -------------------------------------------------------------------------

    # Fields shared by reference across all actor states
    # NOTE: "input" is deliberately NOT shared because State.__getitem__/__setitem__
    # forward reads/writes for INPUT_FIELDS (prompt, answer, etc.) to input[key].
    # If we shared input, all actor_states would read the same prompt.
    # NOTE: "timing" is deliberately NOT shared - each actor state gets its own copy
    # to avoid bugs where score_group() updates the same dict multiple times.
    SHARED_STATE_FIELDS = {
        "client",        # AsyncOpenAI API client
        "model",         # Model name string (e.g., "gpt-4o-mini")
        "trajectory_id", # Unique rollout identifier
    }

    def create_actor_state(
        self,
        parent_state: State,
        actor_id: str,
        actor_trajectory: list[TrajectoryStep],
    ) -> State:
        """
        Create a child state for a specific actor from a parent state.

        This splits a multi-actor game state into per-actor states for:
        - Per-actor reward computation (via MultiAgentRubric)
        - GRPO advantage calculation per actor
        - Training only specific actors (is_trainable filtering)

        Args:
            parent_state: The full game state with all actors' turns
            actor_id: The actor this state is for (e.g., "guesser", "player1")
            actor_trajectory: Only this actor's trajectory steps (filtered)

        Returns:
            A new State with shared fields referenced and actor-specific fields fresh
        """
        # Create empty State - no "input" key means INPUT_FIELDS forwarding doesn't apply
        actor_state = State()

        # Copy shared fields by reference (not duplicated in memory)
        for key in parent_state.keys():
            if key in self.SHARED_STATE_FIELDS:
                actor_state[key] = parent_state[key]

        # Copy timing as a new dict (not shared) to avoid score_group() updating same dict multiple times
        if "timing" in parent_state:
            actor_state["timing"] = dict(parent_state["timing"])

        # Copy INPUT_FIELDS directly (safe because actor_state has no "input" key)
        actor_state["answer"] = parent_state.get("answer", "")
        actor_state["task"] = parent_state.get("task", "")
        actor_state["example_id"] = parent_state.get("example_id", 0)
        actor_state["info"] = parent_state.get("info", {})

        # Set actor-specific trajectory (filtered to just this actor's steps)
        actor_state["trajectory"] = actor_trajectory

        # Copy extras but override actor_id to mark whose state this is
        actor_state["extras"] = {
            **parent_state.get("extras", {}),
            "current_actor_id": actor_id,
        }

        # Fresh fields for scoring (will be computed by rubric)
        actor_state["child_states"] = []
        actor_state["reward"] = None
        actor_state["advantage"] = None
        actor_state["metrics"] = None

        # Copy trainability from Actor to State (so downstream doesn't need Protocol lookup)
        actor = self.get_actor(actor_id)
        actor_state["is_trainable"] = actor.is_trainable

        # Extract actor-specific prompt and completion
        if actor_trajectory:
            # Prompt: Find the LAST system message (actor's own prompt)
            # The raw prompt may contain accumulated context from other actors
            raw_prompt = actor_trajectory[0].get("prompt", [])
            prompt_ref = raw_prompt
            for i in range(len(raw_prompt) - 1, -1, -1):
                if raw_prompt[i].get("role") == "system":
                    prompt_ref = raw_prompt[i:]  # From last system message onward
                    break
            actor_state["prompt"] = prompt_ref

            # Completion: Collect all responses across all turns
            all_completions = []
            for step in actor_trajectory:
                step_completion = step.get("completion", [])
                all_completions.extend(step_completion)
            actor_state["completion"] = all_completions
        else:
            # No trajectory for this actor - use parent's prompt
            actor_state["prompt"] = parent_state.get("prompt", [])
            actor_state["completion"] = []

        return actor_state

    def create_actor_states(self, state: State, actor_ids: list[str] | None = None) -> list[State]:
        """
        Split a parent state into per-actor child states.

        Filters the full trajectory by actor_id (set in add_trajectory_step),
        then creates a state for each actor with their filtered trajectory.

        Args:
            state: The full game state with all actors' turns
            actor_ids: List of actor IDs to create states for.
                       Defaults to self.actors if not provided.

        Returns:
            List of per-actor states, one for each actor_id
        """
        if actor_ids is None:
            actor_ids = self.actors

        actor_states = []
        for actor_id in actor_ids:
            # Filter trajectory to only this actor's steps
            actor_trajectory = [
                step for step in state.get("trajectory", [])
                if step.get("extras", {}).get("actor_id") == actor_id
            ]

            new_state = self.create_actor_state(state, actor_id, actor_trajectory)
            actor_states.append(new_state)

        return actor_states

    # -------------------------------------------------------------------------
    # run_group Override (Flattening for prime-rl)
    # -------------------------------------------------------------------------

    async def run_group(
        self,
        group_inputs: list[RolloutInput],
        client: AsyncOpenAI,
        model: str,
        gen_sampling_args: SamplingArgs,
        gen_sem: asyncio.Semaphore,
        score_sem: asyncio.Semaphore,
        score: bool = True,
    ) -> list[State]:
        """
        Run rollouts and flatten to per-actor states for training.

        This is what prime-rl calls. Returns flattened states so GRPO
        advantages get computed per-actor automatically.

        Flow:
        1. Run game rollouts via parent (each produces one game trajectory)
        2. Flatten: split each game into per-actor states
        3. Include any spawned child_states (proposer-solver pattern)
        4. Score all flattened states together (per-actor GRPO)
        """
        # Run game rollouts (don't score yet - we'll score after flattening)
        game_states = await super().run_group(
            group_inputs, client, model, gen_sampling_args,
            gen_sem, score_sem, score=False
        )

        # Flatten: one game -> multiple per-actor states
        flattened = []
        for game_state in game_states:
            # Split game trajectory by actor_id
            flattened.extend(self.create_actor_states(game_state))
            # Include spawned children (proposer-solver pattern)
            flattened.extend(game_state.get("child_states", []))

        # Score flattened states (per-actor GRPO advantages)
        if score and self.rubric:
            await self.rubric.score_group(flattened, score_sem=score_sem)

        return flattened

    # -------------------------------------------------------------------------
    # Result Building (for generate/eval)
    # -------------------------------------------------------------------------

    def _prepare_rollout_results(
        self,
        all_states: list[State],
        model: str,
        client: AsyncOpenAI,
        state_columns: list[str] | None,
        results_path,
        gen_sampling_args: SamplingArgs,
        start_time: float,
    ):
        """Add actor_id to result dict for multi-agent environments."""
        result = super()._prepare_rollout_results(
            all_states, model, client, state_columns,
            results_path, gen_sampling_args, start_time
        )
        result["actor_id"] = [
            s.get("extras", {}).get("current_actor_id", "unknown")
            for s in all_states
        ]
        return result
