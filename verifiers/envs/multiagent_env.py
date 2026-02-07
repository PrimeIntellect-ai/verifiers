"""
Multi-agent environment for turn-based games.

This module provides the base class for multi-agent RL environments, extending
MultiTurnEnv with support for:
- Multiple actors with distinct system prompts
- Turn order management via get_initial_actor() / get_next_actor()
- Per-actor trajectory tagging for credit assignment

Key concepts:
- Actor: A trainable entity with its own system prompt (defined in actor.py)

Game Implementation:
- Subclasses implement these main hooks:
  - get_initial_actor(state): Who goes first
  - get_next_actor(state): Who goes next
  - build_actor_prompt(actor_id, state): Build fresh prompt for this actor
  - on_turn_complete(state): Update game state after each turn
"""

from abc import abstractmethod

import verifiers as vf
from verifiers.envs.actor import Actor
from verifiers.envs.multiturn_env import MultiTurnEnv
from verifiers.types import Messages, State, TrajectoryStep


class MultiAgentEnv(MultiTurnEnv):
    """
    Base class for multi-agent environments.

    Subclasses must implement:
    - get_initial_actor(): Who goes first
    - get_next_actor(): Who goes next (for alternating turns)
    - build_actor_prompt(): Build prompt for current actor

    Subclasses may optionally override:
    - on_turn_complete(): Game logic after each turn
    """

    # List of actor IDs this environment uses (e.g., ["player_0", "player_1"])
    # Subclasses should override this or set in __init__
    actors: list[str] = []

    def __init__(self, **kwargs):
        """Initialize multi-agent environment."""
        super().__init__(**kwargs)
        # Internal storage for Actor objects (when not using Protocol)
        self._actor_registry: dict[str, Actor] = {}

    def register_actor(self, actor: Actor) -> None:
        """Register an Actor object for lookup by get_actor()."""
        self._actor_registry[actor.id] = actor
        if actor.id not in self.actors:
            self.actors.append(actor.id)

    def get_actor(self, actor_id: str) -> Actor:
        """Get an actor by ID."""
        if actor_id not in self._actor_registry:
            raise KeyError(
                f"Actor '{actor_id}' not found. Did you call register_actor()?"
            )
        return self._actor_registry[actor_id]

    # -------------------------------------------------------------------------
    # Turn Management (Subclasses Implement These)
    # -------------------------------------------------------------------------

    @abstractmethod
    def get_initial_actor(self, state: State) -> str:
        """
        Return the actor ID that starts the rollout.

        Example: return "player_0"
        """
        pass

    @abstractmethod
    def get_next_actor(self, state: State) -> str:
        """
        Return the actor ID for the next turn.

        Example: Round-robin through players
        """
        pass

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
            actor_id: The actor who will respond (e.g., "player_0")
            state: Current game state with trajectory and extras

        Returns:
            Messages list with system prompt and user content
        """
        pass

    async def on_turn_complete(self, state: State) -> None:
        """
        Update game state after a turn completes.

        This is called AFTER the model response is stored in trajectory.
        Use this for game logic:
        - Update scores, counters, flags
        - Check win conditions
        - Parse and validate actions

        The last turn's info is in state["trajectory"][-1]:
        - ["completion"][-1]["content"]: The model's response text
        - ["extras"]["actor_id"]: Which actor just responded

        Args:
            state: Current game state (mutate extras as needed)
        """
        pass

    # -------------------------------------------------------------------------
    # State Setup
    # -------------------------------------------------------------------------

    async def setup_state(self, state: State) -> State:
        """Initialize multi-agent state fields."""
        state = await super().setup_state(state)
        state["extras"] = state.get("extras", {})
        state["extras"]["current_actor_id"] = None
        return state

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
        """
        return []

    # -------------------------------------------------------------------------
    # Trajectory Management
    # -------------------------------------------------------------------------

    async def add_trajectory_step(
        self, state: State, trajectory_step: TrajectoryStep
    ) -> None:
        """Tag trajectory step with actor_id."""
        current_actor_id = state["extras"].get("current_actor_id")
        if current_actor_id:
            trajectory_step["extras"]["actor_id"] = current_actor_id
            # Copy trainability from Actor to step
            actor = self.get_actor(current_actor_id)
            trajectory_step["extras"]["is_trainable"] = actor.is_trainable
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
           a. Determine current actor
           b. Build prompt via build_actor_prompt()
           c. Get model response
           d. Store in trajectory
           e. Process via on_turn_complete()
        3. Return final state
        """
        state = await self.init_state(input, client, model, sampling_args)
        try:
            state = await self.setup_state(state)
        except vf.Error as e:
            state["error"] = e
            return state

        # Determine first actor
        state["extras"]["current_actor_id"] = self.get_initial_actor(state)

        while not await self.is_completed(state):
            actor_id = state["extras"]["current_actor_id"]

            try:
                # 1. Build prompt for this actor
                prompt_messages = await self.build_actor_prompt(actor_id, state)

                # 2. Get model response
                response = await self.get_model_response(state, prompt_messages)

                # 3. Store in trajectory (tags with actor_id)
                await self.add_model_response(state, prompt_messages, response)

                # 4. Process turn (game logic)
                await self.on_turn_complete(state)

                # 5. Determine next actor (if game continues)
                if not await self.is_completed(state):
                    state["extras"]["current_actor_id"] = self.get_next_actor(state)

            except vf.OverlongPromptError:
                state["prompt_too_long"] = True
                state["is_truncated"] = True
                break
            except vf.Error as e:
                state["error"] = e
                break

        await self.render_completion(state)
        return state
