"""
Multi-agent environment with turn order management and hierarchical spawning.

This module provides the base class for multi-agent RL environments, extending
MultiTurnEnv with support for:
- Multiple actors with distinct system prompts and sampling args
- Turn order management via get_initial_actor() / get_next_actor()
- Per-actor trajectory tagging for credit assignment
- Per-actor state splitting for individual reward computation
- Hierarchical episode spawning via Protocol.spawn()

Key concepts:
- Actor: A trainable entity with its own system prompt (defined in actor.py)
- Protocol: Wires actors to environments, enables spawning (defined in protocol.py)
- State splitting: One game state → multiple actor states for per-actor rewards

"""

from __future__ import annotations

import uuid
from abc import abstractmethod
from typing import TYPE_CHECKING, Any

from datasets import Dataset
from openai import AsyncOpenAI

import verifiers as vf
from verifiers.envs.multiturn_env import MultiTurnEnv

from verifiers.envs.protocol import GenerateResult
from verifiers.types import (
    Messages,
    RolloutInput,
    SamplingArgs,
    State,
    TrajectoryStep,
)
from verifiers.utils.message_utils import concat_messages
from verifiers.utils.async_utils import maybe_semaphore
from verifiers.utils.eval_utils import save_rollout_results
from verifiers.utils.response_utils import (
    parse_is_truncated,
    parse_response_messages,
    parse_response_tokens,
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
    - env_response(): Game logic between turns

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
    protocol: "Protocol"

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
    # Turn Management (Abstract Methods)
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
        Return actor IDs that can act this turn.

        Default: Single actor (standard alternating turns).
        Override for simultaneous moves (e.g., RPS returns ["player1", "player2"]).
        """
        return [self.get_next_actor(state)]

    def get_actor(self, actor_id: str) -> "Actor":
        """Get an actor by ID from Protocol."""
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
    # Trajectory Management
    # -------------------------------------------------------------------------

    async def add_trajectory_step(
        self, state: State, trajectory_step: TrajectoryStep
    ) -> None:
        """
        Add trajectory step, tagging with current actor for credit assignment.

        This tagging is critical for create_actor_states() which filters
        the trajectory by actor_id to split into per-actor states.
        """
        current_actor_id = state["extras"]["current_actor_id"]

        if current_actor_id:
            # Tag this step with who generated it
            if "extras" not in trajectory_step:
                trajectory_step["extras"] = {}
            trajectory_step["extras"]["actor_id"] = current_actor_id

            # Record in history: "actor X spoke at turn Y"
            turn_index = len(state["trajectory"])
            state["extras"]["actor_history"].append((current_actor_id, turn_index))

        await super().add_trajectory_step(state, trajectory_step)

    async def add_model_response(
        self,
        state: State,
        prompt_messages: Messages,
        response: Any,
    ) -> None:
        """
        Parse API response and add to trajectory with actor_id tag.

        Extracts completion text, truncation status, and token data,
        then creates a TrajectoryStep tagged with the current actor.
        """
        current_actor_id = state["extras"]["current_actor_id"]

        # Parse the raw API response
        completion_messages = await parse_response_messages(response, self.message_type)
        response_is_truncated = await parse_is_truncated(response, self.message_type)
        tokens = await parse_response_tokens(
            response, self.message_type, self.max_seq_len
        )
        is_truncated = response_is_truncated or (
            tokens is not None and bool(tokens.get("is_truncated"))
        )

        # Build trajectory step with actor tag
        trajectory_step = TrajectoryStep(
            prompt=prompt_messages,
            completion=completion_messages,
            response=response,
            tokens=tokens,
            reward=None,      # Filled by rubric later
            advantage=None,   # Filled by GRPO later
            is_truncated=is_truncated,
            trajectory_id=state["trajectory_id"],
            extras={"actor_id": current_actor_id},  # Critical for splitting
        )
        await self.add_trajectory_step(state, trajectory_step)

    # -------------------------------------------------------------------------
    # Direct Actor Invocation
    # -------------------------------------------------------------------------

    async def call_actor(
        self,
        actor_id: str,
        messages: Messages,
        state: State,
        sampling_args: SamplingArgs | None = None,
    ) -> GenerateResult:
        """
        Directly call a specific actor outside the standard rollout loop.

        Useful for custom rollouts (like RPS simultaneous moves) or
        spawning patterns. Handles system prompt injection and sampling
        args merging automatically.
        """
        actor = self.get_actor(actor_id)

        # Mark who is speaking
        state["extras"]["current_actor_id"] = actor_id

        # Build messages with actor's system prompt
        system_msg = actor.get_system_message()
        actor_messages = [system_msg] if system_msg else []
        actor_messages.extend(messages)

        # Merge sampling args: environment → actor → call overrides
        merged_args = actor.merge_sampling_args(state.get("sampling_args") or {})
        if sampling_args:
            merged_args.update(sampling_args)

        # Call model and record in trajectory
        response = await self.get_model_response(state, actor_messages, sampling_args=merged_args)
        await self.add_model_response(state, actor_messages, response)

        return GenerateResult(
            actor_id=actor_id,
            state=state,
            is_trainable=actor.is_trainable,
            episode_id=state["extras"]["episode_id"],
            parent_episode_id=state["extras"]["parent_episode_id"],
        )

    # -------------------------------------------------------------------------
    # Prompt Building
    # -------------------------------------------------------------------------

    async def get_prompt_messages(self, state: State) -> Messages:
        """
        Build prompt messages, injecting current actor's system prompt.

        For each turn:
        1. First turn: Use initial prompt from dataset
        2. Later turns: Concatenate previous turn + env_response()
        3. Always: Replace/prepend system prompt for current actor

        This ensures each actor sees their own instructions regardless
        of what accumulated in the conversation context.
        """
        current_actor_id = state["extras"]["current_actor_id"]

        # Build base messages
        if len(state["trajectory"]) == 0:
            # First turn: start with dataset prompt
            messages = list(state["prompt"])  # Copy to avoid mutation
        else:
            # Later turns: build from previous turn + env_response
            prev_turn_prompt = state["trajectory"][-1]["prompt"]
            prev_turn_completion = state["trajectory"][-1]["completion"]
            messages = concat_messages([prev_turn_prompt, prev_turn_completion])
            env_response = await self.env_response(messages, state)
            messages = concat_messages([messages, env_response])

        # Inject current actor's system prompt
        actor = self.get_actor(current_actor_id)
        system_prompt = actor.system_prompt

        if messages and messages[0].get("role") == "system":
            # Replace existing system prompt with actor's
            messages[0] = {"role": "system", "content": system_prompt}
        elif system_prompt:
            # Prepend actor's system prompt
            messages = [{"role": "system", "content": system_prompt}] + messages

        return messages

    # -------------------------------------------------------------------------
    # Main Rollout Loop
    # -------------------------------------------------------------------------

    async def rollout(
        self,
        input: RolloutInput,
        client: AsyncOpenAI,
        model: str,
        sampling_args: SamplingArgs | None = None,
    ) -> State:
        """
        Standard alternating-turn rollout using get_initial_actor/get_next_actor.

        Flow:
        1. init_state() - create base state from input
        2. setup_state() - initialize multi-agent fields
        3. Loop until stop condition:
           a. get_prompt_messages() - build prompt for current actor
           b. get_model_response() - call the model
           c. add_model_response() - store in trajectory (tagged)
           d. get_next_actor() - determine next speaker
        4. render_completion() - finalize state

        For simultaneous moves (like RPS), override this entirely and use
        get_active_actors() instead.
        """
        state = await self.init_state(input, client, model, sampling_args)

        try:
            state = await self.setup_state(state)
        except vf.Error as e:
            state["error"] = e
            return state

        # Set first actor
        initial_actor_id = self.get_initial_actor(state)
        state["extras"]["current_actor_id"] = initial_actor_id

        # Main loop
        while not await self.is_completed(state):
            try:
                current_actor_id = state["extras"]["current_actor_id"]

                # Build prompt and check for early termination
                prompt_messages = await self.get_prompt_messages(state)
                if state.get("final_env_response") is not None:
                    break

                # Get actor-specific sampling args and call model
                actor = self.get_actor(current_actor_id)
                merged_args = actor.merge_sampling_args(sampling_args or {})

                response = await self.get_model_response(state, prompt_messages, sampling_args=merged_args)
                await self.add_model_response(state, prompt_messages, response)

                # Advance to next actor
                next_actor_id = self.get_next_actor(state)
                state["extras"]["current_actor_id"] = next_actor_id

            except vf.Error as e:
                if isinstance(e, vf.OverlongPromptError):
                    state["prompt_too_long"] = True
                    state["is_truncated"] = True
                else:
                    state["error"] = e
                break

        await self.render_completion(state)
        return state

    # -------------------------------------------------------------------------
    # Abstract: Game Logic
    # -------------------------------------------------------------------------

    @abstractmethod
    async def env_response(
        self, messages: Messages, state: State, **kwargs
    ) -> Messages:
        """
        Generate environment response between turns.

        This is where game logic lives:
        - Process the last actor's output
        - Update game state (scores, flags, etc.)
        - Build the prompt for the next actor

        Return [] if no additional messages needed.
        Set state["final_env_response"] to terminate early.
        """
        pass

    # -------------------------------------------------------------------------
    # Actor Helpers
    # -------------------------------------------------------------------------

    def get_trainable_actors(self) -> list["Actor"]:
        """Get actors with is_trainable=True (will be trained)."""
        return [a for a in self.protocol.actors.values() if a.is_trainable]

    def get_frozen_actors(self) -> list["Actor"]:
        """Get actors with is_trainable=False (frozen, not trained)."""
        return [a for a in self.protocol.actors.values() if not a.is_trainable]

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
    SHARED_STATE_FIELDS = {
        "client",        # AsyncOpenAI API client
        "model",         # Model name string (e.g., "gpt-4o-mini")
        "timing",        # Performance metrics (generation_ms, scoring_ms)
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
        actor_state = State()

        # Copy shared fields by reference (not duplicated in memory)
        for key in parent_state.keys():
            if key in self.SHARED_STATE_FIELDS:
                actor_state[key] = parent_state[key]

        # Copy INPUT_FIELDS using dict.__setitem__ to bypass State forwarding
        # State.__setitem__ forwards writes for INPUT_FIELDS to input[key] if
        # input exists. We bypass this to store directly on actor_state.
        dict.__setitem__(actor_state, "answer", parent_state.get("answer", ""))
        dict.__setitem__(actor_state, "task", parent_state.get("task", ""))
        dict.__setitem__(actor_state, "example_id", parent_state.get("example_id", 0))
        dict.__setitem__(actor_state, "info", parent_state.get("info", {}))

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
            dict.__setitem__(actor_state, "prompt", prompt_ref)

            # Completion: Collect all responses across all turns
            all_completions = []
            for step in actor_trajectory:
                step_completion = step.get("completion", [])
                all_completions.extend(step_completion)
            dict.__setitem__(actor_state, "completion", all_completions)
        else:
            # No trajectory for this actor - use parent's prompt
            dict.__setitem__(actor_state, "prompt", parent_state.get("prompt", []))
            dict.__setitem__(actor_state, "completion", [])

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
    # Result Building
    # -------------------------------------------------------------------------

    def build_generate_result(self, state: State) -> GenerateResult:
        """
        Build a GenerateResult tree from completed state with children attached.

        Creates a hierarchical structure linking parent episodes to children,
        useful for hierarchical credit assignment (e.g., Proposer-Solver).
        """
        extras = state["extras"]

        # Determine root actor (first in history, or first declared)
        actor_history = extras["actor_history"]
        root_actor_id = actor_history[0][0] if actor_history else (self.actors[0] if self.actors else "unknown")

        # Get trainability (default True if actor not found)
        try:
            is_trainable = self.get_actor(root_actor_id).is_trainable
        except KeyError:
            is_trainable = True

        # Create root result
        root_result = GenerateResult(
            actor_id=root_actor_id,
            state=state,
            is_trainable=is_trainable,
            episode_id=extras["episode_id"],
            parent_episode_id=extras["parent_episode_id"],
        )

        # Attach child states as GenerateResults linked to parent
        parent_episode_id = extras["episode_id"]
        for child_state in state["child_states"]:
            child_extras = child_state.get("extras", {})
            child_actor_id = child_extras.get("current_actor_id", "unknown")

            try:
                child_trainable = self.get_actor(child_actor_id).is_trainable
            except KeyError:
                child_trainable = True

            child_result = GenerateResult(
                actor_id=child_actor_id,
                state=child_state,
                is_trainable=child_trainable,
                episode_id=child_extras.get("episode_id", uuid.uuid4().hex),
                parent_episode_id=parent_episode_id,
            )
            root_result.add_child(child_result)

        return root_result

    # -------------------------------------------------------------------------
    # Generate Override (Flattening & Per-Actor Scoring)
    # -------------------------------------------------------------------------

    async def generate(self, inputs, client, model, **kwargs):
        """
        Generate rollouts, flatten child_states, then score per-actor.

        The parent's generate() returns one state per game. For multi-agent,
        we need one state per actor per game for proper per-actor rewards.

        Flow:
        1. Intercept save/score options (we'll handle them after flattening)
        2. Run parent's generate() to get game states
        3. Flatten: Replace each game state with its child_states (per-actor)
        4. Score all flattened states together (proper GRPO grouping)
        5. Rebuild result arrays to match flattened states
        6. Update metadata and save

        Before: result["state"] = [game1, game2]
        After:  result["state"] = [g1_actor1, g1_actor2, g2_actor1, g2_actor2]
        """
        # Intercept options - we'll handle them after flattening
        original_save_results = kwargs.pop("save_results", False)
        push_to_hf_hub = kwargs.pop("push_to_hf_hub", False)
        hf_hub_dataset_name = kwargs.pop("hf_hub_dataset_name", None)

        # Disable parent's scoring - we'll score after flattening
        original_score_rollouts = getattr(self, 'score_rollouts', True)
        self.score_rollouts = False

        try:
            result = await super().generate(inputs, client, model, save_results=False, **kwargs)
        finally:
            self.score_rollouts = original_score_rollouts

        # Flatten: replace parent states with their child_states
        original_states = result.get("state", [])
        flattened_states = []
        needs_flatten = False

        for state in original_states:
            child_states = state.get("child_states", [])
            if child_states:
                flattened_states.extend(child_states)
                needs_flatten = True
            else:
                flattened_states.append(state)

        # Early exit if no flattening occurred
        if not needs_flatten:
            # Still need to score since we disabled it above
            if original_states and self.rubric and original_score_rollouts:
                score_sem = await maybe_semaphore(-1)
                await self.rubric.score_group(original_states, score_sem=score_sem)
                result["reward"] = [s.get("reward", 0.0) for s in original_states]
            if original_save_results:
                save_rollout_results(result, push_to_hf_hub, hf_hub_dataset_name)
            return result

        # Score all flattened states together for proper GRPO grouping
        if flattened_states and self.rubric and original_score_rollouts:
            score_sem = await maybe_semaphore(-1)
            await self.rubric.score_group(flattened_states, score_sem=score_sem)

        # Rebuild all result columns from flattened states
        prompts, completions, answers, tasks, example_ids, rewards, infos, actor_ids = [], [], [], [], [], [], [], []
        metrics: dict[str, list[float]] = {}

        for s in flattened_states:
            prompts.append(s.get("prompt", []))
            completions.append(s.get("completion"))
            answers.append(s.get("answer", ""))
            tasks.append(s.get("task", "default"))
            example_ids.append(s.get("example_id", 0))
            rewards.append(s.get("reward", 0.0))
            infos.append(s.get("info", {}))
            actor_ids.append(s.get("extras", {}).get("current_actor_id", "unknown"))

            state_metrics = s.get("metrics")
            if state_metrics:
                for name, value in state_metrics.items():
                    if name not in metrics:
                        metrics[name] = []
                    metrics[name].append(value)

        # Update result with flattened data
        result["state"] = flattened_states
        result["prompt"] = prompts
        result["completion"] = completions
        result["answer"] = answers
        result["task"] = tasks
        result["example_id"] = example_ids
        result["reward"] = rewards
        result["info"] = infos
        result["actor_id"] = actor_ids  # New field for multi-agent
        result["metrics"] = metrics

        # Update metadata to reflect flattened counts
        if "metadata" in result and rewards:
            num_examples = len(set(example_ids))
            result["metadata"]["avg_reward"] = sum(rewards) / len(rewards)
            result["metadata"]["num_examples"] = num_examples
            result["metadata"]["rollouts_per_example"] = len(flattened_states) // num_examples if num_examples > 0 else 1

        if original_save_results:
            save_rollout_results(result, push_to_hf_hub, hf_hub_dataset_name)

        return result
