"""
Multi-agent environment for turn-based games.

This module provides the base class for multi-agent RL environments, extending
MultiTurnEnv with support for:
- Multiple agents with distinct system prompts
- Turn order management via Protocol or get_initial_agent() / get_next_agent()
- Per-agent trajectory tagging for credit assignment

Key concepts:
- Agent: A participant with its own identity/prompt (defined in agent.py)
- Protocol: Defines turn order and interaction patterns (defined in protocol.py)

Environment Implementation:
- Subclasses implement these main hooks:
  - get_initial_agent(state): Who goes first (or use a Protocol)
  - get_next_agent(state): Who goes next (or use a Protocol)
  - build_agent_prompt(agent_id, state): Build fresh prompt for this agent
  - on_turn_complete(state): Update game state after each turn
"""

from abc import abstractmethod

import verifiers as vf
from verifiers.envs.agent import Agent
from verifiers.envs.multiturn_env import MultiTurnEnv
from verifiers.envs.protocol import Protocol
from verifiers.types import Messages, State, TrajectoryStep


class MultiAgentEnv(MultiTurnEnv):
    """
    Base class for multi-agent environments.

    Turn order can be specified either by:
    1. Passing a Protocol to __init__ (reusable turn logic)
    2. Implementing get_initial_agent() and get_next_agent() in subclass

    Subclasses must implement:
    - build_agent_prompt(): Build prompt for current agent

    Subclasses may optionally override:
    - on_turn_complete(): Game logic after each turn
    - get_initial_agent() / get_next_agent(): If not using a Protocol
    """

    # List of agent IDs this environment uses (e.g., ["player_0", "player_1"])
    # Subclasses should override this or set in __init__
    agents: list[str] = []

    def __init__(self, protocol: Protocol | None = None, **kwargs):
        """
        Initialize multi-agent environment.

        Args:
            protocol: Optional Protocol for turn order. If not provided,
                      subclass must implement get_initial_agent/get_next_agent.
            **kwargs: Passed to MultiTurnEnv
        """
        super().__init__(**kwargs)
        self._protocol = protocol
        self._agent_registry: dict[str, Agent] = {}

    def register_agent(self, agent: Agent) -> None:
        """Register an Agent for lookup by get_agent()."""
        self._agent_registry[agent.id] = agent
        if agent.id not in self.agents:
            self.agents.append(agent.id)

    def get_agent(self, agent_id: str) -> Agent:
        """Get an agent by ID."""
        if agent_id not in self._agent_registry:
            raise KeyError(
                f"Agent '{agent_id}' not found. Did you call register_agent()?"
            )
        return self._agent_registry[agent_id]

    # -------------------------------------------------------------------------
    # Turn Management
    # -------------------------------------------------------------------------

    def get_initial_agent(self, state: State) -> str:
        """
        Return the agent ID that starts the rollout.

        Default: delegates to Protocol if provided.
        Override in subclass if not using a Protocol.
        """
        if self._protocol:
            return self._protocol.get_initial_agent(state)
        raise NotImplementedError("Provide a Protocol or override get_initial_agent()")

    def get_next_agent(self, state: State) -> str:
        """
        Return the agent ID for the next turn.

        Default: delegates to Protocol if provided.
        Override in subclass if not using a Protocol.
        """
        if self._protocol:
            return self._protocol.get_next_agent(state)
        raise NotImplementedError("Provide a Protocol or override get_next_agent()")

    # -------------------------------------------------------------------------
    # Agent Prompt Building (Subclasses Implement This)
    # -------------------------------------------------------------------------

    @abstractmethod
    async def build_agent_prompt(self, agent_id: str, state: State) -> Messages:
        """
        Build the prompt for the given agent's turn.

        This is called BEFORE the model generates a response.
        Build a fresh prompt with whatever context this agent needs.

        Args:
            agent_id: The agent who will respond (e.g., "player_0")
            state: Current game state with trajectory and extras

        Returns:
            Messages list with system prompt and user content
        """
        pass

    # -------------------------------------------------------------------------
    # Game Logic Hook
    # -------------------------------------------------------------------------

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
        - ["extras"]["agent_id"]: Which agent just responded

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
        state["extras"]["current_agent_id"] = None
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
        """Tag trajectory step with agent_id."""
        current_agent_id = state["extras"].get("current_agent_id")
        if current_agent_id:
            trajectory_step["extras"]["agent_id"] = current_agent_id
            # Copy trainability from Agent to step
            agent = self.get_agent(current_agent_id)
            trajectory_step["extras"]["is_trainable"] = agent.is_trainable
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
           a. Determine current agent
           b. Build prompt via build_agent_prompt()
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

        # Determine first agent
        state["extras"]["current_agent_id"] = self.get_initial_agent(state)

        while not await self.is_completed(state):
            agent_id = state["extras"]["current_agent_id"]

            try:
                # 1. Build prompt for this agent
                prompt_messages = await self.build_agent_prompt(agent_id, state)

                # 2. Get model response
                response = await self.get_model_response(state, prompt_messages)

                # 3. Store in trajectory (tags with agent_id)
                await self.add_model_response(state, prompt_messages, response)

                # 4. Process turn (game logic)
                await self.on_turn_complete(state)

                # 5. Determine next agent (if game continues)
                if not await self.is_completed(state):
                    state["extras"]["current_agent_id"] = self.get_next_agent(state)

            except vf.OverlongPromptError:
                state["prompt_too_long"] = True
                state["is_truncated"] = True
                break
            except vf.Error as e:
                state["error"] = e
                break

        await self.render_completion(state)
        return state
