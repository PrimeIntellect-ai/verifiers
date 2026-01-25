"""
MultiAgentEnv: Multi-turn environment with multiple actors.

Extends MultiTurnEnv to add actor management - tracking which actor
is currently active, and rewriting system prompts accordingly.

Dataset lives on the Protocol, not the environment. MultiAgentEnv provides
a dummy dataset to satisfy the base class requirement.
"""
from abc import abstractmethod
from typing import TYPE_CHECKING

from datasets import Dataset
from openai import AsyncOpenAI

import verifiers as vf
from verifiers.types import (
    Messages,
    ModelResponse,
    RolloutInput,
    SamplingArgs,
    State,
    TrajectoryStep,
)
from verifiers.utils.message_utils import concat_messages
from verifiers.utils.response_utils import (
    parse_is_truncated,
    parse_response_messages,
    parse_response_tokens,
)

from .multiturn_env import MultiTurnEnv

if TYPE_CHECKING:
    from .protocol import Protocol


def _dummy_dataset() -> Dataset:
    """Create a minimal dummy dataset to satisfy base class requirement."""
    return Dataset.from_dict({
        "example_id": [0],
        "prompt": [[{"role": "user", "content": "dummy"}]],
        "answer": [""],
    })


class MultiAgentEnv(MultiTurnEnv):
    """
    Multi-turn environment with multiple actors.

    Extends MultiTurnEnv to add:
    - self.current_actor tracking
    - get_initial_actor() / get_next_actor() for turn management
    - System prompt rewriting based on current actor (naive approach, needs more sophisticated rearchitecture)
    - actor_id stored in TrajectoryStep.extras

    Subclasses must implement:
    - get_initial_actor(state) -> str
    - get_next_actor(state) -> str
    - env_response(messages, state) -> Messages (inherited from MultiTurnEnv)
    """

    # Subclasses declare which actors they use
    actors: list[str] = []

    # Current actor tracked as instance field (set during rollout)
    current_actor: str = ""

    # Protocol reference (injected by Protocol.__init__)
    protocol: "Protocol"

    def __init__(self, **kwargs):
        """
        Initialize MultiAgentEnv with a dummy dataset.

        Dataset lives on the Protocol, not the environment. The dummy dataset
        satisfies the base class requirement but is never actually used.
        """
        # Inject dummy dataset if none provided (protocol manages the real dataset)
        if "dataset" not in kwargs and "eval_dataset" not in kwargs:
            kwargs["dataset"] = _dummy_dataset()
        super().__init__(**kwargs)

    @abstractmethod
    def get_initial_actor(self, state: State) -> str:
        """Return the actor ID that starts the rollout."""
        pass

    @abstractmethod
    def get_next_actor(self, state: State) -> str:
        """Return the actor ID for the next turn."""
        pass

    async def get_prompt_messages(self, state: State) -> Messages:
        """
        Build prompt messages, rewriting system prompt for current actor.

        Gets system prompt from protocol.get_actor(self.current_actor).
        """
        # Get base messages from parent logic
        if len(state["trajectory"]) == 0:
            messages = list(state["prompt"])  # copy
        else:
            prev_turn_prompt = state["trajectory"][-1]["prompt"]
            prev_turn_completion = state["trajectory"][-1]["completion"]
            messages = concat_messages([prev_turn_prompt, prev_turn_completion])
            env_response = await self.env_response(messages, state)
            messages = concat_messages([messages, env_response])

        # Rewrite system prompt for current actor
        actor = self.protocol.get_actor(self.current_actor)

        if messages and messages[0].get("role") == "system":
            # Replace existing system prompt
            messages[0] = {"role": "system", "content": actor.system_prompt}
        elif actor.system_prompt:
            # Prepend system prompt
            messages = [{"role": "system", "content": actor.system_prompt}] + messages

        return messages

    async def add_model_response(
        self,
        state: State,
        prompt_messages: Messages,
        response: ModelResponse,
    ):
        """Add model response to trajectory, storing actor_id in extras."""
        completion_messages = await parse_response_messages(response, self.message_type)
        response_is_truncated = await parse_is_truncated(response, self.message_type)
        tokens = await parse_response_tokens(
            response, self.message_type, self.max_seq_len
        )
        is_truncated = response_is_truncated or (
            tokens is not None and bool(tokens.get("is_truncated"))
        )

        trajectory_step = TrajectoryStep(
            prompt=prompt_messages,
            completion=completion_messages,
            response=response,
            tokens=tokens,
            reward=None,
            advantage=None,
            is_truncated=is_truncated,
            trajectory_id=state["trajectory_id"],
            extras={"actor_id": self.current_actor},  # Store actor_id in extras
        )
        await self.add_trajectory_step(state, trajectory_step)

    async def rollout(
        self,
        input: RolloutInput,
        client: AsyncOpenAI,
        model: str,
        sampling_args: SamplingArgs | None = None,
    ) -> State:
        """
        Multi-agent rollout loop.

        Similar to MultiTurnEnv.rollout() but:
        1. Sets self.current_actor at start via get_initial_actor()
        2. Updates self.current_actor after each turn via get_next_actor()
        """
        state = await self.init_state(input, client, model, sampling_args)

        try:
            state = await self.setup_state(state)
        except vf.Error as e:
            state["error"] = e
            return state

        # Initialize current actor (instance field)
        self.current_actor = self.get_initial_actor(state)

        while not await self.is_completed(state):
            try:
                prompt_messages = await self.get_prompt_messages(state)
                if state.get("final_env_response") is not None:
                    continue
                response = await self.get_model_response(state, prompt_messages)
                await self.add_model_response(state, prompt_messages, response)

                # Update actor for next turn
                self.current_actor = self.get_next_actor(state)

            except vf.Error as e:
                if isinstance(e, vf.OverlongPromptError):
                    state["prompt_too_long"] = True
                    state["is_truncated"] = True
                else:
                    state["error"] = e

        await self.render_completion(state)
        return state


class SingleTurnMAEnv(MultiAgentEnv):
    """
    Single-turn multi-agent environment with exactly one actor.

    Similar to how SingleTurnEnv simplifies MultiTurnEnv, this class simplifies
    MultiAgentEnv for single-turn, single-actor use cases:
    - max_turns=1 (only one model response)
    - env_response raises NotImplementedError (no multi-turn interaction)
    - get_initial_actor/get_next_actor return the single declared actor

    Subclasses must declare exactly one actor in the `actors` list.
    """

    def __init__(self, **kwargs):
        """Initialize SingleTurnMAEnv with max_turns=1."""
        super().__init__(max_turns=1, **kwargs)
        # Validate single actor requirement
        if len(self.actors) != 1:
            raise ValueError(
                f"SingleTurnMAEnv requires exactly one actor, got {len(self.actors)}: {self.actors}"
            )

    async def env_response(
        self, messages: Messages, state: State, **kwargs
    ) -> Messages:
        """Not implemented - single turn means no environment responses."""
        raise NotImplementedError("env_response is not implemented for SingleTurnMAEnv")

    def get_initial_actor(self, state: State) -> str:
        """Return the single declared actor."""
        return self.actors[0]

    def get_next_actor(self, state: State) -> str:
        """Return the single declared actor (though this should never be called)."""
        return self.actors[0]
