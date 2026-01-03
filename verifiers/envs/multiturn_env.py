import logging
from abc import abstractmethod

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

logger = logging.getLogger(__name__)


class MultiTurnEnv(vf.Environment):
    def __init__(self, max_turns: int = -1, **kwargs):
        super().__init__(**kwargs)
        self.max_turns = max_turns

    async def setup_state(self, state: State) -> State:
        return state

    @vf.stop(priority=100)  # high priority to always check for errors first
    async def has_error(self, state: State, **kwargs) -> bool:
        """Abrupts rollout early if an error has occurred."""
        return state.get("error") is not None

    @vf.stop
    async def prompt_too_long(self, state: State) -> bool:
        return state.get("prompt_too_long", False)

    @vf.stop
    async def max_turns_reached(self, state: State) -> bool:
        """Check if the maximum number of turns has been reached."""
        return len(state["trajectory"]) >= self.max_turns and self.max_turns > 0

    @abstractmethod
    async def env_response(
        self, messages: Messages, state: State, **kwargs
    ) -> Messages:
        """
        Generate a response from the environment.
        """
        pass

    async def get_prompt_messages(self, state: State) -> Messages:
        if len(state["trajectory"]) == 0:
            return state["prompt"]
        else:
            prev_turn_prompt = state["trajectory"][-1]["prompt"]
            prev_turn_completion = state["trajectory"][-1]["completion"]
            messages = concat_messages([prev_turn_prompt, prev_turn_completion])
            env_response = await self.env_response(messages, state)
            return concat_messages([messages, env_response])

    async def add_model_response(
        self,
        state: State,
        prompt_messages: Messages,
        response: ModelResponse,
    ):
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
            extras={},
        )
        trajectory_step["completion"] = completion_messages
        state["trajectory"].append(trajectory_step)

    async def rollout(
        self,
        input: RolloutInput,
        client: AsyncOpenAI,
        model: str,
        sampling_args: SamplingArgs | None = None,
    ) -> State:
        """
        Generate a multi-turn rollout with the environment.
        """
        state = await self.init_state(input, client, model, sampling_args)
        try:
            state = await self.setup_state(state)
        except vf.Error as e:
            state["error"] = e
        while not await self.is_completed(state):
            try:
                prompt_messages = await self.get_prompt_messages(state)
                response = await self.get_model_response(state, prompt_messages)
                await self.add_model_response(state, prompt_messages, response)
            except vf.Error as e:
                if isinstance(e, vf.OverlongPromptError):
                    state["prompt_too_long"] = True
                    state["is_truncated"] = True
                else:
                    state["error"] = e
        return state
