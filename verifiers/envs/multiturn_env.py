import asyncio
import logging
from abc import abstractmethod
from typing import final

from anthropic import AsyncAnthropic
from openai import AsyncOpenAI

import verifiers as vf
from verifiers.types import (
    Messages,
    Response,
    RolloutInput,
    SamplingArgs,
    State,
    TextMessage,
    TrajectoryStep,
)
from verifiers.utils.message_utils import concat_messages
from verifiers.utils.response_utils import (
    parse_response_message,
    parse_response_tokens,
)

logger = logging.getLogger(__name__)


class MultiTurnMonitorRubric(vf.Rubric):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.add_metric(self.num_turns)

    async def num_turns(self, state: State) -> int:
        return len(state["trajectory"])


class MultiTurnEnv(vf.Environment):
    def __init__(self, max_turns: int = -1, **kwargs):
        super().__init__(**kwargs)
        self.max_turns = max_turns

        self.add_rubric(MultiTurnMonitorRubric())

    @abstractmethod
    async def env_response(
        self, messages: Messages, state: State, **kwargs
    ) -> Messages | str:
        """
        Generate a response from the environment.
        """
        pass

    @vf.stop(priority=100)  # always check for errors first
    async def has_error(self, state: State, **kwargs) -> bool:
        return state.get("error") is not None

    @vf.stop
    async def prompt_too_long(self, state: State) -> bool:
        return state.get("prompt_too_long", False)

    @vf.stop
    async def max_turns_reached(self, state: State) -> bool:
        return len(state["trajectory"]) >= self.max_turns and self.max_turns > 0

    @vf.stop
    async def has_final_env_response(self, state: State) -> bool:
        """Check if env_response signaled termination via final_env_response."""
        return state.get("final_env_response") is not None

    async def setup_state(self, state: State) -> State:
        """Override to add environment-specific state fields."""
        return state

    async def get_prompt_messages(self, state: State) -> Messages:
        """Override for rollouts with non-linear message sequences."""
        def as_messages(value: Messages | str) -> Messages:
            if isinstance(value, str):
                return [TextMessage(content=value)]
            return value

        if len(state["trajectory"]) == 0:
            return as_messages(state["prompt"])
        prev_turn_prompt = as_messages(state["trajectory"][-1]["prompt"])
        prev_turn_completion = as_messages(state["trajectory"][-1]["completion"])
        messages = concat_messages([prev_turn_prompt, prev_turn_completion])
        env_response = await self.env_response(messages, state)
        env_response_messages = as_messages(env_response)
        return concat_messages([messages, env_response_messages])

    async def render_completion(self, state: State):
        """Override for rollouts with non-linear message sequences."""
        def as_messages(value: Messages | str) -> Messages:
            if isinstance(value, str):
                return [TextMessage(content=value)]
            return value

        if len(state["trajectory"]) == 0:
            state["completion"] = []
            return
        last_prompt = as_messages(state["trajectory"][-1]["prompt"])
        last_completion = as_messages(state["trajectory"][-1]["completion"])
        full_conversation = concat_messages([last_prompt, last_completion])
        if state.get("final_env_response"):
            full_conversation = concat_messages([full_conversation, as_messages(state["final_env_response"])])
        prompt_messages = as_messages(state["prompt"])
        state["completion"] = full_conversation[len(prompt_messages) :]

    async def add_trajectory_step(self, state: State, trajectory_step: TrajectoryStep):
        """Override to set intermediate rewards, advantages, or extra metadata."""
        state["trajectory"].append(trajectory_step)

    async def add_model_response(
        self,
        state: State,
        prompt_messages: Messages,
        response: Response,
    ):
        completion_messages = await parse_response_message(response)
        tokens = await parse_response_tokens(response, self.max_seq_len)
        response_is_truncated = response.message.is_truncated or False
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
        await self.add_trajectory_step(state, trajectory_step)

    @final
    async def rollout(
        self,
        input: RolloutInput,
        client: AsyncOpenAI | AsyncAnthropic,
        model: str,
        sampling_args: SamplingArgs | None = None,
    ) -> State:
        state = await self.init_state(input, client, model, sampling_args)
        try:
            try:
                state = await self.setup_state(state)
            except vf.Error as e:
                state["error"] = e
            # checks all @vf.stop methods, runs all @vf.cleanup methods if any are True
            while not await self.is_completed(state):
                try:
                    prompt_messages = await self.get_prompt_messages(state)
                    if state.get("final_env_response") is not None:
                        continue
                    response = await self.get_model_response(state, prompt_messages)
                    await self.add_model_response(state, prompt_messages, response)
                except vf.Error as e:
                    if isinstance(e, vf.OverlongPromptError):
                        state["prompt_too_long"] = True
                        state["is_truncated"] = True
                    else:
                        state["error"] = e
            await self.render_completion(state)
            return state
        except asyncio.CancelledError:
            await self._cleanup(state)
            raise
