
import logging
import asyncio
from abc import abstractmethod

from openai import AsyncOpenAI, BadRequestError

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
    parse_response_messages,
    parse_response_tokens,
)

logger = logging.getLogger(__name__)


class MultiTurnEnv(vf.Environment):
    def __init__(self, max_turns: int = -1, max_retries: int = 3, **kwargs):
        super().__init__(**kwargs)
        self.max_turns = max_turns
        self.max_retries = max_retries

    async def setup_state(self, state: State) -> State:
        return state

    @vf.stop
    async def prompt_too_long(self, state: State) -> bool:
        return state.get("prompt_too_long", False)

    @vf.stop
    async def max_turns_reached(self, state: State) -> bool:
        """Check if the maximum number of turns has been reached."""
        return len(state["trajectory"]) >= self.max_turns and self.max_turns > 0

    @vf.stop
    async def error_occurred(self, state: State) -> bool:
        """Stop if an unrecoverable error occurred after max retries."""
        return state.get("error") is not None

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
        if response is not None and response.id == "overlong-prompt":
            state["prompt_too_long"] = True
        completion_messages = await parse_response_messages(response, self.message_type)
        tokens = await parse_response_tokens(
            response, self.message_type, self.max_seq_len
        )
        trajectory_step = TrajectoryStep(
            prompt=prompt_messages,
            completion=completion_messages,
            response=response,
            tokens=tokens,
            reward=None,
            advantage=None,
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
        state = await self.setup_state(state)
        num_retries = 0
        while not await self.is_completed(state):
            # Save trajectory length to restore on error (avoids copying non-picklable objects
            # like AsyncOpenAI client, and preserves State class forwarding behavior)
            trajectory_len = len(state["trajectory"])
            prompt_messages = await self.get_prompt_messages(state)
            try:
                response = await self.get_model_response(
                    client,
                    model,
                    prompt_messages,
                    oai_tools=state["oai_tools"],
                    sampling_args=sampling_args,
                    message_type=self.message_type,
                )
                await self.add_model_response(state, prompt_messages, response)
                num_retries = 0
            except BadRequestError as e:
                num_retries += 1
                logger.warning(
                    f"BadRequestError (attempt {num_retries}/{self.max_retries}): {e}"
                )
                if num_retries > self.max_retries:
                    # Gracefully fail instead of crashing
                    logger.warning(
                        f"Max retries ({self.max_retries}) exceeded, marking rollout as failed"
                    )
                    state["error"] = str(e)
                    # Create error message as fake completion so judge sees the error
                    error_completion = [
                        {"role": "assistant", "content": f"[Error: {e}]"}
                    ]
                    trajectory_step = TrajectoryStep(
                        prompt=prompt_messages,
                        completion=error_completion,
                        response=None,
                        tokens=None,
                        reward=None,
                        advantage=None,
                        extras={"error": str(e)},
                    )
                    state["trajectory"].append(trajectory_step)
                    # Don't break - let the loop check is_completed, which will trigger
                    # error_occurred stop condition and run cleanup properly
                    continue
                # Restore trajectory to before the failed attempt
                state["trajectory"] = state["trajectory"][:trajectory_len]
                state.pop("prompt_too_long", None)
                await asyncio.sleep(1)
                continue

        return state
