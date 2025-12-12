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

    async def get_prompt_ids(self, state: State, client: AsyncOpenAI) -> list[int]:
        async def tokenize_vllm(
            prompt: Messages, client: AsyncOpenAI, model: str
        ) -> list[int]:
            """Tokenize a prompt using the vLLM tokenize API."""
            http_client = client._client
            base_url = str(client.base_url).replace("/v1/", "")
            tokenize_url = base_url + "/tokenize"  # vLLM specific
            try:
                response = await http_client.post(
                    tokenize_url, json={"model": model, "messages": prompt}
                )
                if not response.status_code == 200:
                    raise Exception(f"Failed to tokenize prompt: {response.text}")
                return response.json()["tokens"]
            except Exception as e:
                raise vf.ModelError(e)

        if len(state["trajectory"]) == 0:
            return await tokenize_vllm(state["prompt"], client, state["model"])
        else:
            prev_turn_prompt = state["trajectory"][-1]["prompt"]
            prev_turn_completion = state["trajectory"][-1]["completion"]
            prev_turn_tokens = state["trajectory"][-1]["tokens"]
            assert prev_turn_tokens is not None
            prev_turn_prompt_ids = prev_turn_tokens["prompt_ids"]
            prev_turn_completion_ids = prev_turn_tokens["completion_ids"]
            messages = concat_messages([prev_turn_prompt, prev_turn_completion])
            env_response = await self.env_response(messages, state)
            env_response_ids = await tokenize_vllm(env_response, client, state["model"])
            return prev_turn_prompt_ids + prev_turn_completion_ids + env_response_ids

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
        use_token_prompts: bool | None = None,
    ) -> State:
        """
        Generate a multi-turn rollout with the environment.

        If use_token_prompts is set, the environment will prepare a token
        prompt. This requires that the inference server supports token-in
        prompts. Currently, this is a hand-crated feature for PRIME-RL's vLLM
        server extension, and is not recommended for general use outside of
        PRIME-RL.
        """
        use_token_prompts = (
            use_token_prompts
            if use_token_prompts is not None
            else self.use_token_prompts
        )
        state = await self.init_state(input, client, model, sampling_args)
        try:
            state = await self.setup_state(state)
        except vf.Error as e:
            state["error"] = e
        while not await self.is_completed(state):
            try:
                prompt_messages = await self.get_prompt_messages(state)
                if use_token_prompts:
                    prompt_ids = await self.get_prompt_ids(state, client)
                    response = await self.get_model_response_with_tokens(
                        state, prompt_messages, prompt_ids
                    )
                else:
                    response = await self.get_model_response(state, prompt_messages)
                await self.add_model_response(state, prompt_messages, response)
            except vf.Error as e:
                state["error"] = e
        return state
