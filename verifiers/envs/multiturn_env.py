import logging
from abc import abstractmethod
from typing import Optional

from openai import AsyncOpenAI, BaseModel
from openai.types.chat import ChatCompletionToolParam

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

    async def get_prompt_messages_and_ids(
        self, state: State, client: AsyncOpenAI
    ) -> tuple[Messages, list[int]]:
        async def tokenize_vllm(
            messages: Messages,
            tools: list[ChatCompletionToolParam] | None,
            model: str,
            client: AsyncOpenAI,
            default_body: dict = {},
        ) -> list[int]:
            """Tokenize a prompt using the vLLM tokenize API."""
            if getattr(self, "tokens_client", None) is None:
                url_without_v1 = str(client.base_url).replace("/v1/", "")
                tokens_client: AsyncOpenAI = client.copy(base_url=url_without_v1)
                setattr(self, "tokens_client", tokens_client)
            else:
                tokens_client = getattr(self, "tokens_client")
            body = dict(
                model=model,
                messages=messages,
                tools=tools,
                **default_body,
            )

            # Copy from vllm/entrypoints/openai/protocol.py
            class TokenizeResponse(BaseModel):
                count: int
                max_model_len: int
                tokens: list[int]
                token_strs: Optional[list[str]] = None

            try:
                tokenize_response = await tokens_client.post(
                    "/tokenize", body=body, cast_to=TokenizeResponse
                )
                return tokenize_response.tokens
            except Exception as e:
                raise vf.ModelError(e)

        if len(state["trajectory"]) == 0:
            prompt_messages = state["prompt"]
            prompt_ids = await tokenize_vllm(
                messages=state["prompt"],
                tools=state["oai_tools"],
                model=state["model"],
                client=client,
            )
            return prompt_messages, prompt_ids
        else:
            prev_turn_prompt = state["trajectory"][-1]["prompt"]
            prev_turn_completion = state["trajectory"][-1]["completion"]
            prev_turn_tokens = state["trajectory"][-1]["tokens"]
            assert prev_turn_tokens is not None
            prev_turn_prompt_ids = prev_turn_tokens["prompt_ids"]
            prev_turn_completion_ids = prev_turn_tokens["completion_ids"]

            # We build the env_response_ids by tokenizing a (a) all previous
            # messages + env response (b) all previous messages and determine
            # the env_response_ids as the difference between those two. The main
            # reason to do this (rather than just tokenizing the env response)
            # is to avoid the edge case where the chat template adds prefix
            # tokens to the env response if and only if the message follows
            # other messages
            #
            # NOTE: This only works if the chat template is incremental, i.e. if
            # (b) is a prefix of (a). For non-incremental chat templates, use
            # branching rollouts.
            messages = concat_messages([prev_turn_prompt, prev_turn_completion])
            env_response = await self.env_response(messages, state)
            messages_and_env_response_ids = await tokenize_vllm(
                messages=concat_messages([messages, env_response]),
                tools=state["oai_tools"],
                model=state["model"],
                client=client,
            )
            messages_ids = await tokenize_vllm(
                messages=messages,
                tools=state["oai_tools"],
                model=state["model"],
                client=client,
                default_body=dict(add_generation_prompt=False),
            )
            assert messages_and_env_response_ids[: len(messages_ids)] == messages_ids, (
                f"Detected violation in incremental tokenization assumption\n{messages_and_env_response_ids[: len(messages_ids)]}\n{messages_ids}"
            )
            env_response_ids = messages_and_env_response_ids[len(messages_ids) :]

            # We build prev_turn_ids by concatenating the prev_turn_ids and
            # completion_ids, as returned by the inference engine. We then
            # tokenize the previous turn's messages and check for any suffix
            # tokens that might be missing. These are tokens that are added by
            # the chat template after messages, but not generated by the model,
            # i.e. they will be part of messages_ids (from the tokenizer) but
            # not of prev_turn_ids (from the engine). To not train OOD w.r.t.
            # the chat template, we add these suffix tokens to prev_turn_ids.
            prev_turn_ids = prev_turn_prompt_ids + prev_turn_completion_ids
            # Find the index of the end of message token in messages_ids
            #
            # NOTE: This assumes that the final token in prev_turn_ids is the
            # end of message token. This *should* be a safe assumption because
            # it is the stop token for chat models.
            eom_idxs = [i for i, x in enumerate(messages_ids) if x == prev_turn_ids[-1]]
            if eom_idxs:
                missing_suffix = messages_ids[eom_idxs[-1] + 1 :]
                prev_turn_ids += missing_suffix

            prompt_messages = concat_messages([messages, env_response])
            prompt_ids = prev_turn_ids + env_response_ids

            return prompt_messages, prompt_ids

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
                if use_token_prompts:
                    (
                        prompt_messages,
                        prompt_ids,
                    ) = await self.get_prompt_messages_and_ids(state, client)
                    response = await self.get_model_response_with_tokens(
                        state, prompt_messages, prompt_ids
                    )
                else:
                    prompt_messages = await self.get_prompt_messages(state)
                    response = await self.get_model_response(state, prompt_messages)
                await self.add_model_response(state, prompt_messages, response)
            except vf.Error as e:
                state["error"] = e
        return state
