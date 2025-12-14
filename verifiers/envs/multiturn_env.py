import asyncio
import logging
from abc import abstractmethod
from typing import Literal

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
    tokenize_local,
    tokenize_vllm,
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
        self, state: State, client: AsyncOpenAI, exact_tokenization: bool = True
    ) -> tuple[Messages, list[int]]:
        assert state["tokenize_method"] is not None
        tokenize = (
            tokenize_vllm if state["tokenize_method"] == "vllm" else tokenize_local
        )
        if len(state["trajectory"]) == 0:
            logger.warning(
                "Calling `get_prompt_messages_and_ids` on the initial prompt. This creates unnecessary overhead, and should not happen. It is save to directly call /v1/chat/completions because no retokenization can happen on the initial prompt."
            )
            prompt_messages = state["prompt"]
            prompt_ids = await tokenize(
                client=client,
                messages=state["prompt"],
                tools=state["oai_tools"],
                model=state["model"],
            )
            return prompt_messages, prompt_ids
        else:
            prev_turn_prompt = state["trajectory"][-1]["prompt"]
            prev_turn_completion = state["trajectory"][-1]["completion"]
            prev_turn_tokens = state["trajectory"][-1]["tokens"]
            assert prev_turn_tokens is not None
            prev_turn_prompt_ids = prev_turn_tokens["prompt_ids"]
            prev_turn_completion_ids = prev_turn_tokens["completion_ids"]

            messages = concat_messages([prev_turn_prompt, prev_turn_completion])
            env_response = await self.env_response(messages, state)
            messages_and_env_response = concat_messages([messages, env_response])

            if not exact_tokenization:
                env_response_ids = await tokenize(
                    client=client,
                    messages=env_response,
                    tools=state["oai_tools"],
                    model=state["model"],
                )
            else:
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
                messages_ids_task = tokenize(
                    client=client,
                    messages=messages,
                    tools=state["oai_tools"],
                    model=state["model"],
                    extra_kwargs=dict(add_generation_prompt=False),
                )
                messages_and_env_response_ids_task = tokenize(
                    client=client,
                    messages=messages_and_env_response,
                    tools=state["oai_tools"],
                    model=state["model"],
                )

                # Parallelize the two tokenization calls
                messages_ids, messages_and_env_response_ids = await asyncio.gather(
                    messages_ids_task,
                    messages_and_env_response_ids_task,
                )

                assert (
                    messages_and_env_response_ids[: len(messages_ids)] == messages_ids
                ), (
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
                #
                # NOTE: This assumes that the final token in prev_turn_ids is the
                # end of message token. This *should* be a safe assumption because
                # it is the stop token for chat models.
                def find_last_index(lst: list[int], value: int) -> int:
                    for i in range(len(lst) - 1, -1, -1):
                        if lst[i] == value:
                            return i
                    raise ValueError

                try:
                    maybe_eom_token = prev_turn_completion_ids[-1]
                    eom_idx = find_last_index(messages_ids, maybe_eom_token)
                    missing_suffix = messages_ids[eom_idx + 1 :]
                    prev_turn_completion_ids += missing_suffix
                except ValueError:
                    # end of message token not found, so we don't need to add any suffix tokens
                    pass

            prompt_messages = messages_and_env_response
            prompt_ids = (
                prev_turn_prompt_ids + prev_turn_completion_ids + env_response_ids
            )

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
        use_token_prompts: bool = False,
        tokenize_method: Literal["local", "vllm"] | None = None,
    ) -> State:
        """
        Generate a multi-turn rollout with the environment.
        """
        state = await self.init_state(
            input, client, model, sampling_args, use_token_prompts, tokenize_method
        )
        try:
            state = await self.setup_state(state)
        except vf.Error as e:
            state["error"] = e
        while not await self.is_completed(state):
            try:
                if state["use_token_prompts"] and len(state["trajectory"]) > 0:
                    (
                        prompt_messages,
                        prompt_ids,
                    ) = await self.get_prompt_messages_and_ids(state, client)
                    response = await self.get_model_response(
                        state, prompt_messages, prompt_ids=prompt_ids
                    )
                else:
                    prompt_messages = await self.get_prompt_messages(state)
                    response = await self.get_model_response(state, prompt_messages)
                await self.add_model_response(state, prompt_messages, response)
            except vf.Error as e:
                state["error"] = e
        return state
