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
        self, state: State, client: AsyncOpenAI
    ) -> tuple[Messages, list[int]]:
        assert state["tokenize_method"] is not None
        assert state["exact_tokenization"] is not None
        tokenize = (
            tokenize_vllm if state["tokenize_method"] == "vllm" else tokenize_local
        )
        if len(state["trajectory"]) == 0:
            logger.warning(
                "Calling `get_prompt_messages_and_ids` in first turn. This creates unnecessary overhead, and should not happen."
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
            prev_turn_ids = prev_turn_prompt_ids + prev_turn_completion_ids

            messages = concat_messages([prev_turn_prompt, prev_turn_completion])
            env_response = await self.env_response(messages, state)
            messages_and_env_response = concat_messages([messages, env_response])

            def compute_suffix_ids(lst: list[int], value: int) -> list[int]:
                """Returns all tokens after the last occurrence of `value` in `lst`, if any."""

                def find_last_index(lst: list[int], value: int) -> int:
                    for i in range(len(lst) - 1, -1, -1):
                        if lst[i] == value:
                            return i
                    raise ValueError

                try:
                    i = find_last_index(lst, value)
                    suffix_ids = lst[i + 1 :]
                    return suffix_ids
                except ValueError:
                    # end of message token not found, so we don't need to add any suffix tokens
                    return []

            def find_largest_overlap(a: list[int], b: list[int]) -> int:
                """Find the largest overlapping sequence between the end of a and beginning of b."""
                if not a or not b:
                    return 0

                max_possible = min(len(a), len(b))
                for overlap_len in range(1, max_possible + 1):
                    a_suffix = a[-overlap_len:]
                    b_prefix = b[:overlap_len]

                    if a_suffix != b_prefix:
                        return overlap_len - 1

                return 0

            if not state["exact_tokenization"]:  # default
                # we build the env_response_ids using simple tokenization
                env_response_ids = await tokenize(
                    client=client,
                    messages=env_response,
                    tools=state["oai_tools"],
                    model=state["model"],
                )

                # we add suffix_ids to prev_turn_ids. suffix_ids are tokens that
                # are added by the chat template after messages, but not
                # generated by the model, i.e. they will be part of messages_ids
                # (from the chat template) but not of prev_turn_ids (from the
                # engine). to not train OOD w.r.t. the chat template, we add
                # these suffix tokens to prev_turn_ids. we compute the
                # suffix_ids once, and cache them for future use in 'non-exact'
                # tokenization mode. then, for each turn, we find the largest
                # overlap between the end of prev_turn_ids and the beginning of
                # the suffix_ids. this is to correctly handle truncated turns
                # that did not produce message delimiting tokens.
                if getattr(self, "cached_suffix_ids", None) is None:
                    dummy_content = "World!"
                    dummy_messages: vf.Messages = [
                        {"role": "user", "content": "Hello"},
                        {"role": "assistant", "content": dummy_content},
                    ]
                    dummy_content_ids = await tokenize(
                        client=client,
                        messages=dummy_content,
                        tools=state["oai_tools"],
                        model=state["model"],
                    )
                    dummy_messages_ids = await tokenize(
                        client=client,
                        messages=dummy_messages,
                        tools=state["oai_tools"],
                        model=state["model"],
                        extra_kwargs=dict(add_generation_prompt=False),
                    )
                    # these are typically chat template specific tokens, such as
                    # eom tokens, newlines, etc.
                    suffix_ids = compute_suffix_ids(
                        dummy_messages_ids, dummy_content_ids[-1]
                    )
                    setattr(self, "cached_suffix_ids", suffix_ids)
                else:
                    suffix_ids = getattr(self, "cached_suffix_ids")
                overlap_len = find_largest_overlap(prev_turn_ids, suffix_ids)
                prev_turn_ids += suffix_ids[-overlap_len:]
            else:
                # in 'exact' tokenization mode, we build the env_response_ids by
                # tokenizing a (a) all previous messages + env response (b) all
                # previous messages and determine the env_response_ids as the
                # difference between those two. the main reason to do thi is to
                # avoid the edge case where the chat template adds prefix tokens
                # to the env response if and only if the message is embedded in
                # the context of the previous messages.
                #
                # NOTE: this only works if the chat template is incremental,
                # i.e. if (b) is a prefix of (a). for non-incremental chat
                # templates, the assert will trigger. in this case, building
                # token prompts from past messages is not the right thing to do
                # anyways, and one should use branching rollouts.
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

                # in 'exact' tokenization mode, we compute suffix_ids on-the-fly
                # for every turn. this already correctly handles truncated turns
                maybe_eom_token = prev_turn_ids[-1]
                suffix_ids = compute_suffix_ids(messages_ids, maybe_eom_token)
                prev_turn_ids += suffix_ids

            prompt_messages = messages_and_env_response
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
        use_token_prompts: bool = False,
        tokenize_method: Literal["local", "vllm"] | None = None,
        exact_tokenization: bool | None = None,
    ) -> State:
        """
        Generate a multi-turn rollout with the environment.
        """
        state = await self.init_state(
            input,
            client,
            model,
            sampling_args,
            use_token_prompts,
            tokenize_method,
            exact_tokenization,
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
