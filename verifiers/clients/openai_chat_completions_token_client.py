from typing import Optional, cast

from openai import AsyncOpenAI, BaseModel
from openai.types.chat import ChatCompletion

from verifiers.clients.openai_chat_completions_client import (
    OpenAIChatCompletionsClient,
    OpenAIChatMessages,
    OpenAIChatResponse,
    OpenAITool,
    handle_openai_overlong_prompt,
)
from verifiers.types import SamplingArgs, State
from verifiers.utils.message_utils import concat_messages


# copy from vllm/entrypoints/openai/protocol.py
class TokenizeResponse(BaseModel):
    count: int
    max_model_len: int
    tokens: list[int]
    token_strs: Optional[list[str]] = None


class OpenAIChatCompletionsTokenClient(OpenAIChatCompletionsClient):
    """Wrapper for custom vLLM route /v1/chat/completions/tokens via AsyncOpenAI client."""

    @property
    def token_client(self) -> AsyncOpenAI:
        """Strips trailing /v1 from the OpenAI client."""
        base_url = str(self.client.base_url).rstrip("/")
        if base_url.endswith("/v1"):
            base_url = base_url[:-3]
        return self.client.with_options(base_url=base_url)

    @handle_openai_overlong_prompt
    async def get_native_response(
        self,
        prompt: OpenAIChatMessages,
        model: str,
        sampling_args: SamplingArgs,
        tools: list[OpenAITool] | None = None,
        **kwargs,
    ) -> OpenAIChatResponse:
        def normalize_sampling_args(sampling_args: SamplingArgs):
            sampling_args = dict(sampling_args)
            if "max_tokens" in sampling_args:
                sampling_args["max_completion_tokens"] = sampling_args.pop("max_tokens")
            sampling_args["logprobs"] = True
            extra_body = dict(return_token_ids=True)
            if "extra_body" in sampling_args:
                sampling_args["extra_body"] = {
                    **sampling_args["extra_body"],
                    **extra_body,
                }
            else:
                sampling_args["extra_body"] = extra_body
            return {k: v for k, v in sampling_args.items() if v is not None}

        sampling_args = normalize_sampling_args(sampling_args)
        state = cast(State, kwargs.pop("state"))
        # use /v1/chat/completions for first turn to avoid redundant tokenization
        if len(state["trajectory"]) == 0:
            return await super().get_native_response(
                prompt, model, sampling_args, tools
            )

        # Token endpoint requires the prompt to extend the previous turn.
        # Multi-agent envs rebuild prompts fresh per actor, breaking this.
        prev_turn = state["trajectory"][-1]
        prev_messages = concat_messages([prev_turn["prompt"], prev_turn["completion"]])
        if len(prompt) <= len(prev_messages):
            return await super().get_native_response(
                prompt, model, sampling_args, tools
            )

        prompt_ids = await self.get_prompt_ids(state, prompt, tools)
        extra_body = sampling_args.pop("extra_body", {})
        body = dict(
            model=model,
            messages=prompt,
            tools=tools,
            tokens=prompt_ids,
            **sampling_args,
            **extra_body,
        )

        return await self.client.post(
            "/chat/completions/tokens",
            body=body,
            cast_to=ChatCompletion,
        )

    async def get_prompt_ids(
        self,
        state: State,
        prompt_messages: OpenAIChatMessages,
        oai_tools: list[OpenAITool] | None,
    ) -> list[int]:
        """
        Build prompt_ids (token prompt) corresponding to prompt_messages. We assume
        that this method is called *before* making the model response from
        prompt_messages, i.e. the previous turn's prompt and completion do not yet
        include the environment response and next turn's model response.
        """
        prev_turn_prompt = state["trajectory"][-1]["prompt"]
        prev_turn_completion = state["trajectory"][-1]["completion"]
        prev_turn_tokens = state["trajectory"][-1]["tokens"]
        assert prev_turn_tokens is not None
        prev_turn_prompt_ids = prev_turn_tokens["prompt_ids"]
        prev_turn_completion_ids = prev_turn_tokens["completion_ids"]
        prev_turn_ids = prev_turn_prompt_ids + prev_turn_completion_ids

        # the env response is all messages after the previous turn
        messages = concat_messages([prev_turn_prompt, prev_turn_completion])
        env_response = prompt_messages[len(messages) :]

        def compute_suffix_ids(lst: list[int], value: int) -> list[int]:
            """Returns all tokens after the last occurrence of `value` in `lst`, if any."""

            def find_last_index(lst: list[int], target: int) -> int:
                for i in range(len(lst) - 1, -1, -1):
                    if lst[i] == target:
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
            for overlap_len in reversed(range(1, max_possible + 1)):
                a_suffix = a[-overlap_len:]
                b_prefix = b[:overlap_len]

                if a_suffix == b_prefix:
                    return overlap_len

            return 0

        # we build the env_response_ids using simple tokenization
        env_response_ids = await self.tokenize(
            messages=env_response,
            tools=None,
            model=state["model"],
        )

        # we add suffix_ids to prev_turn_ids. suffix_ids are tokens that are added
        # by the chat template after messages, but not generated by the model, i.e.
        # they will be part of messages_ids (from the chat template) but not of
        # prev_turn_ids (from the engine). to not train OOD w.r.t. the chat
        # template, we add these suffix tokens to prev_turn_ids. we compute the
        # suffix_ids once, and cache them for future use. then, for each turn, we
        # find the largest overlap between the end of prev_turn_ids and the
        # beginning of the suffix_ids. this is to correctly handle truncated turns
        # that did not produce message delimiting tokens.
        if state.get("_cached_suffix_ids") is None:
            dummy_content = "World!"
            dummy_messages = cast(
                OpenAIChatMessages,
                [
                    {"role": "user", "content": "Hello"},
                    {"role": "assistant", "content": dummy_content},
                ],
            )
            dummy_content_ids = await self.tokenize(
                messages=dummy_content,
                tools=oai_tools,
                model=state["model"],
            )
            dummy_messages_ids = await self.tokenize(
                messages=dummy_messages,
                tools=oai_tools,
                model=state["model"],
                extra_kwargs=dict(add_generation_prompt=False),
            )
            # these are typically chat template specific tokens, such as
            # eom tokens, newlines, etc.
            suffix_ids = compute_suffix_ids(dummy_messages_ids, dummy_content_ids[-1])
            state["_cached_suffix_ids"] = suffix_ids
        else:
            suffix_ids = state["_cached_suffix_ids"]
        overlap_len = find_largest_overlap(prev_turn_ids, suffix_ids)
        prev_turn_ids += suffix_ids[overlap_len:]

        prompt_ids = prev_turn_ids + env_response_ids

        return prompt_ids

    async def tokenize(
        self,
        messages: str | OpenAIChatMessages,
        tools: list[OpenAITool] | None,
        model: str,
        extra_kwargs: dict = {},
        **kwargs,
    ) -> list[int]:
        """Tokenize messages using the vLLM /tokenize API."""
        if isinstance(messages, str):
            body = dict(
                model=model,
                prompt=messages,
                **extra_kwargs,
            )
            tokenize_response = await self.token_client.post(
                "/tokenize", body=body, cast_to=TokenizeResponse
            )
        else:
            body = dict(
                model=model,
                messages=messages,
                tools=tools,
                **extra_kwargs,
            )
            tokenize_response = await self.token_client.post(
                "/tokenize", body=body, cast_to=TokenizeResponse
            )
        return tokenize_response.tokens
