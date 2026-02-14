from typing import cast

from openai.types.chat import ChatCompletion

from verifiers.clients.openai_chat_completions_client import (
    OpenAIChatCompletionsClient,
    OpenAIChatMessages,
    OpenAIChatResponse,
    OpenAITool,
)
from verifiers.clients.openai_chat_completions_client import (
    handle_openai_overlong_prompt,
)
from verifiers.types import SamplingArgs, State
from verifiers.utils.token_utils import get_prompt_ids


class OpenAIChatCompletionsTokenClient(OpenAIChatCompletionsClient):
    """Wrapper for custom vLLM route /v1/chat/completions/tokens via AsyncOpenAI client."""

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
        prompt_ids = await get_prompt_ids(state, prompt, tools, self.client)
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
