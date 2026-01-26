import functools
import os

from openai import AsyncOpenAI, BadRequestError
from openai.types import Completion
from openai.types.chat import ChatCompletion

from verifiers.clients.client import (
    Client,
    NormalizedMessageResponse,
    NormalizedTextResponse,
)
from verifiers.errors import (
    EmptyModelResponseError,
    InvalidModelResponseError,
    ModelError,
    OverlongPromptError,
)
from verifiers.types import ClientConfig, SamplingArgs
from verifiers.utils.client_utils import setup_http_client


def handle_overlong_prompt(func):
    """Decorator to handle overlong prompt errors from the model API."""

    @functools.wraps(func)
    async def wrapper(*args, **kwargs):
        try:
            return await func(*args, **kwargs)
        except Exception as e:
            # in case of making a request with an overlong prompt, e.g
            # we raise a special overlong prompt error
            if isinstance(e, BadRequestError):
                error_text = e.response.text.lower()
                context_length_phrases = [
                    "this model's maximum context length is",
                    "is longer than the model's context length",
                    "exceeds the model's context length",
                    "exceed the configured limit",
                    "exceeds the configured limit",
                    "exceeded model",
                ]
                if any(phrase in error_text for phrase in context_length_phrases):
                    raise OverlongPromptError from e
            # in all other case we raise a generic model error
            raise ModelError from e

    return wrapper


class OAIClient(Client[AsyncOpenAI, Completion, ChatCompletion]):
    """Wrapper for AsyncOpenAI client."""

    def setup_client(self, config: ClientConfig) -> AsyncOpenAI:
        return AsyncOpenAI(
            api_key=os.getenv(config.api_key_var) or "EMPTY",
            base_url=config.api_base_url,
            max_retries=config.max_retries,
            http_client=setup_http_client(config),
        )

    @handle_overlong_prompt
    async def get_text_response(
        self, prompt: str, model: str, sampling_args: SamplingArgs
    ) -> Completion:
        def normalize_sampling_args(sampling_args: SamplingArgs) -> SamplingArgs:
            return {k: v for k, v in sampling_args.items() if v is not None}

        response = await self.client.completions.create(
            model=model,
            prompt=prompt,
            **normalize_sampling_args(sampling_args),
        )
        return response

    async def raise_from_text_response(self, response: Completion) -> None:
        if response is None:
            raise EmptyModelResponseError("Model returned no response")
        if response.choices is None:
            raise EmptyModelResponseError("Model returned no response choices")
        if not len(response.choices) == 1:
            raise InvalidModelResponseError(
                f"Model returned {len(response.choices)} choices, expected 1"
            )
        if not response.choices[0].text:
            raise EmptyModelResponseError("Model returned no content")

    async def normalize_text_response(
        self, response: Completion
    ) -> NormalizedTextResponse:
        return response

    @handle_overlong_prompt
    async def get_message_response(
        self,
        prompt: list,
        model: str,
        sampling_args: SamplingArgs,
        tools: list | None = None,
    ) -> ChatCompletion:
        def normalize_sampling_args(sampling_args: SamplingArgs) -> SamplingArgs:
            if "max_tokens" in sampling_args:
                sampling_args["max_completion_tokens"] = sampling_args.pop("max_tokens")
            return {k: v for k, v in sampling_args.items() if v is not None}

        if tools:
            response = await self.client.chat.completions.create(
                model=model,
                messages=prompt,
                tools=tools,
                **normalize_sampling_args(sampling_args),
            )
        else:
            response = await self.client.chat.completions.create(
                model=model,
                messages=prompt,
                **normalize_sampling_args(sampling_args),
            )
        return response

    async def raise_from_message_response(self, response: ChatCompletion) -> None:
        if response is None:
            raise EmptyModelResponseError("Model returned no response")
        if response.choices is None:
            raise EmptyModelResponseError("Model returned no response choices")
        if not len(response.choices) == 1:
            raise InvalidModelResponseError(
                f"Model returned {len(response.choices)} choices, expected 1"
            )
        if not (
            response.choices[0].message.content
            or response.choices[0].message.tool_calls
        ):
            raise EmptyModelResponseError(
                "Model returned no content and did not call any tools"
            )

    async def normalize_message_response(
        self, response: ChatCompletion
    ) -> NormalizedMessageResponse:
        return response

    @handle_overlong_prompt
    async def get_message_with_tokens(
        self,
        prompt: list,
        prompt_ids: list[int],
        model: str,
        sampling_args: SamplingArgs,
        tools: list | None = None,
    ) -> NormalizedMessageResponse:
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
