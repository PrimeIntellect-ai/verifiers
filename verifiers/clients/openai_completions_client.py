from openai import (
    AsyncOpenAI,
)
from openai.types import Completion

from verifiers.clients.client import Client
from verifiers.clients.openai_chat_completions_client import (
    content_to_text,
    get_usage_field,
    handle_openai_overlong_prompt,
)
from verifiers.errors import (
    EmptyModelResponseError,
    InvalidModelResponseError,
)
from verifiers.types import (
    ClientConfig,
    FinishReason,
    Messages,
    Response,
    ResponseMessage,
    ResponseTokens,
    SamplingArgs,
    Tool,
    Usage,
)
from verifiers.utils.client_utils import setup_openai_client

OpenAITextMessages = str
OpenAITextResponse = Completion


class OpenAICompletionsClient(
    Client[
        AsyncOpenAI,
        OpenAITextMessages,
        OpenAITextResponse,
        None,
    ]
):
    """Wrapper for Completions API via AsyncOpenAI client."""

    def setup_client(self, config: ClientConfig) -> AsyncOpenAI:
        return setup_openai_client(config)

    async def close(self) -> None:
        await self.client.close()

    async def to_native_prompt(
        self, messages: Messages
    ) -> tuple[OpenAITextMessages, dict]:
        parts: list[str] = []
        for message in messages:
            content = message.content
            if isinstance(content, list):
                for part in content:
                    if isinstance(part, dict):
                        if part.get("type") != "text":
                            raise ValueError(
                                "Completions API does not support non-text content (e.g. images). "
                                "Use chat_completions or messages client_type instead."
                            )
                    elif getattr(part, "type", None) != "text":
                        raise ValueError(
                            "Completions API does not support non-text content (e.g. images). "
                            "Use chat_completions or messages client_type instead."
                        )
            parts.append(content_to_text(content))
        return "\n\n".join(parts), {}

    async def to_native_tool(self, tool: Tool) -> None:
        raise ValueError("Tools are not supported for Completions API")

    @handle_openai_overlong_prompt
    async def get_native_response(
        self,
        prompt: OpenAITextMessages,
        model: str,
        sampling_args: SamplingArgs,
        tools: list[None] | None = None,
        **kwargs,
    ) -> OpenAITextResponse:
        if tools:
            raise ValueError(
                "Completions API does not support tools. "
                "Use chat_completions or messages client_type instead."
            )

        def normalize_sampling_args(sampling_args: SamplingArgs):
            return {k: v for k, v in sampling_args.items() if v is not None}

        response = await self.client.completions.create(
            model=model,
            prompt=prompt,
            **normalize_sampling_args(sampling_args),
        )
        return response

    async def raise_from_native_response(self, response: OpenAITextResponse) -> None:
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

    async def from_native_response(self, response: OpenAITextResponse) -> Response:
        def parse_usage(response: OpenAITextResponse) -> Usage | None:
            usage = getattr(response, "usage", None)
            if usage is None:
                return None
            prompt_tokens = get_usage_field(usage, "prompt_tokens")
            completion_tokens = get_usage_field(usage, "completion_tokens")
            if not isinstance(prompt_tokens, int) or not isinstance(
                completion_tokens, int
            ):
                prompt_tokens = get_usage_field(usage, "input_tokens")
                completion_tokens = get_usage_field(usage, "output_tokens")
            total_tokens = get_usage_field(usage, "total_tokens")
            if not isinstance(prompt_tokens, int) or not isinstance(
                completion_tokens, int
            ):
                return None
            if not isinstance(total_tokens, int):
                total_tokens = prompt_tokens + completion_tokens
            return Usage(
                prompt_tokens=prompt_tokens,
                reasoning_tokens=0,
                completion_tokens=completion_tokens,
                total_tokens=total_tokens,
            )

        def parse_finish_reason(response: OpenAITextResponse) -> FinishReason:
            match response.choices[0].finish_reason:
                case "stop":
                    return "stop"
                case "length":
                    return "length"
                case _:
                    return None

        def parse_is_truncated(response: OpenAITextResponse) -> bool:
            return response.choices[0].finish_reason == "length"

        def parse_tokens(response: OpenAITextResponse) -> ResponseTokens | None:
            if not hasattr(response.choices[0], "prompt_token_ids"):
                return None
            if not hasattr(response.choices[0], "token_ids"):
                return None
            if not hasattr(response.choices[0], "logprobs"):
                return None
            if response.choices[0].logprobs is None:
                return None
            if not hasattr(response.choices[0].logprobs, "token_logprobs"):
                return None
            prompt_ids = getattr(response.choices[0], "prompt_token_ids")
            prompt_mask = [0] * len(prompt_ids)
            completion_ids = getattr(response.choices[0], "token_ids")
            completion_mask = [1] * len(completion_ids)
            completion_logprobs = getattr(
                response.choices[0].logprobs, "token_logprobs"
            )
            return ResponseTokens(
                prompt_ids=prompt_ids,
                prompt_mask=prompt_mask,
                completion_ids=completion_ids,
                completion_mask=completion_mask,
                completion_logprobs=completion_logprobs,
            )

        return Response(
            id=response.id,
            created=response.created,
            model=response.model,
            usage=parse_usage(response),
            message=ResponseMessage(
                content=response.choices[0].text,
                finish_reason=parse_finish_reason(response),
                is_truncated=parse_is_truncated(response),
                tokens=parse_tokens(response),
                reasoning_content=None,
                tool_calls=None,
            ),
        )
