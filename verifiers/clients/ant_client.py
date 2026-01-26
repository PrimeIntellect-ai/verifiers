import json
import os
import time

from anthropic import AsyncAnthropic
from anthropic.types.completion import Completion
from anthropic.types.message import Message
from openai.types import Completion as OAICompletion
from openai.types.chat import ChatCompletion as OAIChatCompletion
from openai.types.chat.chat_completion import Choice as OAIChatCompletionChoice
from openai.types.chat.chat_completion_message import (
    ChatCompletionMessage as OAIChatCompletionMessage,
)
from openai.types.chat.chat_completion_message_function_tool_call import (
    ChatCompletionMessageFunctionToolCall,
    Function,
)
from openai.types.completion_choice import CompletionChoice as OAICompletionChoice

from verifiers.clients.client import (
    Client,
    NormalizedMessageResponse,
    NormalizedTextResponse,
)
from verifiers.types import ClientConfig, SamplingArgs
from verifiers.utils.client_utils import setup_http_client


# Extract system messages and pass as top-level parameter
# Anthropic API doesn't accept "system" role in messages list
def extract_system_messages(messages: list) -> tuple[list, list]:
    system_messages: list = []
    non_system_messages: list = []
    for msg in messages:
        if msg.get("role") == "system":
            system_messages.append(msg)
        else:
            non_system_messages.append(msg)

    return system_messages, non_system_messages


class AntClient(Client[AsyncAnthropic, Completion, Message]):
    """Wrapper for AsyncAnthropic client."""

    def setup_client(self, config: ClientConfig) -> AsyncAnthropic:
        return AsyncAnthropic(
            api_key=os.getenv(config.api_key_var) or "EMPTY",
            base_url=config.api_base_url,
            max_retries=config.max_retries,
            http_client=setup_http_client(config),
        )

    async def get_text_response(
        self, prompt: str, model: str, sampling_args: SamplingArgs
    ) -> Completion:
        def normalize_sampling_args(sampling_args: SamplingArgs) -> SamplingArgs:
            return {k: v for k, v in sampling_args.items() if v is not None}

        return await self.client.completions.create(
            model=model,
            prompt=prompt,
            **normalize_sampling_args(sampling_args),
        )

    async def raise_from_text_response(self, response: Completion) -> None:
        pass

    async def normalize_text_response(
        self, response: Completion
    ) -> NormalizedTextResponse:
        return OAICompletion(
            id=response.id,
            choices=[
                OAICompletionChoice(
                    text=response.completion,
                    index=0,
                    finish_reason=response.stop_reason,  # type: ignore
                )
            ],
            created=int(time.time()),
            model=response.model,
            object="text_completion",
        )

    async def get_message_response(
        self,
        prompt: list,
        model: str,
        sampling_args: SamplingArgs,
        tools: list | None = None,
    ) -> Message:
        def normalize_sampling_args(sampling_args: SamplingArgs) -> dict:
            max_tokens = sampling_args.pop("max_tokens")
            sampling_args.pop("n", None)
            if max_tokens is None:
                self.logger.warning(
                    "max_tokens is not set but Anthropic /v1/messages endpoint requires it, falling back to max_tokens=4096"
                )
                max_tokens = 4096
            sampling_args["max_tokens"] = max_tokens

            return {k: v for k, v in sampling_args.items() if v is not None}

        system_messages, non_system_messages = extract_system_messages(prompt)
        system_part = "\n\n".join([msg["content"] for msg in system_messages])
        kwargs = {}
        if system_part:
            kwargs["system"] = system_part

        return await self.client.messages.create(
            model=model,
            messages=non_system_messages,
            **normalize_sampling_args(sampling_args),
            **kwargs,
        )

    async def raise_from_message_response(self, response: Message) -> None:
        pass

    async def normalize_message_response(
        self, response: Message
    ) -> NormalizedMessageResponse:
        content = ""
        tool_calls = []
        for content_block in response.content:
            if content_block.type == "text":
                content += content_block.text
            elif content_block.type == "thinking":
                content += f"<think>{content_block.thinking}</think>"
            elif content_block.type == "tool_use":
                tool_calls.append(
                    ChatCompletionMessageFunctionToolCall(
                        id=content_block.id,
                        type="function",
                        function=Function(
                            name=content_block.name,
                            arguments=json.dumps(content_block.input),
                        ),
                    )
                )
            else:
                raise ValueError(f"Unsupported content type: {content_block.type}")
        return OAIChatCompletion(
            id=response.id,
            choices=[
                OAIChatCompletionChoice(
                    message=OAIChatCompletionMessage(
                        role="assistant",
                        content=content or None,
                        tool_calls=tool_calls or None,
                    ),
                    finish_reason="stop",  # response.stop_reason,
                    index=0,
                )
            ],
            created=int(time.time()),
            model=response.model,
            object="chat.completion",
        )

    async def get_message_with_tokens(
        self,
        prompt: list,
        prompt_ids: list[int],
        model: str,
        sampling_args: SamplingArgs,
        tools: list | None = None,
    ) -> NormalizedMessageResponse:
        raise NotImplementedError("TITO is not yet implemented for Anthropic client.")
