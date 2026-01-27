import json
import time

from anthropic import AsyncAnthropic
from anthropic.types import ContentBlock, MessageParam
from anthropic.types.completion import Completion
from anthropic.types.message import Message

from verifiers.clients.client import (
    Client,
)
from verifiers.types import (
    ChatMessages,
    ChatResponse,
    ClientConfig,
    FinishReason,
    ResponseMessage,
    SamplingArgs,
    TextMessages,
    TextResponse,
    Tool,
    ToolCall,
)
from verifiers.utils.client_utils import setup_anthropic_client


class AntClient(Client[AsyncAnthropic, Completion, Message, str, list[MessageParam]]):
    """Wrapper for AsyncAnthropic client."""

    @staticmethod
    def setup_client(config: ClientConfig) -> AsyncAnthropic:
        return setup_anthropic_client(config)

    def to_native_text_prompt(self, messages: TextMessages) -> tuple[str, dict]:
        return messages, {}

    async def get_native_text_response(
        self, prompt: str, model: str, sampling_args: SamplingArgs, **kwargs
    ) -> Completion:
        def normalize_sampling_args(sampling_args: SamplingArgs):
            return {k: v for k, v in sampling_args.items() if v is not None}

        return await self.client.completions.create(
            model=model,
            prompt=prompt,
            **normalize_sampling_args(sampling_args),
            **kwargs,
        )

    async def raise_from_native_text_response(self, response: Completion) -> None:
        pass

    def from_native_text_response(self, response: Completion) -> TextResponse:
        return TextResponse(
            id=response.id,
            model=response.model,
            created=int(time.time()),
            usage=None,
            message=ResponseMessage(
                content=response.completion,
                finish_reason=None,
                is_truncated=None,
                tokens=None,
                reasoning_content=None,
                tool_calls=None,
            ),
        )

    def to_native_chat_prompt(
        self, messages: ChatMessages
    ) -> tuple[list[MessageParam], dict]:
        def extract_system_messages(messages: ChatMessages) -> tuple[list, list]:
            system_messages: list = []
            non_system_messages: list = []
            for msg in messages:
                if msg.get("role") == "system":
                    system_messages.append(msg)
                else:
                    non_system_messages.append(msg)

            return system_messages, non_system_messages

        system_messages, non_system_messages = extract_system_messages(messages)
        system = "\n\n".join([msg["content"] for msg in system_messages])
        prompt = []
        for msg in non_system_messages:
            prompt.append(MessageParam(role=msg["role"], content=msg["content"]))

        return prompt, {"system": system}

    async def get_native_chat_response(
        self,
        prompt: list[MessageParam],
        model: str,
        sampling_args: SamplingArgs,
        tools: list[Tool] | None,
        **kwargs,
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

        return await self.client.messages.create(
            model=model,
            messages=prompt,
            **normalize_sampling_args(sampling_args),
            **kwargs,
        )

    async def raise_from_native_chat_response(self, response: Message) -> None:
        pass

    def from_native_chat_response(self, response: Message) -> ChatResponse:
        def parse_content(
            content_blocks: list[ContentBlock],
        ) -> tuple[str, str, list[ToolCall]]:
            content = ""
            reasoning_content = ""
            tool_calls = []
            for content_block in response.content:
                if content_block.type == "text":
                    content += content_block.text
                elif content_block.type == "thinking":
                    reasoning_content += content_block.thinking
                elif content_block.type == "tool_use":
                    tool_calls.append(
                        ToolCall(
                            id=content_block.id,
                            name=content_block.name,
                            arguments=json.dumps(content_block.input),
                        )
                    )
                else:
                    raise ValueError(f"Unsupported content type: {content_block.type}")
            return content, reasoning_content, tool_calls

        def parse_finish_reason(response: Message) -> FinishReason:
            match response.stop_reason:
                case "end_turn":
                    return "stop"
                case "max_tokens":
                    return "length"
                case "tool_use":
                    return "tool_calls"
                case _:
                    return None

        content, reasoning_content, tool_calls = parse_content(response.content)

        return ChatResponse(
            id=response.id,
            model=response.model,
            created=int(time.time()),
            usage=None,
            message=ResponseMessage(
                content=content or None,
                reasoning_content=reasoning_content or None,
                tool_calls=tool_calls or None,
                finish_reason=parse_finish_reason(response),
                is_truncated=None,
                tokens=None,
            ),
        )

    async def get_chat_response_with_tokens(
        self,
        prompt: ChatMessages,
        prompt_ids: list[int],
        model: str,
        sampling_args: SamplingArgs,
        tools: list[Tool] | None,
    ) -> ChatResponse:
        raise NotImplementedError("TITO is not yet implemented for Anthropic client.")
