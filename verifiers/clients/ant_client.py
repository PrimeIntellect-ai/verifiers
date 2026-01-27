import json
import time

from anthropic import AsyncAnthropic

# Anthropic native types
from anthropic.types import (
    ContentBlock,
    MessageParam,
    TextBlockParam,
    ToolParam,
    ToolResultBlockParam,
    ToolUseBlockParam,
)
from anthropic.types.completion import Completion
from anthropic.types.message import Message

from verifiers.clients.client import (
    Client,
)
from verifiers.types import (
    AssistantMessage,
    ChatMessage,
    ChatMessages,
    ChatResponse,
    ClientConfig,
    FinishReason,
    ResponseMessage,
    SamplingArgs,
    SystemMessage,
    TextMessages,
    TextResponse,
    Tool,
    ToolCall,
    ToolMessage,
    UserMessage,
)
from verifiers.utils.client_utils import setup_anthropic_client


class AntClient(
    Client[AsyncAnthropic, Completion, Message, str, list[MessageParam], ToolParam]
):
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
        """Converts vf.ChatMessages to Anthropic chat messages."""

        def _parse_tool_args(tc_args: str | dict | object) -> dict:
            """Parse tool arguments from string or dict."""
            if isinstance(tc_args, str):
                try:
                    return json.loads(tc_args)
                except json.JSONDecodeError:
                    return {}
            elif isinstance(tc_args, dict):
                return tc_args
            return {}

        def from_legacy_chat_message(message: dict) -> MessageParam | None:
            """Convert dict-based message to Anthropic format. Returns None for system."""
            if message["role"] == "system":
                return None  # handled separately

            elif message["role"] == "user":
                return MessageParam(role="user", content=message["content"])

            elif message["role"] == "assistant":
                tool_calls = message.get("tool_calls")
                if tool_calls:
                    content_blocks: list[TextBlockParam | ToolUseBlockParam] = []
                    if message.get("content"):
                        content_blocks.append(
                            TextBlockParam(type="text", text=message["content"])
                        )
                    for tool_call in tool_calls:
                        content_blocks.append(
                            ToolUseBlockParam(
                                type="tool_use",
                                id=tool_call["id"],
                                name=tool_call["name"],
                                input=json.loads(tool_call["arguments"]),  # TODO: safe
                            )
                        )
                    return MessageParam(role="assistant", content=content_blocks)
                return MessageParam(role="assistant", content=message["content"])
            elif message["role"] == "tool":
                return MessageParam(
                    role="user",
                    content=[
                        ToolResultBlockParam(
                            type="tool_result",
                            tool_use_id=message["tool_call_id"],
                            content=message["content"],
                        )
                    ],
                )

            else:
                raise ValueError(f"Invalid chat message: {message}")

        def from_chat_message(message: ChatMessage) -> MessageParam | None:
            """Convert Pydantic message to Anthropic format. Returns None for system."""
            if isinstance(message, SystemMessage):
                return None
            elif isinstance(message, UserMessage):
                return MessageParam(role="user", content=message.content)
            elif isinstance(message, AssistantMessage):
                if message.tool_calls:
                    content_blocks: list[TextBlockParam | ToolUseBlockParam] = []
                    if message.content:
                        content_blocks.append(
                            TextBlockParam(type="text", text=message.content)
                        )
                    for tc in message.tool_calls:
                        content_blocks.append(
                            ToolUseBlockParam(
                                type="tool_use",
                                id=tc.id,
                                name=tc.name,
                                input=_parse_tool_args(tc.arguments),
                            )
                        )
                    return MessageParam(role="assistant", content=content_blocks)
                return MessageParam(role="assistant", content=message.content or "")
            elif isinstance(message, ToolMessage):
                return MessageParam(
                    role="user",
                    content=[
                        ToolResultBlockParam(
                            type="tool_result",
                            tool_use_id=message.tool_call_id,
                            content=message.content,
                        )
                    ],
                )
            else:
                raise ValueError(f"Invalid chat message: {message}")

        def extract_system_content(messages: ChatMessages) -> str:
            """Extract and concatenate system message contents."""
            system_contents = []
            for msg in messages:
                if isinstance(msg, SystemMessage):
                    system_contents.append(msg.content)
                elif isinstance(msg, dict) and msg.get("role") == "system":
                    system_contents.append(msg["content"])
            return "\n\n".join(system_contents)

        system = extract_system_content(messages)

        try:
            prompt = [
                converted
                for msg in messages
                if (converted := from_chat_message(msg)) is not None
            ]
        except Exception:
            self.logger.warning(
                "Found invalid chat message type, falling back to legacy dict parsing"
            )
            prompt = [
                converted
                for msg in messages
                if (converted := from_legacy_chat_message(dict(msg))) is not None
            ]

        return prompt, {"system": system}

    def to_native_tool(self, tool: Tool) -> ToolParam:
        return ToolParam(
            name=tool.name,
            description=tool.description,
            input_schema=tool.parameters,
        )

    async def get_native_chat_response(
        self,
        prompt: list[MessageParam],
        model: str,
        sampling_args: SamplingArgs,
        tools: list[ToolParam] | None,
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

        if tools:
            return await self.client.messages.create(
                model=model,
                messages=prompt,
                tools=tools,
                **normalize_sampling_args(sampling_args),
                **kwargs,
            )
        else:
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
