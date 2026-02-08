import json
import time
from collections.abc import Mapping
from typing import Any, cast

from anthropic import AsyncAnthropic
from anthropic.types import (
    ContentBlock,
    TextBlockParam,
    ToolResultBlockParam,
    ToolUseBlockParam,
)
from anthropic.types import (
    Message as AnthropicMessage,
)
from anthropic.types import (
    MessageParam as AnthropicMessageParam,
)
from anthropic.types import (
    ToolParam as AnthropicToolParam,
)

from verifiers.clients.client import Client
from verifiers.types import (
    AssistantMessage,
    ClientConfig,
    FinishReason,
    Message,
    Messages,
    Response,
    ResponseMessage,
    SamplingArgs,
    SystemMessage,
    Tool,
    ToolCall,
    ToolMessage,
    UserMessage,
)
from verifiers.utils.client_utils import setup_anthropic_client


class AnthropicMessagesClient(
    Client[
        AsyncAnthropic,
        list[AnthropicMessageParam],
        AnthropicMessage,
        AnthropicToolParam,
    ]
):
    """Wrapper for Messages API via AsyncAnthropic client."""

    def setup_client(self, config: ClientConfig) -> AsyncAnthropic:
        return setup_anthropic_client(config)

    async def to_native_prompt(
        self, messages: Messages
    ) -> tuple[list[AnthropicMessageParam], dict]:
        def parse_legacy_tool_call(tool_call: Any) -> tuple[str, str, dict[str, Any]]:
            if isinstance(tool_call, Mapping):
                tool_call_id = tool_call.get("id")
                function_obj = tool_call.get("function")
                if isinstance(function_obj, Mapping):
                    name = function_obj.get("name")
                    raw_input = function_obj.get("arguments")
                else:
                    name = tool_call.get("name")
                    raw_input = tool_call.get("arguments")
            else:
                tool_call_id = getattr(tool_call, "id", None)
                function_obj = getattr(tool_call, "function", None)
                if function_obj is not None:
                    name = getattr(function_obj, "name", None)
                    raw_input = getattr(function_obj, "arguments", None)
                else:
                    name = getattr(tool_call, "name", None)
                    raw_input = getattr(tool_call, "arguments", None)

            if not isinstance(tool_call_id, str):
                tool_call_id = ""
            if not isinstance(name, str):
                name = ""
            return tool_call_id, name, _parse_tool_args(raw_input)

        def parse_data_url(url: str) -> tuple[str, str] | None:
            if not url.startswith("data:"):
                return None
            if "," not in url:
                return None
            header, data = url.split(",", 1)
            if ";base64" not in header:
                return None
            media_type = header[5:].split(";")[0] or "image/png"
            return media_type, data

        def normalize_content_block(block: Any) -> dict[str, Any]:
            if isinstance(block, Mapping):
                return dict(block)
            if hasattr(block, "model_dump"):
                return block.model_dump()
            raise ValueError(f"Invalid content block type: {type(block)}")

        def normalize_anthropic_content(content: Any) -> Any:
            if isinstance(content, str):
                return content
            if not isinstance(content, list):
                return str(content)

            blocks: list[dict[str, Any]] = []
            for raw_part in content:
                part = normalize_content_block(raw_part)
                part_type = part.get("type")
                if part_type == "text":
                    text = part.get("text")
                    if isinstance(text, str):
                        blocks.append({"type": "text", "text": text})
                elif part_type == "image_url":
                    image_url = part.get("image_url", {})
                    url = (
                        image_url.get("url") if isinstance(image_url, Mapping) else None
                    )
                    if isinstance(url, str):
                        parsed = parse_data_url(url)
                        if parsed is not None:
                            media_type, data = parsed
                            blocks.append(
                                {
                                    "type": "image",
                                    "source": {
                                        "type": "base64",
                                        "media_type": media_type,
                                        "data": data,
                                    },
                                }
                            )
                        else:
                            blocks.append({"type": "text", "text": "[image]"})
                elif part_type == "input_audio":
                    blocks.append({"type": "text", "text": "[audio]"})
                else:
                    blocks.append({"type": "text", "text": str(part)})
            return blocks

        def content_to_text_chunks(content: Any) -> list[str]:
            normalized = normalize_anthropic_content(content)
            if isinstance(normalized, str):
                return [normalized] if normalized else []
            chunks: list[str] = []
            for block in normalized:
                block_type = block.get("type")
                if block_type == "text":
                    text = block.get("text")
                    if isinstance(text, str) and text:
                        chunks.append(text)
                elif block_type == "image":
                    chunks.append("[image]")
            return chunks

        def _parse_tool_args(tc_args: str | dict | object | None) -> dict[str, Any]:
            """Parse tool arguments from string or dict."""
            if isinstance(tc_args, str):
                try:
                    parsed = json.loads(tc_args)
                    return parsed if isinstance(parsed, dict) else {}
                except json.JSONDecodeError:
                    return {}
            elif isinstance(tc_args, dict):
                return cast(dict[str, Any], tc_args)
            return {}

        def from_legacy_chat_message(message: dict) -> AnthropicMessageParam | None:
            role = message.get("role")
            if role == "system":
                return None  # handled separately

            elif role == "user":
                return AnthropicMessageParam(
                    role="user",
                    content=cast(
                        Any, normalize_anthropic_content(message.get("content", ""))
                    ),
                )

            elif role == "assistant":
                tool_calls = message.get("tool_calls")
                if tool_calls:
                    content_blocks: list[TextBlockParam | ToolUseBlockParam] = []
                    for text_chunk in content_to_text_chunks(message.get("content")):
                        content_blocks.append(
                            TextBlockParam(type="text", text=text_chunk)
                        )
                    for tool_call in cast(list[Any], tool_calls):
                        tool_call_id, tool_name, tool_input = parse_legacy_tool_call(
                            tool_call
                        )
                        content_blocks.append(
                            ToolUseBlockParam(
                                type="tool_use",
                                id=tool_call_id,
                                name=tool_name,
                                input=tool_input,
                            )
                        )
                    return AnthropicMessageParam(
                        role="assistant", content=content_blocks
                    )
                return AnthropicMessageParam(
                    role="assistant",
                    content=str(message.get("content", "")),
                )
            elif role == "tool":
                tool_call_id = message.get("tool_call_id")
                if not isinstance(tool_call_id, str):
                    tool_call_id = ""
                return AnthropicMessageParam(
                    role="user",
                    content=[
                        ToolResultBlockParam(
                            type="tool_result",
                            tool_use_id=tool_call_id,
                            content=message.get("content", ""),
                        )
                    ],
                )

            else:
                raise ValueError(f"Invalid chat message: {message}")

        def from_chat_message(message: Message) -> AnthropicMessageParam | None:
            assert not isinstance(message, str)
            if isinstance(message, SystemMessage):
                return None
            elif isinstance(message, UserMessage):
                return AnthropicMessageParam(
                    role="user",
                    content=cast(Any, normalize_anthropic_content(message.content)),
                )
            elif isinstance(message, AssistantMessage):
                if message.tool_calls:
                    content_blocks: list[TextBlockParam | ToolUseBlockParam] = []
                    for text_chunk in content_to_text_chunks(message.content):
                        content_blocks.append(
                            TextBlockParam(type="text", text=text_chunk)
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
                    return AnthropicMessageParam(
                        role="assistant", content=content_blocks
                    )
                return AnthropicMessageParam(
                    role="assistant",
                    content=cast(
                        Any,
                        message.content
                        if isinstance(message.content, str)
                        else " ".join(content_to_text_chunks(message.content)),
                    ),
                )
            elif isinstance(message, ToolMessage):
                return AnthropicMessageParam(
                    role="user",
                    content=[
                        ToolResultBlockParam(
                            type="tool_result",
                            tool_use_id=message.tool_call_id,
                            content=cast(
                                Any,
                                message.content
                                if isinstance(message.content, str)
                                else " ".join(content_to_text_chunks(message.content)),
                            ),
                        )
                    ],
                )
            else:
                raise ValueError(f"Invalid chat message: {message}")

        def extract_system_content(messages: Messages) -> str:
            """Extract and concatenate system message contents."""
            system_contents = []
            for msg in messages:
                if isinstance(msg, SystemMessage):
                    content = msg.content
                    system_contents.append(" ".join(content_to_text_chunks(content)))
                elif isinstance(msg, Mapping) and msg.get("role") == "system":
                    raw_content = msg["content"]
                    system_contents.append(
                        " ".join(content_to_text_chunks(raw_content))
                    )
            return "\n\n".join(system_contents)

        system = extract_system_content(messages)
        prompt = [
            (
                from_legacy_chat_message(cast(dict, msg))
                if isinstance(msg, Mapping)
                else from_chat_message(msg)
            )
            for msg in messages
        ]
        prompt = [converted for converted in prompt if converted is not None]

        return prompt, {"system": system}

    async def to_native_tool(self, tool: Tool) -> AnthropicToolParam:
        return AnthropicToolParam(
            name=tool.name,
            description=tool.description,
            input_schema=tool.parameters,
        )

    async def get_native_response(
        self,
        prompt: list[AnthropicMessageParam],
        model: str,
        sampling_args: SamplingArgs,
        tools: list[AnthropicToolParam] | None = None,
        **kwargs,
    ) -> AnthropicMessage:
        def normalize_sampling_args(sampling_args: SamplingArgs) -> dict:
            sampling_args = dict(sampling_args)
            max_tokens = sampling_args.pop("max_tokens", None)
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
            )
        else:
            return await self.client.messages.create(
                model=model,
                messages=prompt,
                **normalize_sampling_args(sampling_args),
            )

    async def raise_from_native_response(self, response: AnthropicMessage) -> None:
        pass

    async def from_native_response(self, response: AnthropicMessage) -> Response:
        def parse_content(
            content_blocks: list[ContentBlock],
        ) -> tuple[str, str, list[ToolCall]]:
            content = ""
            reasoning_content = ""
            tool_calls = []
            for content_block in content_blocks:
                if content_block.type == "text":
                    text_value = getattr(content_block, "text", None)
                    if isinstance(text_value, str):
                        content += text_value
                elif content_block.type == "thinking":
                    thinking_value = getattr(content_block, "thinking", None)
                    if isinstance(thinking_value, str):
                        reasoning_content += thinking_value
                elif content_block.type == "tool_use":
                    tool_id = getattr(content_block, "id", None)
                    tool_name = getattr(content_block, "name", None)
                    tool_input = getattr(content_block, "input", None)
                    if not isinstance(tool_id, str) or not isinstance(tool_name, str):
                        continue
                    tool_calls.append(
                        ToolCall(
                            id=tool_id,
                            name=tool_name,
                            arguments=json.dumps(tool_input),
                        )
                    )
                else:
                    raise ValueError(f"Unsupported content type: {content_block.type}")
            return content, reasoning_content, tool_calls

        def parse_finish_reason(response: AnthropicMessage) -> FinishReason:
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

        return Response(
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
