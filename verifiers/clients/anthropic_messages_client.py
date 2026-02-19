import functools
import json
import time
from collections.abc import Mapping
from typing import Any, cast

from anthropic import (
    AsyncAnthropic,
    AuthenticationError,
    BadRequestError,
    PermissionDeniedError,
)
from anthropic.types import (
    ContentBlock,
    RedactedThinkingBlock,
    TextBlockParam,
    ThinkingBlock,
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
from verifiers.errors import OverlongPromptError, RateLimitError as VFRateLimitError
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
    TextMessage,
    Tool,
    ToolCall,
    ToolMessage,
    Usage,
    UserMessage,
)
from verifiers.utils.client_utils import setup_anthropic_client


def _handle_anthropic_overlong_prompt(func):
    """Decorator to handle overlong prompt errors from the Anthropic API."""

    @functools.wraps(func)
    async def wrapper(*args, **kwargs):
        try:
            return await func(*args, **kwargs)
        except (AuthenticationError, PermissionDeniedError):
            raise
        except BadRequestError as e:
            # Check for HTTP 429 rate limit
            if hasattr(e, "response") and hasattr(e.response, "status_code") and e.response.status_code == 429:
                raise VFRateLimitError(f"Anthropic rate limit: {e}") from e
            error_text = e.message.lower()
            context_length_phrases = [
                "prompt is too long",
                "exceed context limit",
                "exceeds context limit",
                "too many total text bytes",
                "context length",
                "input is too long",
            ]
            if any(phrase in error_text for phrase in context_length_phrases):
                raise OverlongPromptError from e
            raise

    return wrapper


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

    async def close(self) -> None:
        await self.client.close()

    async def to_native_prompt(
        self, messages: Messages
    ) -> tuple[list[AnthropicMessageParam], dict]:
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

        def extract_thinking_blocks(message: AssistantMessage) -> list[dict[str, str]]:
            """Preserve provider-specific reasoning blocks across tool-call turns."""
            if not message.thinking_blocks:
                return []

            blocks: list[dict[str, str]] = []
            for block in message.thinking_blocks:
                if isinstance(block, ThinkingBlock):
                    blocks.append(
                        {
                            "type": "thinking",
                            "thinking": block.thinking,
                            "signature": block.signature,
                        }
                    )
                elif isinstance(block, RedactedThinkingBlock):
                    blocks.append({"type": "redacted_thinking", "data": block.data})
                elif isinstance(block, Mapping):
                    block_type = block.get("type")
                    if (
                        block_type == "thinking"
                        and block.get("thinking")
                        and block.get("signature")
                    ):
                        blocks.append(
                            {
                                "type": "thinking",
                                "thinking": block["thinking"],
                                "signature": block["signature"],
                            }
                        )
                    elif block_type == "redacted_thinking" and block.get("data"):
                        blocks.append(
                            {"type": "redacted_thinking", "data": block["data"]}
                        )
            return blocks

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

        def build_tool_result_block(message: ToolMessage) -> ToolResultBlockParam:
            return ToolResultBlockParam(
                type="tool_result",
                tool_use_id=message.tool_call_id,
                content=cast(
                    Any,
                    message.content
                    if isinstance(message.content, str)
                    else " ".join(content_to_text_chunks(message.content)),
                ),
            )

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
                thinking_blocks = extract_thinking_blocks(message)
                if message.tool_calls:
                    content_blocks: list[Any] = [*thinking_blocks]
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
                if thinking_blocks:
                    content_blocks: list[Any] = [*thinking_blocks]
                    for text_chunk in content_to_text_chunks(message.content):
                        content_blocks.append(
                            TextBlockParam(type="text", text=text_chunk)
                        )
                    return AnthropicMessageParam(
                        role="assistant",
                        content=content_blocks,
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
                    content=[build_tool_result_block(message)],
                )
            elif isinstance(message, TextMessage):
                return AnthropicMessageParam(role="user", content=message.content)
            else:
                raise ValueError(f"Invalid chat message: {message}")

        def extract_system_content(messages: Messages) -> str:
            """Extract and concatenate system message contents."""
            system_contents = []
            for msg in messages:
                if isinstance(msg, SystemMessage):
                    content = msg.content
                    system_contents.append(" ".join(content_to_text_chunks(content)))
            return "\n\n".join(system_contents)

        system = extract_system_content(messages)
        prompt: list[AnthropicMessageParam] = []
        pending_tool_results: list[ToolResultBlockParam] = []

        def flush_tool_results() -> None:
            nonlocal pending_tool_results
            if not pending_tool_results:
                return
            prompt.append(
                AnthropicMessageParam(role="user", content=pending_tool_results)
            )
            pending_tool_results = []

        for message in messages:
            if isinstance(message, ToolMessage):
                pending_tool_results.append(build_tool_result_block(message))
                continue

            flush_tool_results()
            converted = from_chat_message(message)
            if converted is not None:
                prompt.append(converted)

        flush_tool_results()

        return prompt, {"system": system}

    async def to_native_tool(self, tool: Tool) -> AnthropicToolParam:
        return AnthropicToolParam(
            name=tool.name,
            description=tool.description,
            input_schema=tool.parameters,
        )

    @_handle_anthropic_overlong_prompt
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
            sampling_args.pop("stop", None)
            if max_tokens is None:
                self.logger.warning(
                    "max_tokens is not set but Anthropic /v1/messages endpoint requires it, falling back to max_tokens=4096"
                )
                max_tokens = 4096
            sampling_args["max_tokens"] = max_tokens

            return {k: v for k, v in sampling_args.items() if v is not None}

        # Remove internal framework keys not recognized by the Anthropic SDK
        kwargs.pop("state", None)

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

    async def raise_from_native_response(self, response: AnthropicMessage) -> None:
        pass

    async def from_native_response(self, response: AnthropicMessage) -> Response:
        def parse_content(
            content_blocks: list[ContentBlock],
        ) -> tuple[
            str, str, list[ToolCall], list[ThinkingBlock | RedactedThinkingBlock]
        ]:
            content = ""
            reasoning_content = ""
            tool_calls: list[ToolCall] = []
            thinking_blocks: list[ThinkingBlock | RedactedThinkingBlock] = []
            for content_block in content_blocks:
                if content_block.type == "text":
                    text_value = getattr(content_block, "text", None)
                    if isinstance(text_value, str):
                        content += text_value
                elif content_block.type == "thinking":
                    thinking_value = getattr(content_block, "thinking", None)
                    if isinstance(thinking_value, str):
                        reasoning_content += thinking_value
                        signature_value = getattr(content_block, "signature", None)
                        if isinstance(signature_value, str):
                            thinking_blocks.append(
                                ThinkingBlock(
                                    type="thinking",
                                    thinking=thinking_value,
                                    signature=signature_value,
                                )
                            )
                elif content_block.type == "redacted_thinking":
                    data_value = getattr(content_block, "data", None)
                    if isinstance(data_value, str):
                        thinking_blocks.append(
                            RedactedThinkingBlock(
                                type="redacted_thinking", data=data_value
                            )
                        )
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
            return content, reasoning_content, tool_calls, thinking_blocks

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

        content, reasoning_content, tool_calls, thinking_blocks = parse_content(
            response.content
        )
        is_truncated = response.stop_reason == "max_tokens"

        input_tokens = response.usage.input_tokens
        output_tokens = response.usage.output_tokens

        return Response(
            id=response.id,
            model=response.model,
            created=int(time.time()),
            usage=Usage(
                prompt_tokens=input_tokens,
                completion_tokens=output_tokens,
                reasoning_tokens=0,
                total_tokens=input_tokens + output_tokens,
            ),
            message=ResponseMessage(
                content=content,
                reasoning_content=reasoning_content or None,
                thinking_blocks=thinking_blocks or None,
                tool_calls=tool_calls or None,
                finish_reason=parse_finish_reason(response),
                is_truncated=is_truncated,
                tokens=None,
            ),
        )
