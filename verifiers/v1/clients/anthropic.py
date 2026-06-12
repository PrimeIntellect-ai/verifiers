"""Anthropic Messages API client."""

import json
import time
from typing import Any, cast

from anthropic import AnthropicError, AsyncAnthropic
from anthropic.types import (
    Base64ImageSourceParam,
    ContentBlockParam,
    ImageBlockParam,
    Message as AnthropicMessage,
    MessageParam,
    RedactedThinkingBlock,
    TextBlock,
    TextBlockParam,
    ThinkingBlock,
    ToolParam as AnthropicTool,
    ToolResultBlockParam,
    ToolUseBlock,
    ToolUseBlockParam,
    URLImageSourceParam,
)

from verifiers.v1.clients.client import Client
from verifiers.v1.errors import ModelError
from verifiers.v1.types import (
    AssistantMessage,
    FinishReason,
    Message,
    Messages,
    Response,
    SamplingConfig,
    SystemMessage,
    TextContentPart,
    Tool,
    ToolCall,
    ToolMessage,
    Usage,
    UserMessage,
)


FINISH_REASONS: dict[str, FinishReason] = {
    "end_turn": "stop",
    "stop_sequence": "stop",
    "max_tokens": "length",
    "tool_use": "tool_calls",
    "refusal": "stop",
}


def content_to_wire(content) -> str | list[ContentBlockParam]:
    if isinstance(content, str):
        return content
    parts: list[ContentBlockParam] = []
    for part in content:
        if isinstance(part, TextContentPart):
            parts.append(TextBlockParam(type="text", text=part.text))
            continue
        url = part.image_url.url
        if url.startswith("data:"):
            metadata, data = url.removeprefix("data:").split(",", 1)
            media_type, *parameters = metadata.split(";")
            if not any(parameter.lower() == "base64" for parameter in parameters):
                raise ValueError("Anthropic image data URIs must be base64 encoded")
            source: Base64ImageSourceParam | URLImageSourceParam = (
                Base64ImageSourceParam(
                    type="base64", media_type=cast(Any, media_type.lower()), data=data
                )
            )
        else:
            source = URLImageSourceParam(type="url", url=url)
        parts.append(ImageBlockParam(type="image", source=source))
    return parts


def system_to_wire(messages: Messages) -> str:
    """Join system messages into Anthropic's top-level `system` string."""
    texts: list[str] = []
    for message in messages:
        if not isinstance(message, SystemMessage):
            continue
        if isinstance(message.content, str):
            texts.append(message.content)
        elif all(isinstance(part, TextContentPart) for part in message.content):
            texts.append(
                "".join(
                    part.text
                    for part in message.content
                    if isinstance(part, TextContentPart)
                )
            )
        else:
            raise ValueError("Anthropic system messages do not support images")
    return "\n\n".join(texts)


def message_to_wire(message: Message) -> MessageParam | None:
    if isinstance(message, SystemMessage):
        return None
    if isinstance(message, UserMessage):
        return MessageParam(role="user", content=content_to_wire(message.content))
    if isinstance(message, ToolMessage):
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
    assert isinstance(message, AssistantMessage)
    content = [cast(ContentBlockParam, block) for block in message.provider_state or []]
    if message.content:
        content.append(TextBlockParam(type="text", text=message.content))
    for call in message.tool_calls or []:
        content.append(
            ToolUseBlockParam(
                type="tool_use",
                id=call.id,
                name=call.name,
                input=json.loads(call.arguments),
            )
        )
    return MessageParam(role="assistant", content=content)


def messages_to_wire(messages: Messages) -> list[MessageParam]:
    """Convert the prompt, folding consecutive tool results into one user message
    (Anthropic requires tool results as blocks of the following user turn)."""
    prompt: list[MessageParam] = []
    for message in messages:
        wire = message_to_wire(message)
        if wire is None:  # system messages go in the top-level `system` field
            continue
        last_content = prompt[-1]["content"] if prompt else None
        if (
            isinstance(message, ToolMessage)
            and isinstance(last_content, list)
            and prompt[-1]["role"] == "user"
        ):
            last_content.extend(cast(list[ContentBlockParam], wire["content"]))
        else:
            prompt.append(wire)
    return prompt


def response_from_wire(response: AnthropicMessage) -> Response:
    content = ""
    reasoning = ""
    thinking: list[dict] = []
    tool_calls: list[ToolCall] = []
    for block in response.content:
        if isinstance(block, TextBlock):
            content += block.text
        elif isinstance(block, ThinkingBlock):
            thinking.append(block.model_dump(mode="json"))
            reasoning += block.thinking
        elif isinstance(block, RedactedThinkingBlock):
            thinking.append(block.model_dump(mode="json"))
        elif isinstance(block, ToolUseBlock):
            tool_calls.append(
                ToolCall(
                    id=block.id,
                    name=block.name,
                    arguments=json.dumps(block.input),
                )
            )
    if not content and not thinking and not tool_calls:
        raise ModelError("Anthropic Messages returned no output")
    return Response(
        id=response.id,
        created=int(time.time()),
        model=response.model,
        message=AssistantMessage(
            content=content or None,
            reasoning_content=reasoning or None,
            tool_calls=tool_calls or None,
            provider_state=thinking or None,
        ),
        finish_reason=FINISH_REASONS.get(response.stop_reason or ""),
        usage=Usage(
            prompt_tokens=response.usage.input_tokens,
            completion_tokens=response.usage.output_tokens,
        ),
    )


class AnthropicMessagesClient(Client):
    def __init__(self, anthropic: AsyncAnthropic) -> None:
        self.anthropic = anthropic

    async def get_response(
        self,
        prompt: Messages,
        model: str,
        sampling_args: SamplingConfig,
        tools: list[Tool] | None = None,
    ) -> Response:
        sampling: dict[str, Any] = sampling_args.model_dump(exclude_none=True)
        streaming = bool(sampling.pop("stream", False))
        sampling.pop("n", None)  # Anthropic has no n parameter
        if "max_tokens" not in sampling:
            raise ValueError("Anthropic Messages requires max_tokens")
        if stop := sampling.pop("stop", None):
            sampling["stop_sequences"] = [stop] if isinstance(stop, str) else stop
        body: dict[str, Any] = {
            "model": model,
            "messages": messages_to_wire(prompt),
            **sampling,
        }
        if system := system_to_wire(prompt):
            body["system"] = system
        if tools:
            body["tools"] = [
                AnthropicTool(
                    name=tool.name,
                    description=tool.description,
                    input_schema=tool.parameters,
                )
                for tool in tools
            ]
        try:
            if streaming:
                async with self.anthropic.messages.stream(**body) as stream:
                    response = await stream.get_final_message()
            else:
                response = await self.anthropic.messages.create(**body)
        except AnthropicError as e:
            raise ModelError(str(e)) from e
        return response_from_wire(response)

    async def close(self) -> None:
        await self.anthropic.close()
