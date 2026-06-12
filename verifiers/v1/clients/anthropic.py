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


def content_to_wire(content) -> str | list[ContentBlockParam]:
    if isinstance(content, str):
        return content
    parts: list[ContentBlockParam] = []
    for part in content:
        if isinstance(part, TextContentPart):
            parts.append(TextBlockParam(type="text", text=part.text))
            continue
        if part.image_url.url.startswith("data:"):
            metadata, data = part.image_url.url.removeprefix("data:").split(",", 1)
            media_type, *parameters = metadata.split(";")
            if not any(parameter.lower() == "base64" for parameter in parameters):
                raise ValueError("Anthropic image data URIs must be base64 encoded")
            source = Base64ImageSourceParam(
                type="base64",
                media_type=cast(Any, media_type.lower()),
                data=data,
            )
        else:
            source = URLImageSourceParam(type="url", url=part.image_url.url)
        parts.append(
            ImageBlockParam(
                type="image",
                source=source,
            )
        )
    return parts


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
    prompt: list[MessageParam] = []
    for message in messages:
        wire = message_to_wire(message)
        if wire is None:
            continue
        previous_content = prompt[-1]["content"] if prompt else None
        if (
            isinstance(message, ToolMessage)
            and prompt[-1]["role"] == "user"
            and isinstance(previous_content, list)
        ):
            content = wire["content"]
            assert not isinstance(content, str)
            cast(list[ContentBlockParam], previous_content).extend(
                cast(list[ContentBlockParam], content)
            )
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
    if not content and not tool_calls:
        raise ModelError("Anthropic Messages returned no content or tool calls")

    finish_reasons: dict[str, FinishReason] = {
        "end_turn": "stop",
        "stop_sequence": "stop",
        "max_tokens": "length",
        "tool_use": "tool_calls",
        "refusal": "stop",
    }
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
        finish_reason=finish_reasons.get(response.stop_reason or ""),
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
        sampling.pop("n", None)
        if stop := sampling.pop("stop", None):
            sampling["stop_sequences"] = [stop] if isinstance(stop, str) else stop
        max_tokens = sampling.pop("max_tokens", None)
        if max_tokens is None:
            raise ValueError("Anthropic Messages requires max_tokens")
        if any(
            isinstance(message, SystemMessage)
            and not isinstance(message.content, str)
            and any(not isinstance(part, TextContentPart) for part in message.content)
            for message in prompt
        ):
            raise ValueError("Anthropic system messages do not support images")
        body: dict[str, Any] = {
            "model": model,
            "messages": messages_to_wire(prompt),
            "max_tokens": max_tokens,
            **sampling,
        }
        system = "\n\n".join(
            (
                message.content
                if isinstance(message.content, str)
                else "".join(part.text for part in message.content)
            )
            for message in prompt
            if isinstance(message, SystemMessage)
        )
        if system:
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
        streaming = bool(body.pop("stream", False))
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
