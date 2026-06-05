"""OpenAI-compatible chat-completions client.

Distilled from v1's 545-line client: message<->wire translation, tool schemas,
best-effort reasoning_content. Token-id/logprob/routed-experts/audio handling is
dropped (training-only). This is the one place raw provider dicts cross into our
typed `Response`.
"""

from openai import AsyncOpenAI, OpenAIError

from verifiers.v2.clients.client import Client
from verifiers.v2.errors import ModelError
from verifiers.v2.types import (
    AssistantMessage,
    FinishReason,
    Message,
    Messages,
    Response,
    SamplingConfig,
    Tool,
    ToolCall,
    Usage,
)

FINISH_REASONS = frozenset({"stop", "length", "tool_calls"})


def message_to_wire(message: Message) -> dict:
    if message.role == "assistant":
        wire: dict = {"role": "assistant", "content": message.content}
        if message.tool_calls:
            wire["tool_calls"] = [
                {
                    "id": call.id,
                    "type": "function",
                    "function": {"name": call.name, "arguments": call.arguments},
                }
                for call in message.tool_calls
            ]
        return wire
    if message.role == "tool":
        return {
            "role": "tool",
            "tool_call_id": message.tool_call_id,
            "content": message.content,
        }
    return {"role": message.role, "content": message.content}


def tool_to_wire(tool: Tool) -> dict:
    function: dict = {
        "name": tool.name,
        "description": tool.description,
        "parameters": tool.parameters,
    }
    if tool.strict is not None:
        function["strict"] = tool.strict
    return {"type": "function", "function": function}


def response_from_wire(completion) -> Response:
    choice = completion.choices[0]
    message = choice.message
    tool_calls = [
        ToolCall(id=tc.id, name=tc.function.name, arguments=tc.function.arguments)
        for tc in (message.tool_calls or [])
    ] or None
    finish: FinishReason = (
        choice.finish_reason if choice.finish_reason in FINISH_REASONS else None
    )
    usage = (
        Usage(
            prompt_tokens=completion.usage.prompt_tokens,
            completion_tokens=completion.usage.completion_tokens,
        )
        if completion.usage
        else None
    )
    return Response(
        id=completion.id,
        created=completion.created,
        model=completion.model,
        message=AssistantMessage(
            content=message.content,
            reasoning_content=getattr(message, "reasoning_content", None),
            tool_calls=tool_calls,
        ),
        finish_reason=finish,
        usage=usage,
    )


class OpenAIChatCompletionsClient(Client):
    def __init__(self, openai: AsyncOpenAI) -> None:
        self.openai = openai

    async def get_response(
        self,
        prompt: Messages,
        model: str,
        sampling_args: SamplingConfig,
        tools: list[Tool] | None = None,
    ) -> Response:
        body: dict = {
            "model": model,
            "messages": [message_to_wire(m) for m in prompt],
            **sampling_args.model_dump(exclude_none=True),
        }
        if tools:
            body["tools"] = [tool_to_wire(t) for t in tools]
        try:
            completion = await self.openai.chat.completions.create(**body)
        except OpenAIError as e:
            raise ModelError(str(e)) from e
        return response_from_wire(completion)

    async def close(self) -> None:
        await self.openai.close()
