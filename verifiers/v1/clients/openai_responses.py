"""OpenAI Responses API client."""

from typing import Any, cast

import httpx
from openai import AsyncOpenAI, OpenAIError
from openai.types.responses import (
    EasyInputMessageParam,
    FunctionToolParam,
    Response as OpenAIResponse,
    ResponseFunctionToolCall,
    ResponseFunctionToolCallParam,
    ResponseInputContentParam,
    ResponseInputImageParam,
    ResponseInputItemParam,
    ResponseInputTextParam,
    ResponseOutputMessage,
    ResponseOutputRefusal,
    ResponseReasoningItem,
)
from openai.types.responses.response_input_param import FunctionCallOutput

from verifiers.v1.clients.client import Client, RelayReply, relay_headers, relay_post
from verifiers.v1.clients.openai import model_error
from verifiers.v1.errors import ModelError
from verifiers.v1.types import (
    AssistantMessage,
    FinishReason,
    Message,
    Messages,
    Response,
    SamplingConfig,
    SystemMessage,
    Tool,
    ToolCall,
    ToolMessage,
    Usage,
    UserMessage,
)

FINISH_REASONS: dict[str, FinishReason] = {"completed": "stop", "incomplete": "length"}
FINAL_EVENTS = ("response.completed", "response.incomplete", "response.failed")


def content_to_wire(content) -> str | list[ResponseInputContentParam]:
    if isinstance(content, str):
        return content
    return [
        ResponseInputTextParam(type="input_text", text=part.text)
        if part.type == "text"
        else ResponseInputImageParam(
            type="input_image",
            image_url=part.image_url.url,
            detail=part.image_url.detail or "auto",
        )
        for part in content
    ]


def message_to_wire(message: Message) -> list[ResponseInputItemParam]:
    if isinstance(message, ToolMessage):
        return [
            FunctionCallOutput(
                type="function_call_output",
                call_id=message.tool_call_id,
                output=message.content,
            )
        ]
    if isinstance(message, AssistantMessage):
        # Native output items carry reasoning and tool-call state required by the
        # Responses API on the next turn, so replay them unchanged when available.
        if message.provider_state:
            return cast(list[ResponseInputItemParam], message.provider_state)
        items: list[ResponseInputItemParam] = []
        if message.content:
            items.append(
                EasyInputMessageParam(role="assistant", content=message.content)
            )
        items.extend(
            ResponseFunctionToolCallParam(
                type="function_call",
                call_id=call.id,
                name=call.name,
                arguments=call.arguments,
            )
            for call in message.tool_calls or []
        )
        return items
    assert isinstance(message, (SystemMessage, UserMessage))
    return [
        EasyInputMessageParam(
            role=message.role,
            content=content_to_wire(message.content),
        )
    ]


def response_from_wire(response: OpenAIResponse) -> Response:
    content = response.output_text
    reasoning: list[str] = []
    has_reasoning = False
    tool_calls: list[ToolCall] = []
    # `output_text` is only the visible text. Inspect output items for refusals,
    # reasoning summaries, tool calls, and the provider state needed for continuation.
    for item in response.output:
        if isinstance(item, ResponseOutputMessage):
            content += "".join(
                part.refusal
                for part in item.content
                if isinstance(part, ResponseOutputRefusal)
            )
        elif isinstance(item, ResponseReasoningItem):
            has_reasoning = True
            reasoning += [part.text for part in item.summary]
            reasoning += [part.text for part in item.content or []]
        elif isinstance(item, ResponseFunctionToolCall):
            tool_calls.append(
                ToolCall(
                    id=item.call_id,
                    name=item.name,
                    arguments=item.arguments,
                )
            )
    if not content and not has_reasoning and not tool_calls:
        raise ModelError("OpenAI Responses returned no output")

    return Response(
        id=response.id,
        created=int(response.created_at),
        model=response.model,
        message=AssistantMessage(
            content=content or None,
            reasoning_content="\n".join(reasoning) or None,
            tool_calls=tool_calls or None,
            provider_state=[
                item.model_dump(mode="json", exclude_none=True)
                for item in response.output
            ],
        ),
        finish_reason=(
            "tool_calls" if tool_calls else FINISH_REASONS.get(response.status or "")
        ),
        usage=(
            Usage(
                prompt_tokens=response.usage.input_tokens,
                completion_tokens=response.usage.output_tokens,
            )
            if response.usage
            else None
        ),
    )


class OpenAIResponsesClient(Client):
    dialect = "responses"

    def __init__(self, openai: AsyncOpenAI) -> None:
        self.openai = openai
        self._http: httpx.AsyncClient | None = None

    async def relay(self, body: bytes, route: str) -> RelayReply:
        # The SDK's base_url already carries `/v1` (the OpenAI convention).
        if self._http is None:
            self._http = httpx.AsyncClient(timeout=None)
        url = str(self.openai.base_url).rstrip("/") + route.removeprefix("/v1")
        return await relay_post(self._http, url, relay_headers(self.openai), body)

    async def get_response(
        self,
        prompt: Messages,
        model: str,
        sampling_args: SamplingConfig,
        tools: list[Tool] | None = None,
    ) -> Response:
        sampling: dict[str, Any] = sampling_args.model_dump(exclude_none=True)
        streaming = bool(sampling.pop("stream", False))
        if max_tokens := sampling.pop("max_tokens", None):
            sampling["max_output_tokens"] = max_tokens
        if sampling.pop("stop", None):
            raise ValueError("OpenAI Responses does not support stop sequences")
        if sampling.pop("n", 1) != 1:
            raise ValueError("OpenAI Responses only supports n=1")
        body: dict[str, Any] = {
            "model": model,
            "input": [item for message in prompt for item in message_to_wire(message)],
            **sampling,
        }
        if tools:
            body["tools"] = [
                FunctionToolParam(
                    type="function",
                    name=tool.name,
                    description=tool.description,
                    parameters=tool.parameters,
                    strict=tool.strict,
                )
                for tool in tools
            ]
        try:
            if streaming:
                # The SDK's final-response helper rejects valid incomplete
                # responses, so pick the terminal event out of the stream ourselves.
                response = None
                async with await self.openai.responses.create(
                    **body, stream=True
                ) as events:
                    async for event in events:
                        if event.type in FINAL_EVENTS:
                            response = event.response
                            break
                if response is None:
                    raise ModelError(
                        "OpenAI Responses stream ended without a final response"
                    )
            else:
                response = await self.openai.responses.create(**body)
        except OpenAIError as e:
            raise model_error(e) from e
        return response_from_wire(response)

    async def close(self) -> None:
        await self.openai.close()
        if self._http is not None:
            await self._http.aclose()
