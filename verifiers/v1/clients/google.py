"""Google Gemini generateContent client."""

import base64
import json
import time
from typing import Any, cast

from google import genai
from google.genai import errors, types

from verifiers.v1.clients.client import Client
from verifiers.v1.errors import ModelError
from verifiers.v1.types import (
    AssistantMessage,
    FinishReason,
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


MEDIA_RESOLUTIONS = {
    "low": types.PartMediaResolutionLevel.MEDIA_RESOLUTION_LOW,
    "high": types.PartMediaResolutionLevel.MEDIA_RESOLUTION_HIGH,
    "original": types.PartMediaResolutionLevel.MEDIA_RESOLUTION_ULTRA_HIGH,
}


def content_to_wire(content) -> list[types.Part]:
    if isinstance(content, str):
        return [types.Part.from_text(text=content)]
    parts: list[types.Part] = []
    for part in content:
        if isinstance(part, TextContentPart):
            parts.append(types.Part.from_text(text=part.text))
            continue
        url = part.image_url.url
        if not url.startswith("data:"):
            raise ValueError("Google images must use data URIs")
        metadata, data = url.removeprefix("data:").split(",", 1)
        mime_type, *parameters = metadata.split(";")
        if not any(parameter.lower() == "base64" for parameter in parameters):
            raise ValueError("Google image data URIs must be base64 encoded")
        parts.append(
            types.Part.from_bytes(
                data=base64.b64decode(data),
                mime_type=mime_type.lower(),
                media_resolution=MEDIA_RESOLUTIONS.get(part.image_url.detail),
            )
        )
    return parts


def system_to_wire(messages: Messages) -> str | list[types.Part] | None:
    """Google takes system instructions in the request config: joined text when all
    system messages are plain strings, content parts otherwise (e.g. images)."""
    systems = [m for m in messages if isinstance(m, SystemMessage)]
    if not systems:
        return None
    if all(isinstance(m.content, str) for m in systems):
        return "\n\n".join(cast(str, m.content) for m in systems)
    return [part for m in systems for part in content_to_wire(m.content)]


def assistant_to_wire(message: AssistantMessage) -> types.Content:
    # Native parts carry thought signatures required on the next turn, so replay
    # them unchanged when available; otherwise rebuild from the portable fields.
    if message.provider_state:
        parts = [types.Part.model_validate(part) for part in message.provider_state]
        return types.Content(role="model", parts=parts)
    parts: list[types.Part] = []
    if message.reasoning_content:
        parts.append(types.Part(text=message.reasoning_content, thought=True))
    if message.content:
        parts.append(types.Part.from_text(text=message.content))
    for call in message.tool_calls or []:
        parts.append(
            types.Part(
                function_call=types.FunctionCall(
                    id=call.id,
                    name=call.name,
                    args=json.loads(call.arguments),
                )
            )
        )
    return types.Content(role="model", parts=parts)


def tool_result_to_wire(message: ToolMessage, name: str) -> types.Part:
    # Google wants a JSON object; wrap non-object results.
    try:
        result = json.loads(message.content)
    except json.JSONDecodeError:
        result = message.content
    if not isinstance(result, dict):
        result = {"result": result}
    return types.Part(
        function_response=types.FunctionResponse(
            id=message.tool_call_id,
            name=name,
            response=result,
        )
    )


def messages_to_wire(messages: Messages) -> list[types.Content]:
    """Convert the prompt, folding consecutive tool results into one user turn
    (Google expects parallel function responses grouped in a single message)."""
    prompt: list[types.Content] = []
    call_names: dict[str, str] = {}  # function responses must repeat the call's name
    for message in messages:
        if isinstance(message, SystemMessage):  # sent as system_instruction
            continue
        if isinstance(message, UserMessage):
            prompt.append(
                types.Content(role="user", parts=content_to_wire(message.content))
            )
        elif isinstance(message, AssistantMessage):
            call_names.update({call.id: call.name for call in message.tool_calls or []})
            prompt.append(assistant_to_wire(message))
        else:
            part = tool_result_to_wire(message, call_names[message.tool_call_id])
            last_parts = prompt[-1].parts if prompt else None
            if (
                last_parts
                and prompt[-1].role == "user"
                and all(item.function_response for item in last_parts)
            ):
                last_parts.append(part)
            else:
                prompt.append(types.Content(role="user", parts=[part]))
    return prompt


def merge_stream_chunks(
    chunks: list[types.GenerateContentResponse],
) -> types.GenerateContentResponse:
    """Combine streamed chunks into one response (Google streams response deltas
    with no final-aggregate helper). The last chunk carries the metadata (usage,
    finish reason); candidate zero's content parts concatenate across chunks."""
    if not chunks:
        raise ModelError("Google stream returned no chunks")
    response = chunks[-1].model_copy(deep=True)
    candidates = [
        candidate
        for chunk in chunks
        for candidate in chunk.candidates or []
        if candidate.index in (None, 0)
    ]
    if not candidates:
        return response
    candidate = candidates[-1].model_copy(deep=True)
    candidate.content = types.Content(
        role=candidate.content.role if candidate.content else None,
        parts=[
            part
            for item in candidates
            if item.content
            for part in item.content.parts or []
        ],
    )
    response.candidates = [candidate]
    return response


def response_from_wire(response: types.GenerateContentResponse, model: str) -> Response:
    if not response.candidates:
        raise ModelError("Google returned no candidates")
    candidate = response.candidates[0]
    parts = candidate.content.parts if candidate.content else None
    if not parts:
        raise ModelError("Google returned no content")

    content = ""
    reasoning = ""
    tool_calls: list[ToolCall] = []
    for part in parts:
        if part.text:
            if part.thought:
                reasoning += part.text
            else:
                content += part.text
        if part.function_call and part.function_call.name:
            tool_calls.append(
                ToolCall(
                    id=part.function_call.id or f"call_{len(tool_calls)}",
                    name=part.function_call.name,
                    arguments=json.dumps(part.function_call.args or {}),
                )
            )
    if not content and not tool_calls:
        raise ModelError("Google returned no content or tool calls")

    finish_reason: FinishReason = None
    if tool_calls:
        finish_reason = "tool_calls"
    elif candidate.finish_reason == types.FinishReason.STOP:
        finish_reason = "stop"
    elif candidate.finish_reason == types.FinishReason.MAX_TOKENS:
        finish_reason = "length"

    usage = response.usage_metadata
    prompt_tokens = usage.prompt_token_count if usage else None
    completion_tokens = usage.candidates_token_count if usage else None
    return Response(
        id=response.response_id or "",
        created=(
            int(response.create_time.timestamp())
            if response.create_time
            else int(time.time())
        ),
        model=response.model_version or model,
        message=AssistantMessage(
            content=content or None,
            reasoning_content=reasoning or None,
            tool_calls=tool_calls or None,
            provider_state=[
                part.model_dump(mode="json", by_alias=True, exclude_none=True)
                for part in parts
            ],
        ),
        finish_reason=finish_reason,
        usage=(
            Usage(
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
            )
            if prompt_tokens is not None and completion_tokens is not None
            else None
        ),
    )


class GoogleResponsesClient(Client):
    def __init__(self, google: genai.Client) -> None:
        self.google = google

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
        if n := sampling.pop("n", None):
            sampling["candidate_count"] = n
        if stop := sampling.pop("stop", None):
            sampling["stop_sequences"] = [stop] if isinstance(stop, str) else stop
        # OpenAI's `logprobs` flag is Google's `response_logprobs`; OpenAI's
        # `top_logprobs` count is what Google calls `logprobs`.
        logprobs = sampling.pop("logprobs", None)
        if logprobs is not None:
            sampling["response_logprobs"] = bool(logprobs)
        if top_logprobs := sampling.pop("top_logprobs", None):
            sampling["logprobs"] = top_logprobs
        if (system := system_to_wire(prompt)) is not None:
            sampling["system_instruction"] = system
        if tools:
            sampling["tools"] = [
                types.Tool(
                    function_declarations=[
                        types.FunctionDeclaration(
                            name=tool.name,
                            description=tool.description,
                            parameters_json_schema=tool.parameters,
                        )
                        for tool in tools
                    ]
                )
            ]
        request = {
            # eval model ids may carry a provider prefix (e.g. "google/gemini-...")
            "model": model.rsplit("/", 1)[-1],
            "contents": cast(types.ContentListUnion, messages_to_wire(prompt)),
            "config": types.GenerateContentConfig.model_validate(sampling),
        }
        try:
            if streaming:
                stream = await self.google.aio.models.generate_content_stream(**request)
                response = merge_stream_chunks([chunk async for chunk in stream])
            else:
                response = await self.google.aio.models.generate_content(**request)
        except errors.APIError as e:
            raise ModelError(str(e)) from e
        return response_from_wire(response, model)

    async def close(self) -> None:
        await self.google.aio.aclose()
        self.google.close()
