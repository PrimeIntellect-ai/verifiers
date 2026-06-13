"""The Google Gemini GenerateContent dialect.

The Google Gen AI SDK sends model turns to model-scoped `generateContent` or
`streamGenerateContent` routes. Requests and responses are relayed unchanged; this module
parses copies into Verifiers messages for the trace and maps eval sampling into
`generationConfig`.
"""

import json
from collections.abc import Mapping
from typing import Any

from pydantic import BaseModel, ConfigDict, Field
from pydantic.alias_generators import to_camel

from verifiers.v1.dialects.base import Dialect, iter_sse
from verifiers.v1.types import (
    AssistantMessage,
    ContentPart,
    FinishReason,
    ImageUrlContentPart,
    ImageUrlSource,
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

FINISH_REASONS = {"STOP": "stop", "MAX_TOKENS": "length"}
_SAMPLING_KEYS = ("temperature", "topP", "maxOutputTokens")


class GoogleModel(BaseModel):
    """Permissive typed view of Google wire objects."""

    model_config = ConfigDict(
        alias_generator=to_camel, populate_by_name=True, extra="allow"
    )


class InlineData(GoogleModel):
    mime_type: str = ""
    data: str = ""


class FileData(GoogleModel):
    mime_type: str = ""
    file_uri: str = ""


class FunctionCall(GoogleModel):
    id: str | None = None
    name: str = ""
    args: dict[str, Any] = Field(default_factory=dict)


class FunctionResponse(GoogleModel):
    id: str | None = None
    name: str = ""
    response: dict[str, Any] = Field(default_factory=dict)


class Part(GoogleModel):
    text: str | None = None
    thought: bool = False
    inline_data: InlineData | None = None
    file_data: FileData | None = None
    function_call: FunctionCall | None = None
    function_response: FunctionResponse | None = None


class Content(GoogleModel):
    role: str | None = None
    parts: list[Part] = Field(default_factory=list)


class FunctionDeclaration(GoogleModel):
    name: str
    description: str = ""
    parameters: dict[str, Any] | None = None
    parameters_json_schema: dict[str, Any] | None = None


class GoogleTool(GoogleModel):
    function_declarations: list[FunctionDeclaration] = Field(default_factory=list)


class GenerateContentRequest(GoogleModel):
    system_instruction: Content | None = None
    contents: list[Content] = Field(default_factory=list)
    tools: list[GoogleTool] = Field(default_factory=list)


class Candidate(GoogleModel):
    content: Content | None = None
    finish_reason: str | None = None
    index: int = 0


class UsageMetadata(GoogleModel):
    prompt_token_count: int = 0
    candidates_token_count: int = 0
    thoughts_token_count: int = 0
    total_token_count: int | None = None


class GenerateContentResponse(GoogleModel):
    candidates: list[Candidate] = Field(default_factory=list)
    usage_metadata: UsageMetadata | None = None
    model_version: str = ""
    response_id: str = ""


def parse_content(parts: list[Part]) -> str | list[ContentPart]:
    """Google text/image parts -> typed message content."""
    content: list[ContentPart] = []
    for part in parts:
        if part.text is not None and not part.thought:
            content.append(TextContentPart(text=part.text))
            continue
        if part.inline_data and part.inline_data.mime_type.startswith("image/"):
            content.append(
                ImageUrlContentPart(
                    image_url=ImageUrlSource(
                        url=f"data:{part.inline_data.mime_type};base64,"
                        f"{part.inline_data.data}"
                    )
                )
            )
            continue
        if part.file_data and part.file_data.mime_type.startswith("image/"):
            content.append(
                ImageUrlContentPart(
                    image_url=ImageUrlSource(url=part.file_data.file_uri)
                )
            )
    if all(isinstance(part, TextContentPart) for part in content):
        return "".join(part.text for part in content)
    return content


def parse_assistant(parts: list[Part]) -> AssistantMessage:
    """Google model parts -> one assistant message."""
    calls = [
        ToolCall(
            id=call.id or call.name,
            name=call.name,
            arguments=json.dumps(call.args),
        )
        for part in parts
        if (call := part.function_call)
    ]
    return AssistantMessage(
        content="".join(part.text or "" for part in parts if not part.thought) or None,
        reasoning_content="".join(part.text or "" for part in parts if part.thought)
        or None,
        tool_calls=calls or None,
    )


def response_from_wire(response: GenerateContentResponse) -> Response:
    """A GenerateContentResponse -> a Verifiers response."""
    candidate = next((c for c in response.candidates if c.index == 0), Candidate())
    message = parse_assistant(candidate.content.parts if candidate.content else [])
    finish: FinishReason = (
        "tool_calls"
        if message.tool_calls
        else FINISH_REASONS.get(candidate.finish_reason or "")
    )
    usage = None
    if metadata := response.usage_metadata:
        completion_tokens = metadata.total_token_count
        if completion_tokens is not None:
            completion_tokens -= metadata.prompt_token_count
        else:
            completion_tokens = (
                metadata.candidates_token_count + metadata.thoughts_token_count
            )
        usage = Usage(
            prompt_tokens=metadata.prompt_token_count,
            completion_tokens=completion_tokens,
        )
    return Response(
        id=response.response_id,
        created=0,
        model=response.model_version,
        message=message,
        finish_reason=finish,
        usage=usage,
    )


class GoogleGenerateContentDialect(Dialect[dict, GenerateContentResponse]):
    """The Gemini Developer API GenerateContent wire format."""

    routes = (
        "/v1beta/models/{model}:generateContent",
        "/v1beta/models/{model}:streamGenerateContent",
        "/v1beta/tunedModels/{model}:generateContent",
        "/v1beta/tunedModels/{model}:streamGenerateContent",
    )
    response_type = GenerateContentResponse

    def upstream_route(self, model: str, stream: bool = False) -> str:
        resource = (
            model
            if model.startswith(("models/", "tunedModels/"))
            else f"models/{model}"
        )
        method = "streamGenerateContent?alt=sse" if stream else "generateContent"
        return f"/v1beta/{resource}:{method}"

    def auth_headers(self, api_key: str) -> dict[str, str]:
        return {"x-goog-api-key": api_key}

    def secret(self, headers: Mapping[str, str]) -> str:
        return headers.get("x-goog-api-key", "")

    def streaming(self, body: dict, route: str = "") -> bool:
        return route.endswith(":streamGenerateContent")

    def error_body(self, message: str) -> dict:
        return {
            "error": {
                "code": 400,
                "message": message,
                "status": "INVALID_ARGUMENT",
            }
        }

    def parse_request(self, body: dict) -> tuple[Messages, list[Tool] | None]:
        request = GenerateContentRequest.model_validate(body)
        prompt: Messages = []
        if request.system_instruction:
            prompt.append(
                SystemMessage(content=parse_content(request.system_instruction.parts))
            )
        for content in request.contents:
            if content.role == "model":
                prompt.append(parse_assistant(content.parts))
                continue
            prompt.extend(
                ToolMessage(
                    tool_call_id=response.id or response.name,
                    content=json.dumps(response.response),
                )
                for part in content.parts
                if (response := part.function_response)
            )
            user_parts = [
                part for part in content.parts if part.function_response is None
            ]
            if user_parts:
                prompt.append(UserMessage(content=parse_content(user_parts)))
        tools = [
            Tool(
                name=declaration.name,
                description=declaration.description,
                parameters=declaration.parameters
                or declaration.parameters_json_schema
                or {},
            )
            for tool in request.tools
            for declaration in tool.function_declarations
        ] or None
        return prompt, tools

    def parse_response(self, response: GenerateContentResponse) -> Response:
        return response_from_wire(response)

    def parse_stream(self, raw: bytes) -> Response:
        """Concatenate candidate-zero parts from each SSE chunk."""
        parts: list[Part] = []
        finish_reason = None
        response_id = ""
        model_version = ""
        usage = None
        for chunk in map(GenerateContentResponse.model_validate, iter_sse(raw)):
            response_id = chunk.response_id or response_id
            model_version = chunk.model_version or model_version
            usage = chunk.usage_metadata or usage
            current = next((c for c in chunk.candidates if c.index == 0), None)
            if current is None:
                continue
            if current.content:
                parts.extend(current.content.parts)
            finish_reason = current.finish_reason or finish_reason
        return response_from_wire(
            GenerateContentResponse(
                candidates=[
                    Candidate(
                        content=Content(role="model", parts=parts),
                        finish_reason=finish_reason,
                    )
                ],
                usage_metadata=usage,
                model_version=model_version,
                response_id=response_id,
            )
        )

    def apply_overrides(self, body: dict, model: str, sampling: SamplingConfig) -> dict:
        config = {
            key: value
            for key, value in (body.get("generationConfig") or {}).items()
            if key not in _SAMPLING_KEYS
        }
        values = sampling.model_dump(exclude_none=True)
        if "temperature" in values:
            config["temperature"] = values["temperature"]
        if "top_p" in values:
            config["topP"] = values["top_p"]
        if "max_tokens" in values:
            config["maxOutputTokens"] = values["max_tokens"]
        result = dict(body)
        if "generationConfig" in body or config:
            result["generationConfig"] = config
        return result
