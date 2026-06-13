"""The Google Gemini GenerateContent dialect.

The Google Gen AI SDK sends model turns to model-scoped `generateContent` or
`streamGenerateContent` routes. Requests and responses are relayed unchanged; this module
parses copies into Verifiers messages for the trace and maps eval sampling into
`generationConfig`.
"""

import json
from collections.abc import Mapping

from pydantic import BaseModel, ConfigDict

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


class GenerateContentResponse(BaseModel):
    """Permissive parse-only view of Google's evolving response object."""

    model_config = ConfigDict(extra="allow")


def parse_content(parts: list[dict] | None) -> str | list[ContentPart]:
    """Google text/image parts -> typed message content."""
    content: list[ContentPart] = []
    for part in parts or []:
        if part.get("text") is not None and not part.get("thought"):
            content.append(TextContentPart(text=part["text"]))
            continue
        data = part.get("inlineData") or {}
        if data.get("mimeType", "").startswith("image/"):
            content.append(
                ImageUrlContentPart(
                    image_url=ImageUrlSource(
                        url=f"data:{data['mimeType']};base64,{data.get('data', '')}"
                    )
                )
            )
            continue
        data = part.get("fileData") or {}
        if data.get("mimeType", "").startswith("image/"):
            content.append(
                ImageUrlContentPart(
                    image_url=ImageUrlSource(url=data.get("fileUri", ""))
                )
            )
    if all(isinstance(part, TextContentPart) for part in content):
        return "".join(part.text for part in content)
    return content


def parse_model_content(content: dict | None) -> AssistantMessage:
    """A Google model Content -> one assistant message."""
    text = ""
    reasoning = ""
    calls: list[ToolCall] = []
    for part in (content or {}).get("parts") or []:
        if part.get("text") is not None:
            if part.get("thought"):
                reasoning += part["text"]
            else:
                text += part["text"]
        if call := part.get("functionCall"):
            calls.append(
                ToolCall(
                    id=call.get("id") or call.get("name", ""),
                    name=call.get("name", ""),
                    arguments=json.dumps(call.get("args") or {}),
                )
            )
    return AssistantMessage(
        content=text or None,
        reasoning_content=reasoning or None,
        tool_calls=calls or None,
    )


def parse_messages(body: dict) -> Messages:
    """Google systemInstruction + contents -> typed messages."""
    prompt: Messages = []
    if system := body.get("systemInstruction"):
        prompt.append(SystemMessage(content=parse_content(system.get("parts"))))
    for content in body.get("contents") or []:
        parts = content.get("parts") or []
        if content.get("role") == "model":
            prompt.append(parse_model_content(content))
            continue
        rest = []
        for part in parts:
            if response := part.get("functionResponse"):
                prompt.append(
                    ToolMessage(
                        tool_call_id=response.get("id") or response.get("name", ""),
                        content=json.dumps(response.get("response") or {}),
                    )
                )
            else:
                rest.append(part)
        if rest:
            prompt.append(UserMessage(content=parse_content(rest)))
    return prompt


def response_from_wire(response: GenerateContentResponse) -> Response:
    """A GenerateContentResponse -> a Verifiers response."""
    data = response.model_dump()
    candidates = data.get("candidates") or []
    candidate = next((c for c in candidates if c.get("index", 0) == 0), {})
    message = parse_model_content(candidate.get("content"))
    finish: FinishReason = (
        "tool_calls"
        if message.tool_calls
        else FINISH_REASONS.get(candidate.get("finishReason"))
    )
    metadata = data.get("usageMetadata") or {}
    usage = None
    if metadata:
        prompt_tokens = metadata.get("promptTokenCount", 0)
        completion_tokens = metadata.get("totalTokenCount")
        if completion_tokens is not None:
            completion_tokens -= prompt_tokens
        else:
            completion_tokens = metadata.get("candidatesTokenCount", 0) + metadata.get(
                "thoughtsTokenCount", 0
            )
        usage = Usage(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
        )
    return Response(
        id=data.get("responseId", ""),
        created=0,
        model=data.get("modelVersion", ""),
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
        tools = [
            Tool(
                name=declaration["name"],
                description=declaration.get("description", ""),
                parameters=declaration.get("parameters")
                or declaration.get("parametersJsonSchema")
                or {},
            )
            for tool in body.get("tools") or []
            for declaration in tool.get("functionDeclarations") or []
        ] or None
        return parse_messages(body), tools

    def parse_response(self, response: GenerateContentResponse) -> Response:
        return response_from_wire(response)

    def parse_stream(self, raw: bytes) -> Response:
        """Concatenate candidate-zero parts from each SSE chunk."""
        response: dict = {"candidates": [{"index": 0, "content": {"parts": []}}]}
        candidate = response["candidates"][0]
        for chunk in iter_sse(raw):
            for key in ("responseId", "modelVersion", "usageMetadata"):
                if chunk.get(key) is not None:
                    response[key] = chunk[key]
            current = next(
                (
                    item
                    for item in chunk.get("candidates") or []
                    if item.get("index", 0) == 0
                ),
                None,
            )
            if current is None:
                continue
            candidate["content"]["parts"].extend(
                (current.get("content") or {}).get("parts") or []
            )
            if current.get("finishReason") is not None:
                candidate["finishReason"] = current["finishReason"]
        return response_from_wire(GenerateContentResponse.model_validate(response))

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
