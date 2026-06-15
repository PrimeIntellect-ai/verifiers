"""The Google Gemini GenerateContent dialect.

Supports the non-streaming Gemini Developer API for text and function-calling rollouts.
Requests and responses are relayed unchanged; this module only parses copies for the trace.
"""

import json
from collections.abc import Mapping

from pydantic import BaseModel, ConfigDict

from verifiers.v1.dialects.base import Dialect
from verifiers.v1.types import (
    AssistantMessage,
    FinishReason,
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

FINISH_REASONS: dict[str, FinishReason] = {"STOP": "stop", "MAX_TOKENS": "length"}
_SAMPLING_KEYS = ("temperature", "topP", "topK", "maxOutputTokens")
_GENERATION_KEYS = {
    *_SAMPLING_KEYS,
    "candidateCount",
    "frequencyPenalty",
    "logprobs",
    "presencePenalty",
    "responseLogprobs",
    "seed",
    "stopSequences",
    "thinkingConfig",
}
GEMINI_25_THINKING_BUDGETS = {
    "none": 0,
    "minimal": 1024,
    "low": 1024,
    "medium": 8192,
    "high": 24576,
}


class GenerateContentResponse(BaseModel):
    """Permissive parse-only view of Google's evolving response object."""

    model_config = ConfigDict(extra="allow")


def text(parts: list[dict]) -> str:
    return "".join(
        part.get("text", "")
        for part in parts
        if part.get("text") is not None and not part.get("thought")
    )


def parse_assistant(parts: list[dict]) -> AssistantMessage:
    calls = [
        ToolCall(
            id=call.get("id") or call.get("name", ""),
            name=call.get("name", ""),
            arguments=json.dumps(call.get("args") or {}),
        )
        for part in parts
        if (call := part.get("functionCall"))
    ]
    return AssistantMessage(
        content=text(parts) or None,
        reasoning_content="".join(
            part.get("text") or "" for part in parts if part.get("thought")
        )
        or None,
        tool_calls=calls or None,
    )


def response_from_wire(response: GenerateContentResponse) -> Response:
    data = response.model_dump()
    candidate = next(
        (item for item in data.get("candidates") or [] if item.get("index", 0) == 0),
        {},
    )
    message = parse_assistant((candidate.get("content") or {}).get("parts") or [])
    finish: FinishReason = (
        "tool_calls"
        if message.tool_calls
        else FINISH_REASONS.get(str(candidate.get("finishReason", "")))
    )
    metadata = data.get("usageMetadata")
    usage = None
    if metadata:
        usage = Usage(
            prompt_tokens=metadata.get("promptTokenCount") or 0,
            completion_tokens=(metadata.get("candidatesTokenCount") or 0)
            + (metadata.get("thoughtsTokenCount") or 0),
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
    """The non-streaming Gemini Developer API GenerateContent wire format."""

    routes = ("/v1beta/models/{model}:generateContent",)
    upstream_path = "/v1beta/models/{model}:generateContent"
    response_type = GenerateContentResponse

    def auth_headers(self, api_key: str) -> dict[str, str]:
        return {"x-goog-api-key": api_key}

    def secret(self, headers: Mapping[str, str]) -> str:
        return headers.get("x-goog-api-key", "")

    def streaming(self, body: dict) -> bool:
        return False

    def error_body(self, message: str) -> dict:
        return {
            "error": {
                "code": 400,
                "message": message,
                "status": "INVALID_ARGUMENT",
            }
        }

    def parse_request(self, body: dict) -> tuple[Messages, list[Tool] | None]:
        prompt: Messages = []
        if system := body.get("systemInstruction"):
            prompt.append(SystemMessage(content=text(system.get("parts") or [])))
        for content in body.get("contents") or []:
            parts = content.get("parts") or []
            if content.get("role") == "model":
                prompt.append(parse_assistant(parts))
                continue
            prompt.extend(
                ToolMessage(
                    tool_call_id=result.get("id") or result.get("name", ""),
                    content=json.dumps(result.get("response") or {}),
                )
                for part in parts
                if (result := part.get("functionResponse"))
            )
            if user_text := text(parts):
                prompt.append(UserMessage(content=user_text))
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
        return prompt, tools

    def parse_response(self, response: GenerateContentResponse) -> Response:
        return response_from_wire(response)

    def parse_stream(self, raw: bytes) -> Response:
        raise NotImplementedError("Google GenerateContent streaming is not supported")

    def apply_overrides(self, body: dict, model: str, sampling: SamplingConfig) -> dict:
        config = dict(body.get("generationConfig") or {})
        for key in _SAMPLING_KEYS:
            config.pop(key, None)
        values = sampling.model_dump(exclude_none=True)
        for source, target in (
            ("top_p", "topP"),
            ("top_k", "topK"),
            ("max_tokens", "maxOutputTokens"),
        ):
            if source in values:
                values[target] = values.pop(source)
        effort = values.pop("reasoning_effort", None)
        config.update(
            {key: value for key, value in values.items() if key in _GENERATION_KEYS}
        )
        gemini_25 = model.rsplit("/", 1)[-1].lower().startswith("gemini-2.5-")
        if gemini_25 and effort not in GEMINI_25_THINKING_BUDGETS:
            effort = None
        if effort:
            thinking = dict(config.get("thinkingConfig") or {})
            if gemini_25:
                thinking.pop("thinkingLevel", None)
                thinking["thinkingBudget"] = GEMINI_25_THINKING_BUDGETS[effort]
            else:
                thinking.pop("thinkingBudget", None)
                thinking["thinkingLevel"] = effort
            config["thinkingConfig"] = thinking
        return {**body, "generationConfig": config}
