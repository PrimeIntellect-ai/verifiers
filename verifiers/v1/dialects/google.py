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
_SAMPLING_KEYS = {"temperature", "topP", "topK", "maxOutputTokens"}
_SAMPLING_ALIASES = {
    "top_p": "topP",
    "top_k": "topK",
    "max_tokens": "maxOutputTokens",
}
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
_GEMINI_25_THINKING_BUDGETS = {
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
    return "".join(part.get("text") or "" for part in parts if not part.get("thought"))


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
            user_text = ""
            for part in parts:
                if result := part.get("functionResponse"):
                    if user_text:
                        prompt.append(UserMessage(content=user_text))
                        user_text = ""
                    prompt.append(
                        ToolMessage(
                            tool_call_id=result.get("id") or result.get("name", ""),
                            content=json.dumps(result.get("response") or {}),
                        )
                    )
                elif not part.get("thought"):
                    user_text += part.get("text") or ""
            if user_text:
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
        values = sampling.model_dump(exclude_none=True)
        for source, target in _SAMPLING_ALIASES.items():
            if source in values:
                values[target] = values[source]
        config = {
            key: value
            for key, value in (body.get("generationConfig") or {}).items()
            if key not in _SAMPLING_KEYS
        }
        thinking = dict(config.pop("thinkingConfig", {}) or {})
        thinking.pop("thinkingBudget", None)
        thinking.pop("thinkingLevel", None)
        if thinking:
            config["thinkingConfig"] = thinking
        config.update({key: values[key] for key in _GENERATION_KEYS if key in values})

        effort = values.get("reasoning_effort")
        model = model.rsplit("/", 1)[-1].lower()
        if model.startswith("gemini-2.5-"):
            effort = _GEMINI_25_THINKING_BUDGETS.get(effort)
            thinking_key = "thinkingBudget"
            stale_key = "thinkingLevel"
        elif model.startswith("gemini-3"):
            thinking_key = "thinkingLevel"
            stale_key = "thinkingBudget"
        else:
            effort = None
        if effort is not None:
            thinking = dict(config.get("thinkingConfig") or {})
            thinking.pop(stale_key, None)
            thinking[thinking_key] = effort
            config["thinkingConfig"] = thinking
        return {**body, "generationConfig": config}
