"""The Anthropic Messages dialect (claude-code and friends).

Request parsing maps Anthropic content blocks onto the typed messages; response parsing reads
the content blocks of a `Message`. Relay-only: the eval client forwards the program's bytes to a
`/v1/messages` endpoint (auth is `x-api-key`, not Bearer) and this dialect parses a copy for the
trace. `count_tokens` is relayed verbatim (an `aux_route`), never recorded.
"""

import json
from collections.abc import Mapping

from anthropic.types import Message as AnthropicMessage

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

# Anthropic stop_reason -> vf finish_reason.
STOP_REASONS = {
    "end_turn": "stop",
    "max_tokens": "length",
    "tool_use": "tool_calls",
    "stop_sequence": "stop",
}


def parse_content(content) -> str | list[ContentPart]:
    """Anthropic user-side content (text + image blocks) -> typed content parts."""
    if isinstance(content, str):
        return content
    parts: list[ContentPart] = []
    for block in content or []:
        if block.get("type") == "text":
            parts.append(TextContentPart(text=block.get("text", "")))
        elif block.get("type") == "image":
            source = block.get("source") or {}
            if source.get("type") == "url":
                url = source.get("url", "")
            else:
                url = f"data:{source.get('media_type', '')};base64,{source.get('data', '')}"
            parts.append(ImageUrlContentPart(image_url=ImageUrlSource(url=url)))
    return parts


def parse_messages(body: dict) -> Messages:
    """The request's top-level `system` + `messages` -> typed messages. Assistant turns fold
    their blocks into one message (thinking -> reasoning, tool_use -> tool calls); a user turn's
    tool_result blocks become individual tool messages, its rest one user message."""
    prompt: Messages = []
    if system := body.get("system"):
        prompt.append(SystemMessage(content=parse_content(system)))
    for message in body.get("messages", []):
        content = message.get("content")
        if message.get("role") == "assistant":
            blocks = (
                [{"type": "text", "text": content}]
                if isinstance(content, str)
                else content or []
            )
            text = "".join(b.get("text", "") for b in blocks if b.get("type") == "text")
            reasoning = "".join(
                b.get("thinking", "") for b in blocks if b.get("type") == "thinking"
            )
            calls = [
                ToolCall(
                    id=b.get("id", ""),
                    name=b.get("name", ""),
                    arguments=json.dumps(b.get("input") or {}),
                )
                for b in blocks
                if b.get("type") == "tool_use"
            ]
            prompt.append(
                AssistantMessage(
                    content=text or None,
                    reasoning_content=reasoning or None,
                    tool_calls=calls or None,
                )
            )
            continue
        rest = []
        for block in [] if isinstance(content, str) else content or []:
            if block.get("type") == "tool_result":
                prompt.append(
                    ToolMessage(
                        tool_call_id=block.get("tool_use_id", ""),
                        content=parse_content(block.get("content")),
                    )
                )
            else:
                rest.append(block)
        if isinstance(content, str) or rest:
            prompt.append(
                UserMessage(
                    content=content if isinstance(content, str) else parse_content(rest)
                )
            )
    return prompt


def response_from_wire(message: AnthropicMessage) -> Response:
    """An Anthropic `Message` -> a vf `Response` (its content blocks folded into one assistant
    message: text -> content, thinking -> reasoning, tool_use -> tool calls)."""
    data = message.model_dump()
    content = ""
    reasoning = ""
    calls: list[ToolCall] = []
    for block in data.get("content") or []:
        kind = block.get("type")
        if kind == "text":
            content += block.get("text", "")
        elif kind == "thinking":
            reasoning += block.get("thinking", "")
        elif kind == "tool_use":
            calls.append(
                ToolCall(
                    id=block.get("id", ""),
                    name=block.get("name", ""),
                    arguments=json.dumps(block.get("input") or {}),
                )
            )
    finish: FinishReason = STOP_REASONS.get(data.get("stop_reason") or "")
    usage = (
        Usage(
            prompt_tokens=data["usage"].get("input_tokens", 0),
            completion_tokens=data["usage"].get("output_tokens", 0),
        )
        if data.get("usage")
        else None
    )
    return Response(
        id=data.get("id", ""),
        created=0,
        model=data.get("model", ""),
        message=AssistantMessage(
            content=content or None,
            reasoning_content=reasoning or None,
            tool_calls=calls or None,
        ),
        finish_reason=finish,
        usage=usage,
    )


class AnthropicDialect(Dialect[dict, AnthropicMessage]):
    """The Anthropic Messages wire format."""

    routes = ("/v1/messages",)
    aux_routes = ("/v1/messages/count_tokens",)
    upstream_path = "/v1/messages"
    response_type = AnthropicMessage

    def auth_headers(self, api_key: str) -> dict[str, str]:
        return {"x-api-key": api_key, "anthropic-version": "2023-06-01"}

    def secret(self, headers: Mapping[str, str]) -> str:
        # The SDK sends the key as `x-api-key`; an ANTHROPIC_AUTH_TOKEN arrives as Bearer.
        return headers.get("x-api-key") or super().secret(headers)

    def error_body(self, message: str) -> dict:
        return {
            "type": "error",
            "error": {"type": "invalid_request_error", "message": message},
        }

    def parse_request(self, body: dict) -> tuple[Messages, list[Tool] | None]:
        tools = [
            Tool(
                name=t["name"],
                description=t.get("description", ""),
                parameters=t.get("input_schema", {}),
            )
            for t in body.get("tools") or []
            if "input_schema" in t  # skip server tools (web_search etc.)
        ] or None
        return parse_messages(body), tools

    def parse_response(self, response: AnthropicMessage) -> Response:
        return response_from_wire(response)

    def parse_stream(self, raw: bytes) -> Response:
        """Assemble message_start / content_block_* / message_delta events into the complete
        message, then parse it."""
        message: dict = {}
        blocks: dict[int, dict] = {}
        partial_json: dict[int, str] = {}
        for event in iter_sse(raw):
            kind = event.get("type")
            if kind == "message_start":
                message = event.get("message") or {}
            elif kind == "content_block_start":
                blocks[event["index"]] = dict(event.get("content_block") or {})
            elif kind == "content_block_delta":
                block = blocks.setdefault(event["index"], {"type": "text", "text": ""})
                delta = event.get("delta") or {}
                if delta.get("type") == "text_delta":
                    block["text"] = block.get("text", "") + delta.get("text", "")
                elif delta.get("type") == "thinking_delta":
                    block["thinking"] = block.get("thinking", "") + delta.get(
                        "thinking", ""
                    )
                elif delta.get("type") == "signature_delta":
                    block["signature"] = block.get("signature", "") + delta.get(
                        "signature", ""
                    )
                elif delta.get("type") == "input_json_delta":
                    partial_json[event["index"]] = partial_json.get(
                        event["index"], ""
                    ) + delta.get("partial_json", "")
            elif kind == "message_delta":
                message.update(
                    {
                        k: v
                        for k, v in (event.get("delta") or {}).items()
                        if v is not None
                    }
                )
                message["usage"] = {
                    **(message.get("usage") or {}),
                    **(event.get("usage") or {}),
                }
        for index, partial in partial_json.items():
            blocks[index]["input"] = json.loads(partial or "{}")
        message["content"] = [blocks[i] for i in sorted(blocks)]
        message.get("usage", {}).pop("service_tier", None)
        return response_from_wire(AnthropicMessage.model_validate(message))

    def apply_overrides(self, body: dict, model: str, sampling: SamplingConfig) -> dict:
        # Forward verbatim except the eval's model + sampling. `temperature`/`top_p` are
        # authoritative (always dropped, the eval's applied if set); `max_tokens` is required by
        # the API, so the program's is kept unless the eval sets one.
        s = sampling.model_dump(exclude_none=True)
        overrides: dict = {"model": model}
        if "temperature" in s:
            overrides["temperature"] = s["temperature"]
        if "top_p" in s:
            overrides["top_p"] = s["top_p"]
        if "max_tokens" in s:
            overrides["max_tokens"] = s["max_tokens"]
        steered = {
            k: v
            for k, v in body.items()
            if k not in ("temperature", "top_p") and k not in overrides
        }
        return {**steered, **overrides}
