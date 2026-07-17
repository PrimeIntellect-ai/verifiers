"""The Anthropic Messages dialect (claude-code and friends).

Request parsing maps Anthropic content blocks onto the typed messages; response parsing reads
the content blocks of a `Message`. Relay-only: the eval client forwards the program's native JSON to a
`/v1/messages` endpoint (auth is `x-api-key`, not Bearer) and this dialect parses a copy for the
trace. `count_tokens` is relayed as native JSON (an `aux_route`), never recorded.
"""

import json
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field

from anthropic.types import Message as AnthropicMessage

from verifiers.v1.dialects.base import (
    Dialect,
    StreamParser,
    parse_sse_event,
    sse_event,
)
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
            state = [
                {"type": "text", "citations": block["citations"]}
                if block.get("type") == "text"
                else block
                for block in blocks
                if block.get("type") not in ("text", "tool_use")
                or block.get("citations")
            ]
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
                    provider_state=state or None,
                )
            )
            continue
        rest = []
        for block in [] if isinstance(content, str) else content or []:
            if block.get("type") == "tool_result":
                tool_id = block.get("tool_use_id", "")
                prompt.append(
                    ToolMessage(
                        tool_call_id=tool_id,
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
    message: text -> content, thinking -> reasoning, tool_use -> tool calls).

    `exclude_unset` reproduces the wire body exactly — absent fields stay absent, explicit
    nulls stay null — so the committed `provider_state` hashes like the harness's verbatim
    replay of the same blocks (see `graph.message_hash`)."""
    data = message.model_dump(exclude_unset=True)
    blocks = data.get("content") or []
    assistant = parse_messages(
        {"messages": [{"role": "assistant", "content": blocks}]}
    )[0]
    assert isinstance(assistant, AssistantMessage)
    finish: FinishReason = STOP_REASONS.get(data.get("stop_reason") or "")
    provider_usage = message.usage
    output_details = data.get("usage", {}).get("output_tokens_details")
    # Anthropic reports three disjoint input buckets. Cache writes are uncached work;
    # cache reads are the reusable subset exposed separately by vf.Usage.
    usage = Usage(
        prompt_tokens=provider_usage.input_tokens
        + (provider_usage.cache_creation_input_tokens or 0),
        completion_tokens=provider_usage.output_tokens,
        cached_input_tokens=provider_usage.cache_read_input_tokens,
        # This is a re-tokenized raw-thinking estimate inside output_tokens, not the
        # token count of the visible thinking summary.
        reasoning_tokens=output_details.get("thinking_tokens")
        if output_details
        else None,
        cost=getattr(provider_usage, "cost", None),
    )
    return Response(
        id=data.get("id", ""),
        created=0,
        model=data.get("model", ""),
        message=assistant,
        finish_reason=finish,
        usage=usage,
    )


@dataclass
class AnthropicStreamParser(StreamParser):
    """Incrementally assemble Anthropic message events without retaining SSE bytes."""

    validate_response: Callable[[dict], AnthropicMessage]
    message: dict = field(default_factory=dict)
    blocks: dict[int, dict] = field(default_factory=dict)
    block_parts: dict[int, dict[str, list[str]]] = field(default_factory=dict)
    partial_json: dict[int, list[str]] = field(default_factory=dict)

    def feed(self, raw: bytes) -> None:
        event = parse_sse_event(raw)
        if event is None:
            return
        kind = event.get("type")
        if kind == "message_start":
            self.message = event.get("message") or {}
        elif kind == "content_block_start":
            index = event["index"]
            self.blocks[index] = dict(event.get("content_block") or {})
            self.block_parts.pop(index, None)
        elif kind == "content_block_delta":
            index = event["index"]
            block = self.blocks.setdefault(index, {"type": "text", "text": ""})
            delta = event.get("delta") or {}
            delta_type = delta.get("type")
            if delta_type in (
                "text_delta",
                "thinking_delta",
                "signature_delta",
            ):
                field_name = delta_type.removesuffix("_delta")
                fields = self.block_parts.get(index)
                if fields is None:
                    fields = {}
                    self.block_parts[index] = fields
                parts = fields.get(field_name)
                if parts is None:
                    parts = [block.get(field_name, "")]
                    fields[field_name] = parts
                parts.append(delta.get(field_name, ""))
            elif delta_type == "input_json_delta":
                parts = self.partial_json.get(index)
                if parts is None:
                    parts = []
                    self.partial_json[index] = parts
                parts.append(delta.get("partial_json", ""))
            elif delta_type == "citations_delta":
                citations = block.get("citations") or []
                block["citations"] = [*citations, delta.get("citation") or {}]
        elif kind == "message_delta":
            self.message.update(
                {
                    key: value
                    for key, value in (event.get("delta") or {}).items()
                    if value is not None
                }
            )
            self.message["usage"] = {
                **(self.message.get("usage") or {}),
                **(event.get("usage") or {}),
            }

    def finish(self) -> Response:
        for index, fields in self.block_parts.items():
            for field_name, parts in fields.items():
                self.blocks[index][field_name] = "".join(parts)
        for index, parts in self.partial_json.items():
            self.blocks[index]["input"] = json.loads("".join(parts) or "{}")
        self.message["content"] = [self.blocks[index] for index in sorted(self.blocks)]
        response = response_from_wire(self.validate_response(self.message))
        response.raw = self.message
        return response


class AnthropicDialect(Dialect[dict, AnthropicMessage]):
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

    def rewrite_response(self, raw: dict, content: str) -> dict:
        raw["content"] = [{"type": "text", "text": content}]
        if raw.get("stop_reason") != "max_tokens":
            raw["stop_reason"] = "end_turn"
        raw["stop_sequence"] = None
        return raw

    def serialize_stream(self, raw: dict) -> list[bytes]:
        head = {**raw, "content": [], "stop_reason": None, "stop_sequence": None}
        if usage := head.get("usage"):
            head["usage"] = {**usage, "output_tokens": 0}
        item = raw["content"][0]
        events = [
            ("message_start", {"message": head}),
            (
                "content_block_start",
                {"index": 0, "content_block": {**item, "text": ""}},
            ),
            (
                "content_block_delta",
                {"index": 0, "delta": {"type": "text_delta", "text": item["text"]}},
            ),
            ("content_block_stop", {"index": 0}),
            (
                "message_delta",
                {
                    "delta": {
                        "stop_reason": raw.get("stop_reason"),
                        "stop_sequence": raw.get("stop_sequence"),
                    },
                    "usage": raw.get("usage") or {},
                },
            ),
            ("message_stop", {}),
        ]
        return [sse_event({"type": kind, **payload}, kind) for kind, payload in events]

    def rewrite_tool_results(self, body: dict, replacements: dict[str, str]) -> dict:
        for raw_message in body.get("messages", []):
            blocks = raw_message.get("content")
            if not isinstance(blocks, list):
                continue
            for block in blocks:
                tool_id = block.get("tool_use_id")
                if block.get("type") != "tool_result" or tool_id not in replacements:
                    continue
                block["content"] = replacements[tool_id]
        return body

    def is_terminal_event(self, chunk: bytes) -> bool:
        return b'"type":"message_stop"' in chunk or b'"type": "message_stop"' in chunk

    def validate_response(self, raw: dict) -> AnthropicMessage:
        usage = raw.get("usage")
        tier = usage.get("service_tier") if usage else None
        if tier not in (None, "standard", "priority", "batch"):
            raw = {**raw, "usage": usage.copy()}
            raw["usage"].pop("service_tier")
        return super().validate_response(raw)

    def stream_parser(self) -> StreamParser:
        return AnthropicStreamParser(self.validate_response)

    def apply_overrides(self, body: dict, model: str, sampling: SamplingConfig) -> dict:
        # Preserve native fields except the eval's model + sampling. `temperature`/`top_p` are
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
        if "reasoning_effort" in s:
            overrides["output_config"] = {
                **dict(body.get("output_config") or {}),
                "effort": s["reasoning_effort"],
            }
        steered = {
            k: v
            for k, v in body.items()
            if k not in ("temperature", "top_p") and k not in overrides
        }
        return {**steered, **overrides}
