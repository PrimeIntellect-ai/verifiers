"""The Anthropic Messages dialect (claude-code and friends).

Request parsing maps Anthropic content blocks onto the typed messages; response parsing reads
the content blocks of a `Message`. Relay-only: the eval client forwards the program's native JSON to a
`/v1/messages` endpoint (auth is `x-api-key`, not Bearer) and this dialect parses a copy for the
trace. `count_tokens` is relayed as native JSON (an `aux_route`), never recorded.
"""

import json
from collections.abc import Mapping
from functools import partial

from anthropic import not_given
from anthropic.lib.streaming import AsyncMessageStream
from anthropic.types import RawMessageStreamEvent

from verifiers.v1.dialects.base import Dialect, RawRequest, Setter, patch_content
from verifiers.v1.types import (
    AssistantMessage,
    ContentPart,
    FinishReason,
    ImageUrlContentPart,
    ImageUrlSource,
    Message,
    MessageContent,
    Messages,
    NativeContentPart,
    Request,
    Response,
    SystemMessage,
    TextContentPart,
    Tool,
    ToolCall,
    ToolMessage,
    Usage,
    UserMessage,
)

# Anthropic stop_reason -> vf finish_reason.
STOP_REASONS: dict[str, FinishReason] = {
    "end_turn": "stop",
    "max_tokens": "length",
    "tool_use": "tool_calls",
    "stop_sequence": "stop",
}
# Claude may reorder mixed thinking block types between a response and its replay.
# Native tool events share the final rank, preserving their relative order.
THINKING_ORDER = {"redacted_thinking": 0, "thinking": 1}


def parse_content(content) -> MessageContent:
    """Anthropic user-side content blocks -> typed content parts; blocks without a typed
    model (documents, search results, file references) stay native."""
    if isinstance(content, str):
        return content
    if (
        isinstance(content, list)
        and len(content) == 1
        and content[0].get("type") == "text"
    ):
        return content[0].get("text", "")
    parts: list[ContentPart] = []
    for block in content or []:
        kind = block.get("type")
        source = block.get("source") or {}
        if kind == "text":
            parts.append(TextContentPart(text=block.get("text", "")))
        elif kind == "image" and source.get("type") in ("url", "base64"):
            url = (
                source.get("url", "")
                if source["type"] == "url"
                else f"data:{source.get('media_type', '')};base64,{source.get('data', '')}"
            )
            parts.append(ImageUrlContentPart(image_url=ImageUrlSource(url=url)))
        else:
            parts.append(NativeContentPart(native=block))
        if not isinstance(parts[-1], NativeContentPart):
            parts[-1]._native = block
    return parts


def content_to_wire(content: MessageContent) -> str | list[dict]:
    """Typed content in Anthropic's native request shape."""
    if isinstance(content, str):
        return content
    blocks = []
    for part in content:
        if isinstance(part, TextContentPart):
            blocks.append({"type": "text", "text": part.text})
            continue
        if isinstance(part, NativeContentPart):
            blocks.append(part.native)
            continue
        metadata, separator, data = part.image_url.url.partition(",")
        if separator and metadata.startswith("data:") and metadata.endswith(";base64"):
            blocks.append(
                {
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": metadata[5:-7],
                        "data": data,
                    },
                }
            )
        else:
            blocks.append(
                {
                    "type": "image",
                    "source": {"type": "url", "url": part.image_url.url},
                }
            )
    return blocks


def _write_content(message: dict, edited: Message) -> None:
    message["content"] = patch_content(
        edited.content, content_to_wire(edited.content), message.get("content")
    )


def _write_rest(message: dict, edited: Message) -> None:
    """Replace a user turn's blocks besides its tool results, where the first of them sat."""
    replacement = patch_content(
        edited.content,
        content_to_wire(edited.content),
        [
            block
            for block in message.get("content") or []
            if block.get("type") != "tool_result"
        ],
    )
    if isinstance(replacement, str):
        replacement = [{"type": "text", "text": replacement}]
    updated = []
    inserted = False
    for block in message.get("content") or []:
        if block.get("type") == "tool_result":
            updated.append(block)
        elif not inserted:
            updated.extend(replacement)
            inserted = True
    message["content"] = updated


def parse_messages(body: dict) -> tuple[Messages, list[Setter | None]]:
    """The request's top-level `system` + `messages` -> typed messages and their setters.
    Assistant turns fold their blocks into one message (thinking -> reasoning, tool_use -> tool
    calls); a user turn's tool_result blocks become individual tool messages, its rest one user
    message."""
    prompt: Messages = []
    setters: list[Setter | None] = []
    if system := body.get("system"):
        prompt.append(SystemMessage(content=parse_content(system)))
        setters.append(None)
    for message in body.get("messages", []):
        content = message.get("content")
        if message.get("role") == "assistant":
            blocks = (
                [{"type": "text", "text": content}]
                if isinstance(content, str)
                else content or []
            )
            state = [
                block for block in blocks if block["type"] not in ("text", "tool_use")
            ]
            state.sort(key=lambda block: THINKING_ORDER.get(block["type"], 2))
            text = "".join(b.get("text", "") for b in blocks if b.get("type") == "text")
            reasoning = "".join(
                b.get("thinking", "") for b in blocks if b.get("type") == "thinking"
            )
            calls = [
                ToolCall(
                    id=b.get("id", ""),
                    name=b.get("name", ""),
                    namespace=b.get("toolset_name"),
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
            setters.append(None)
            continue
        if isinstance(content, str):
            prompt.append(UserMessage(content=content))
            setters.append(partial(_write_content, message))
            continue
        rest = []
        for block in content or []:
            if block.get("type") == "tool_result":
                prompt.append(
                    ToolMessage(
                        tool_call_id=block.get("tool_use_id", ""),
                        content=parse_content(block.get("content")),
                    )
                )
                setters.append(partial(_write_content, block))
            else:
                rest.append(block)
        if rest:
            prompt.append(UserMessage(content=parse_content(rest)))
            setters.append(partial(_write_rest, message))
    return prompt, setters


def response_from_wire(message: dict) -> Response:
    """An Anthropic `Message` -> a vf `Response` (its content blocks folded into one assistant
    message: text -> content, thinking -> reasoning, tool_use -> tool calls)."""
    state: list[dict] = []
    content: list[str] = []
    reasoning: list[str] = []
    calls: list[ToolCall] = []
    if not isinstance(message.get("content"), list):
        raise TypeError("Anthropic response requires a content array")
    for block in message["content"]:
        kind = block.get("type")
        if kind == "text":
            content.append(block.get("text", ""))
        elif kind == "tool_use":
            calls.append(
                ToolCall(
                    id=block.get("id", ""),
                    name=block.get("name", ""),
                    namespace=block.get("toolset_name"),
                    arguments=json.dumps(block.get("input") or {}),
                )
            )
        else:
            # Replayed verbatim on the next turn, so the native block is the state.
            state.append(block)
            if kind == "thinking":
                reasoning.append(block.get("thinking", ""))
    state.sort(key=lambda block: THINKING_ORDER.get(block.get("type"), 2))
    usage = None
    if provider_usage := message.get("usage"):
        # Anthropic reports three disjoint input buckets. Cache writes are uncached work;
        # cache reads are the reusable subset exposed separately by vf.Usage.
        usage = Usage(
            prompt_tokens=provider_usage["input_tokens"]
            + (provider_usage.get("cache_creation_input_tokens") or 0),
            completion_tokens=provider_usage["output_tokens"],
            cached_input_tokens=provider_usage.get("cache_read_input_tokens"),
            # This is a re-tokenized raw-thinking estimate inside output_tokens, not the
            # token count of the visible thinking summary.
            reasoning_tokens=(provider_usage.get("output_tokens_details") or {}).get(
                "thinking_tokens"
            ),
            cost=provider_usage.get("cost"),
        )
    return Response(
        id=message.get("id") or "",
        created=0,
        model=message.get("model") or "",
        message=AssistantMessage(
            content="".join(content) or None,
            reasoning_content="".join(reasoning) or None,
            tool_calls=calls or None,
            provider_state=state or None,
        ),
        finish_reason=STOP_REASONS.get(message.get("stop_reason") or ""),
        usage=usage,
    )


class AnthropicDialect(Dialect):
    sampling_fields = frozenset(
        {
            "temperature",
            "top_p",
            "top_k",
            "max_tokens",
            "service_tier",
            "stop_sequences",
            "thinking",
            "tool_choice",
            "output_config",
        }
    )
    max_tokens_keys = ("max_tokens",)
    effort_path = ("output_config", "effort")
    routes = ("/v1/messages",)
    aux_routes = ("/v1/messages/count_tokens",)
    upstream_path = "/v1/messages"
    event_type = RawMessageStreamEvent

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

    def stream_keepalive(self, first: bool) -> bytes:
        # Anthropic's own keepalive; its SDKs skip it anywhere in the stream.
        return b'event: ping\ndata: {"type": "ping"}\n\n'

    def stream_error(self, error: dict) -> bytes:
        # The Anthropic SDKs raise only on a named `error` event.
        return b"event: error\ndata: " + json.dumps(error).encode() + b"\n\n"

    def parse_request(self, body: RawRequest) -> tuple[Request, list[Setter | None]]:
        native_tools = body.get("tools") or []
        if not isinstance(native_tools, list) or any(
            not isinstance(tool, dict) for tool in native_tools
        ):
            raise ValueError("tools must be an array of objects")
        tools = [
            Tool.model_validate(
                {k: v for k, v in t.items() if k != "input_schema"}
                | {
                    "name": t.get("name") or t.get("mcp_server_name") or t.get("type"),
                    "type": "function"
                    if t.get("type") in (None, "custom")
                    else t["type"],
                    "parameters": t.get("input_schema") or {},
                }
            )
            for t in native_tools
        ] or None
        messages, setters = parse_messages(body)
        return Request(messages=messages, tools=tools), setters

    def parse_response(self, response: dict) -> Response:
        return response_from_wire(response)

    def rewrite_response(self, raw: dict, text: str) -> None:
        raw["content"] = [{"type": "text", "text": text}]
        raw["stop_reason"] = "end_turn"
        raw["stop_sequence"] = None

    def stream_events(self, raw: dict) -> list[bytes]:
        def event(kind: str, payload: dict) -> bytes:
            return f"event: {kind}\ndata: {json.dumps(payload)}\n\n".encode()

        text = raw["content"][0]["text"]
        head = {**raw, "content": [], "stop_reason": None, "stop_sequence": None}
        if isinstance(usage := head.get("usage"), dict):
            head["usage"] = {**usage, "output_tokens": 0}
        return [
            event("message_start", {"type": "message_start", "message": head}),
            event(
                "content_block_start",
                {
                    "type": "content_block_start",
                    "index": 0,
                    "content_block": {"type": "text", "text": ""},
                },
            ),
            event(
                "content_block_delta",
                {
                    "type": "content_block_delta",
                    "index": 0,
                    "delta": {"type": "text_delta", "text": text},
                },
            ),
            event("content_block_stop", {"type": "content_block_stop", "index": 0}),
            event(
                "message_delta",
                {
                    "type": "message_delta",
                    "delta": {"stop_reason": "end_turn", "stop_sequence": None},
                    "usage": raw.get("usage") or {},
                },
            ),
            event("message_stop", {"type": "message_stop"}),
        ]

    async def read_stream(self, stream) -> dict:
        async with AsyncMessageStream(stream, output_format=not_given) as messages:
            stopped = False
            async for event in messages:
                stopped |= event.type == "message_stop"
                if event.type == "message_delta":
                    # The SDK updates standard usage totals; provider extensions are totals too.
                    for name, value in (event.usage.model_extra or {}).items():
                        setattr(messages.current_message_snapshot.usage, name, value)
            if not stopped:
                raise ValueError("Anthropic stream ended without message_stop")
            return (await messages.get_final_message()).to_dict()
