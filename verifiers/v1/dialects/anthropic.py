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
from anthropic.types import Usage as AnthropicUsage

from verifiers.v1.dialects.base import (
    Dialect,
    RawRequest,
    StreamParser,
    parse_sse_event,
    with_provider_identity,
)
from verifiers.v1.dialects.request import (
    ContentTarget,
    NativeRequest,
    provider_declaration,
)
from verifiers.v1.types import (
    AssistantMessage,
    ContentPart,
    FinishReason,
    ImageUrlContentPart,
    ImageUrlSource,
    Messages,
    Request,
    Response,
    Sampling,
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
STOP_REASONS: dict[str, FinishReason] = {
    "end_turn": "stop",
    "max_tokens": "length",
    "tool_use": "tool_calls",
    "stop_sequence": "stop",
}
# Claude may reorder mixed thinking block types between a response and its replay.
# Native tool events share the final rank, preserving their relative order.
THINKING_ORDER = {"redacted_thinking": 0, "thinking": 1}


def parse_content(content) -> str | list[ContentPart]:
    """Anthropic user-side content (text + image blocks) -> typed content parts."""
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


def content_to_wire(content) -> str | list[dict]:
    """Typed text/image content in Anthropic's native request shape."""
    if isinstance(content, str):
        return content
    blocks = []
    for part in content:
        if isinstance(part, TextContentPart):
            blocks.append({"type": "text", "text": part.text})
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


def parse_messages(
    body: dict, targets: dict[int, ContentTarget] | None = None
) -> Messages:
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
                with_provider_identity(
                    AssistantMessage(
                        content=text or None,
                        reasoning_content=reasoning or None,
                        tool_calls=calls or None,
                        provider_state=state or None,
                    )
                )
            )
            continue
        rest = []
        for block in [] if isinstance(content, str) else content or []:
            if block.get("type") == "tool_result":
                if targets is not None:
                    targets[len(prompt)] = ContentTarget(
                        block, "content", content_to_wire, decode=parse_content
                    )
                prompt.append(
                    ToolMessage(
                        tool_call_id=block.get("tool_use_id", ""),
                        content=parse_content(block.get("content")),
                    )
                )
            else:
                rest.append(block)
        if isinstance(content, str) or rest:
            if targets is not None:
                targets[len(prompt)] = ContentTarget(
                    message,
                    "content",
                    # An empty block list would erase this projected user message.
                    lambda content: content_to_wire(content) or "",
                    preserve_blocks=("tool_result",)
                    if isinstance(content, list)
                    else (),
                    decode=parse_content,
                )
            prompt.append(
                UserMessage(
                    content=content if isinstance(content, str) else parse_content(rest)
                )
            )
    return prompt


def response_from_wire(message: AnthropicMessage) -> Response:
    """An Anthropic `Message` -> a vf `Response` (its content blocks folded into one assistant
    message: text -> content, thinking -> reasoning, tool_use -> tool calls)."""
    state: list[dict] = []
    content: list[str] = []
    reasoning: list[str] = []
    calls: list[ToolCall] = []
    for block in message.content:
        if block.type not in ("text", "tool_use"):
            # SDK-inserted defaults are absent when the native response is replayed.
            state.append(block.model_dump(exclude_unset=True))
        if block.type == "text":
            content.append(block.text)
        elif block.type == "thinking":
            reasoning.append(block.thinking)
        elif block.type == "tool_use":
            calls.append(
                ToolCall(
                    id=block.id,
                    name=block.name,
                    namespace=block.toolset_name,
                    arguments=json.dumps(block.input or {}),
                )
            )
    state.sort(key=lambda block: THINKING_ORDER.get(block["type"], 2))
    finish = STOP_REASONS.get(message.stop_reason or "")
    provider_usage = message.usage
    output_details = provider_usage.model_dump().get("output_tokens_details")
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
        id=message.id,
        created=0,
        model=message.model,
        message=with_provider_identity(
            AssistantMessage(
                content="".join(content) or None,
                reasoning_content="".join(reasoning) or None,
                tool_calls=calls or None,
                provider_state=state or None,
            )
        ),
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
                parts = self.block_parts.setdefault(index, {}).setdefault(
                    field_name, [block.get(field_name, "")]
                )
                parts.append(delta.get(field_name, ""))
            elif delta_type == "input_json_delta":
                self.partial_json.setdefault(index, []).append(
                    delta.get("partial_json", "")
                )
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


class ModdedUsage(AnthropicUsage):
    """The SDK closes `service_tier` to a fixed Literal, but Anthropic-compatible gateways
    report their own tiers (e.g. Prime's `provisioned`). Widen to a plain string — we don't
    consume it — so parsing stays lenient about the label instead of dropping it."""

    service_tier: str | None = None  # type: ignore[assignment]


class ModdedAnthropicMessage(AnthropicMessage):
    usage: ModdedUsage  # type: ignore[assignment]


class AnthropicDialect(Dialect[AnthropicMessage]):
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
    routes = ("/v1/messages",)
    aux_routes = ("/v1/messages/count_tokens",)
    upstream_path = "/v1/messages"
    response_type = ModdedAnthropicMessage

    def is_terminal_event(self, chunk: bytes) -> bool:
        return any(
            line.removeprefix(b"event:").strip() == b"message_stop"
            for line in chunk.splitlines()
        )

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

    def bind_request(self, body: RawRequest) -> NativeRequest:
        declarations = body.get("tools") or []
        if not isinstance(declarations, list) or any(
            not isinstance(t, dict) for t in declarations
        ):
            raise ValueError("tools must be an array of objects")
        tools, provider_tools = [], []
        for tool in declarations:
            if tool.get("type") not in (None, "custom"):
                provider_tools.append(provider_declaration(tool))
                continue
            tools.append(
                Tool.model_validate(
                    {k: v for k, v in tool.items() if k != "input_schema"}
                    | {"type": "function", "parameters": tool.get("input_schema") or {}}
                )
            )
        targets = {}
        messages = parse_messages(body, targets)
        return NativeRequest(
            body,
            Request(messages=messages, tools=tools or None),
            targets,
            provider_tools,
        )

    def parse_response(self, response: AnthropicMessage) -> Response:
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

    def stream_parser(self) -> StreamParser:
        return AnthropicStreamParser(self.validate_response)

    def parse_sampling(self, body: RawRequest) -> Sampling:
        settings = {k: v for k, v in body.items() if k in self.sampling_fields}
        # Lift `output_config.effort` (where `apply_overrides` puts the eval's
        # reasoning effort) onto the typed knob; keep any other output-config keys.
        if isinstance(config := settings.get("output_config"), dict):
            config = dict(config)
            if config.get("effort"):
                settings["reasoning_effort"] = config.pop("effort")
            if config:
                settings["output_config"] = config
            else:
                settings.pop("output_config")
        return Sampling.model_validate(settings)

    def apply_overrides(
        self, body: RawRequest, model: str, sampling: SamplingConfig
    ) -> RawRequest:
        # Preserve native fields except the eval's model + sampling. `temperature`/`top_p` are
        # authoritative (always dropped, the eval's applied if set); `max_tokens` is required by
        # the API, so the program's is kept unless the eval sets one.
        s = sampling.wire_args()
        reasoning_effort = s.pop("reasoning_effort", None)
        sampling_output_config = s.pop("output_config", None)
        overrides: dict = {**s, "model": model}
        if sampling_output_config is not None or reasoning_effort is not None:
            overrides["output_config"] = {
                **dict(body.get("output_config") or {}),
                **dict(sampling_output_config or {}),
            }
            if reasoning_effort is not None:
                overrides["output_config"]["effort"] = reasoning_effort
        steered = {
            k: v
            for k, v in body.items()
            if k not in ("temperature", "top_p") and k not in overrides
        }
        return {**steered, **overrides}
