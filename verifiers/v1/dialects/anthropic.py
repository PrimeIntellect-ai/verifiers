"""The Anthropic Messages dialect (claude-code and friends).

Request parsing maps Anthropic content blocks onto the typed messages; response parsing reads
the content blocks of a `Message`. Relay-only: the eval client forwards the program's native JSON to a
`/v1/messages` endpoint (auth is `x-api-key`, not Bearer) and this dialect parses a copy for the
trace. `count_tokens` is relayed as native JSON (an `aux_route`), never recorded.
"""

import json
import re
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import cast

from anthropic.types import Message as AnthropicMessage
from anthropic.types import MessageCreateParams
from anthropic.types import Usage as AnthropicUsage

from verifiers.v1.configs.runtime import NetworkPolicyConfig
from verifiers.v1.dialects.base import (
    Dialect,
    StreamParser,
    append_user_notice,
    blocked_url,
    parse_sse_event,
    provider_allowed_domains,
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
THINKING = ("redacted_thinking", "thinking")
# These versioned tool families return calls to the harness; every other typed tool may execute
# at the provider. Anchoring the pattern keeps new versions client-side without treating an
# arbitrary dated provider tool as safe.
_CLIENT_TOOL_TYPE = re.compile(r"(?:bash|text_editor|computer|memory)_\d{8}").fullmatch
_WEB_TOOL_TYPE = re.compile(r"web_(?:search|fetch)_\d{8}").fullmatch
_CONTENT_WRAPPERS = (
    "tool_result",
    "code_execution_tool_result",
    "bash_code_execution_tool_result",
    "text_editor_code_execution_tool_result",
    "web_search_tool_result",
    "web_fetch_tool_result",
    "tool_search_tool_result",
    "mcp_tool_result",
    "advisor_tool_result",
    "code_execution_result",
    "bash_code_execution_result",
    "encrypted_code_execution_result",
    "web_fetch_result",
)
_SAFE_CONTENT_TYPES = (
    "text",
    "image",
    "document",
    "tool_reference",
    "thinking",
    "redacted_thinking",
    "tool_use",
    "search_result",
    "server_tool_use",
    "mid_conv_system",
    "compaction",
    "fallback",
    "mcp_tool_use",
    "web_search_result",
    "web_search_tool_result_error",
    "web_fetch_tool_result_error",
    "tool_search_tool_search_result",
    "tool_search_tool_result_error",
    "code_execution_tool_result_error",
    "bash_code_execution_tool_result_error",
    "text_editor_code_execution_tool_result_error",
    "text_editor_code_execution_create_result",
    "text_editor_code_execution_str_replace_result",
    "text_editor_code_execution_view_result",
    "advisor_result",
    "advisor_redacted_result",
    "advisor_tool_result_error",
)


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


def blocked_content_path(value, path: str, policy: NetworkPolicyConfig) -> str | None:
    if isinstance(value, list):
        for index, item in enumerate(value):
            if blocked := blocked_content_path(item, f"{path}[{index}]", policy):
                return blocked
        return None
    if not isinstance(value, dict):
        return None

    kind = value.get("type")
    caller = value.get("caller")
    if caller is not None and not (
        isinstance(caller, dict) and caller.get("type") == "direct"
    ):
        return f"{path}.caller.type"
    if kind in ("image", "document"):
        source_path = f"{path}.source"
        source = value.get("source") or {}
        if not isinstance(source, dict):
            return source_path
        source_kind = source.get("type")
        if source_kind == "content":
            return blocked_content_path(
                source.get("content"), f"{source_path}.content", policy
            )
        if source_kind == "url":
            url = source.get("url")
            if not isinstance(url, str) or blocked_url(url, policy):
                return f"{source_path}.url"
        if source_kind == "file":
            return (
                f"{source_path}.file_id"
                if source.get("file_id")
                else f"{source_path}.type"
            )
        if source_kind not in ("base64", "text", "url"):
            return f"{source_path}.type"

    if kind in (
        "container_upload",
        "code_execution_output",
        "bash_code_execution_output",
    ) and value.get("file_id"):
        return f"{path}.file_id"
    if kind in _CONTENT_WRAPPERS:
        return blocked_content_path(value.get("content"), f"{path}.content", policy)
    return None if kind in _SAFE_CONTENT_TYPES else f"{path}.type"


def mediate_content(value, path: str, policy: NetworkPolicyConfig):
    if not isinstance(value, list):
        blocked = blocked_content_path(value, path, policy)
        return ("", [blocked]) if blocked else (value, [])

    mediated = []
    capabilities = []
    for index, block in enumerate(value):
        item_path = f"{path}[{index}]"
        if isinstance(block, dict) and block.get("type") in _CONTENT_WRAPPERS:
            if blocked := blocked_content_path(
                {**block, "content": []}, item_path, policy
            ):
                capabilities.append(blocked)
                continue
            content, removed = mediate_content(
                block.get("content"), f"{item_path}.content", policy
            )
            if removed:
                block["content"] = content or ""
                capabilities.extend(removed)
        elif blocked := blocked_content_path(block, item_path, policy):
            capabilities.append(blocked)
            continue
        mediated.append(block)
    return mediated, capabilities


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
            state = [block for block in blocks if block["type"] in THINKING]
            state.sort(key=lambda block: THINKING.index(block["type"]))
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
    state: list[dict] = []
    content: list[str] = []
    reasoning: list[str] = []
    calls: list[ToolCall] = []
    for block in message.content:
        if block.type in THINKING:
            state.append(block.model_dump())
        if block.type == "text":
            content.append(block.text)
        elif block.type == "thinking":
            reasoning.append(block.thinking)
        elif block.type == "tool_use":
            calls.append(
                ToolCall(
                    id=block.id,
                    name=block.name,
                    arguments=json.dumps(block.input or {}),
                )
            )
    state.sort(key=lambda block: THINKING.index(block["type"]))
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
        message=AssistantMessage(
            content="".join(content) or None,
            reasoning_content="".join(reasoning) or None,
            tool_calls=calls or None,
            provider_state=state or None,
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


class AnthropicDialect(Dialect[MessageCreateParams, AnthropicMessage]):
    sampling_fields = frozenset(
        {
            "temperature",
            "top_p",
            "top_k",
            "max_tokens",
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

    def mediate_external_capabilities(
        self, body: MessageCreateParams, policy: NetworkPolicyConfig
    ) -> tuple[MessageCreateParams, list[str]]:
        mediated = body
        capabilities: list[str] = []

        for key in ("container", "mcp_servers"):
            if mediated.pop(key, None):
                capabilities.append(key)

        system, removed = mediate_content(mediated.get("system"), "system", policy)
        capabilities.extend(removed)
        if removed:
            if system:
                mediated["system"] = system
            else:
                mediated.pop("system")

        for message_index, message in enumerate(mediated.get("messages") or []):
            if not isinstance(message, dict):
                continue
            content, removed = mediate_content(
                message.get("content"),
                f"messages[{message_index}].content",
                policy,
            )
            capabilities.extend(removed)
            if removed:
                message["content"] = content or ""

        raw_tools = mediated.get("tools")
        tool_items = raw_tools if isinstance(raw_tools, list) else []
        if raw_tools is not None and not isinstance(raw_tools, list):
            capabilities.append("tools")
        tools = []
        for index, tool in enumerate(tool_items):
            kind = tool.get("type") if isinstance(tool, dict) else None
            if (
                isinstance(tool, dict)
                and isinstance(kind, str)
                and _WEB_TOOL_TYPE(kind)
            ):
                callers = tool.get("allowed_callers")
                domains = (
                    provider_allowed_domains(policy, tool.get("allowed_domains"))
                    if tool.get("blocked_domains") in (None, [])
                    and (
                        callers is None
                        or isinstance(callers, list)
                        and "direct" in callers
                    )
                    else []
                )
                if domains:
                    web_tool = {
                        **tool,
                        "allowed_domains": domains,
                        "allowed_callers": ["direct"],
                    }
                    web_tool.pop("blocked_domains", None)
                    tools.append(web_tool)
                    continue
                capabilities.append(f"tools[{index}].type")
                continue
            if isinstance(tool, dict) and (
                (isinstance(kind, str) and _CLIENT_TOOL_TYPE(kind))
                or (kind in (None, "custom") and "input_schema" in tool)
            ):
                tools.append(tool)
                continue
            capabilities.append(f"tools[{index}].type")
        if "tools" in mediated:
            mediated["tools"] = tools

        choice = mediated.get("tool_choice")
        valid_choice = choice is None
        if isinstance(choice, dict):
            kind = choice.get("type")
            valid_choice = (
                kind == "none"
                or bool(tools)
                and (
                    kind in ("auto", "any")
                    or kind == "tool"
                    and any(tool.get("name") == choice.get("name") for tool in tools)
                )
            )
        if not valid_choice:
            capabilities.append(
                "tool_choice.type" if isinstance(choice, dict) else "tool_choice"
            )
            mediated.pop("tool_choice", None)

        append_user_notice(mediated.setdefault("messages", []))
        return mediated, capabilities

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

    def parse_request(self, body: MessageCreateParams) -> Request:
        tools = [
            Tool(
                name=t["name"],
                description=t.get("description", ""),
                parameters=t.get("input_schema", {}),
            )
            for t in body.get("tools") or []
            if "input_schema" in t  # skip server tools (web_search etc.)
        ] or None
        return Request(messages=parse_messages(body), tools=tools)

    def parse_response(self, response: AnthropicMessage) -> Response:
        return response_from_wire(response)

    def rewrite_request(self, body: dict, before: Request, after: Request) -> None:
        original = [
            m for m in before.messages if isinstance(m, (UserMessage, ToolMessage))
        ]
        rewritten = [
            m for m in after.messages if isinstance(m, (UserMessage, ToolMessage))
        ]
        targets: list[tuple[dict, dict | None]] = []
        for native in body.get("messages", []):
            if native.get("role") == "assistant":
                continue
            content = native.get("content")
            if isinstance(content, str):
                targets.append((native, None))
                continue
            blocks = content or []
            targets.extend(
                (native, block)
                for block in blocks
                if block.get("type") == "tool_result"
            )
            if any(block.get("type") != "tool_result" for block in blocks):
                targets.append((native, None))

        for (native, block), old, new in zip(targets, original, rewritten, strict=True):
            if old == new:
                continue
            if block is not None:
                block["content"] = content_to_wire(new.content)
                continue
            replacement = content_to_wire(new.content)
            if isinstance(native.get("content"), str):
                native["content"] = replacement
                continue
            replacement = (
                [{"type": "text", "text": replacement}]
                if isinstance(replacement, str)
                else replacement
            )
            blocks = native.get("content") or []
            updated = []
            inserted = False
            for current in blocks:
                if current.get("type") == "tool_result":
                    updated.append(current)
                elif not inserted:
                    updated.extend(replacement)
                    inserted = True
            native["content"] = updated

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

    def parse_sampling(self, body: MessageCreateParams) -> Sampling:
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
        self, body: MessageCreateParams, model: str, sampling: SamplingConfig
    ) -> MessageCreateParams:
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
        return cast(MessageCreateParams, {**steered, **overrides})
