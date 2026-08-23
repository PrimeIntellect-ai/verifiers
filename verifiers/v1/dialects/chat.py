"""The OpenAI chat-completions dialect.

Translates the OpenAI chat-completions wire format into vf types: requests (`parse_request`)
and responses (`parse_response`). Reasoning extraction mirrors the v0 chat client's
`parse_reasoning_content` — providers expose the model's reasoning under different keys, so
read them in the same precedence (`reasoning` / `reasoning_content` / `reasoning_details`).
"""

import json
import time
from collections.abc import Mapping
from dataclasses import dataclass
from dataclasses import field as dataclass_field
from typing import Any
from urllib.parse import urlsplit

from openai.types.chat import ChatCompletion

from verifiers.v1.configs.runtime import NetworkPolicyConfig
from verifiers.v1.dialects.base import (
    Dialect,
    RawRequest,
    StreamParser,
    append_user_notice,
    blocked_url,
    parse_sse_event,
)
from verifiers.v1.types import (
    AssistantMessage,
    FinishReason,
    Message,
    Messages,
    Request,
    Response,
    Sampling,
    SamplingConfig,
    SystemMessage,
    Tool,
    ToolCall,
    ToolMessage,
    Usage,
    UserMessage,
    content_to_parts,
)


class ModdedChatCompletion(ChatCompletion):
    """The OpenAI SDK closes `service_tier` to a fixed `Literal`, but providers return tiers
    outside it (e.g. Prime's `provisioned`), which makes `model_validate` reject an otherwise
    valid completion. Widen the field to a plain string — we don't consume it — so parsing stays
    lenient about the label instead of dropping it."""

    service_tier: str | None = None


FINISH_REASONS = frozenset({"stop", "length", "tool_calls"})
# Client tools return calls to the harness; every other type may execute at the provider.
_CLIENT_TOOL_TYPES = ("function", "custom")
_SAFE_CONTENT_TYPES = ("text", "refusal", "input_audio", "image_url", "file")


# Providers name the model's reasoning differently; read them in the v0 client's precedence.
# `reasoning` (vLLM / Together / OpenRouter), `reasoning_content` (DeepSeek / Qwen / SGLang /
# Fireworks / Kimi), `reasoning_details` (OpenRouter / MiniMax).
REASONING_FIELDS = ("reasoning", "reasoning_content", "reasoning_details")


def reasoning_text(data: Mapping[str, Any]) -> str | None:
    """The model's reasoning string, from whichever field the provider used."""
    for field in REASONING_FIELDS:
        value = data.get(field)
        if isinstance(value, str) and value:
            return value
    details = data.get("reasoning_details")
    if isinstance(details, list):
        parts = []
        for detail in details:
            if not isinstance(detail, Mapping):
                continue
            value = detail.get("text") or detail.get("summary")
            if isinstance(value, str) and value:
                parts.append(value)
        return "\n".join(parts) or None
    return None


def parse_message(raw: dict) -> Message:
    """An OpenAI chat request message dict -> a typed Message. User/system bodies keep their
    image parts (multimodal ingress); assistant bodies flatten to text."""
    role = raw.get("role")
    content = raw.get("content")
    if role == "system":
        return SystemMessage(content=content_to_parts(content))
    if role == "tool":
        return ToolMessage(
            tool_call_id=raw.get("tool_call_id", ""),
            content=content_to_parts(content),
            name=raw.get("name"),
        )
    if role == "assistant":
        details = raw.get("reasoning_details")
        text = (
            "".join(part.get("text", "") for part in content if isinstance(part, dict))
            if isinstance(content, list)
            else content or ""
        )
        calls = []
        for call in raw.get("tool_calls") or []:
            kind = call.get("type", "function")
            native = call[kind]
            calls.append(
                ToolCall(
                    id=call["id"],
                    type=kind,
                    name=native["name"],
                    arguments=native["input" if kind == "custom" else "arguments"],
                )
            )
        return AssistantMessage(
            content=text or None,
            reasoning_content=reasoning_text(raw),
            tool_calls=calls or None,
            provider_state=details if isinstance(details, list) and details else None,
        )
    return UserMessage(content=content_to_parts(content))


def parse_tools(raw: list[dict] | None) -> list[Tool] | None:
    # `or None` so a tools array with no function entries (e.g. only `custom`/built-in
    # tools) parses to None, not [] — the same contract as the anthropic/responses
    # dialects, and what keeps an empty parse from clearing `Trace.tools`.
    if not raw:
        return None
    return [
        Tool(
            name=t["function"]["name"],
            description=t["function"].get("description", ""),
            parameters=t["function"].get("parameters", {}),
            strict=t["function"].get("strict"),
        )
        for t in raw
        if t.get("type", "function") == "function"
    ] or None


# --- vf -> chat wire ----------------------------------------------------------
# `message_to_wire` (chat-only): used by the bash harness (a Messages prompt) and the train
# client (its generate request). The proxy preserves its parsed native JSON independently and
# does not use this serializer.


def _content_to_wire(content):
    """Plain text passes through; a content-part list becomes OpenAI wire dicts (so the
    provider / renderer sees the native `image_url` shape)."""
    if isinstance(content, str):
        return content
    return [part.model_dump() for part in content]


def message_to_wire(message: Message) -> dict:
    if message.role == "assistant":
        # Strict providers reject `content: null` without tool calls.
        content = message.content
        if content is None and not message.tool_calls:
            content = ""
        wire: dict = {"role": "assistant", "content": content}
        if message.provider_state:
            wire["reasoning_details"] = message.provider_state
        elif message.reasoning_content is not None:
            wire["reasoning_content"] = message.reasoning_content
        if message.tool_calls:
            wire["tool_calls"] = [
                {
                    "id": call.id,
                    "type": call.type,
                    call.type: {
                        "name": call.name,
                        "input"
                        if call.type == "custom"
                        else "arguments": call.arguments,
                    },
                }
                for call in message.tool_calls
            ]
        return wire
    if message.role == "tool":
        wire = {
            "role": "tool",
            "tool_call_id": message.tool_call_id,
            "content": _content_to_wire(message.content),
        }
        if message.name:
            wire["name"] = message.name
        return wire
    return {"role": message.role, "content": _content_to_wire(message.content)}


def response_from_wire(completion: ChatCompletion) -> Response:
    """An OpenAI chat.completion -> a vf `Response` (the one place raw provider objects cross
    into our typed `Response`). No token ids: training tokens come from the renderer client."""
    choice = completion.choices[0]
    message = parse_message(choice.message.model_dump())
    assert isinstance(message, AssistantMessage)
    finish: FinishReason = (
        choice.finish_reason if choice.finish_reason in FINISH_REASONS else None
    )
    usage = Usage.from_openai(completion.usage)
    return Response(
        id=completion.id,
        created=completion.created,
        model=completion.model,
        message=message,
        finish_reason=finish,
        usage=usage,
    )


@dataclass
class ChatStreamParser(StreamParser):
    """Incrementally assemble Chat Completions deltas without retaining SSE bytes."""

    message: dict = dataclass_field(
        default_factory=lambda: {"role": "assistant", "content": None}
    )
    message_parts: dict[str, list[str]] = dataclass_field(default_factory=dict)
    tool_calls: dict[int, dict] = dataclass_field(default_factory=dict)
    tool_inputs: dict[int, list[str]] = dataclass_field(default_factory=dict)
    reasoning_details: list[dict] = dataclass_field(default_factory=list)
    reasoning_detail_parts: dict[int, tuple[str, list[str]]] = dataclass_field(
        default_factory=dict
    )
    finish_reason: str | None = None
    usage: dict | None = None
    head: dict | None = None

    def feed(self, raw: bytes) -> None:
        chunk = parse_sse_event(raw)
        if chunk is None:
            return
        if self.head is None:
            self.head = chunk
        self.usage = chunk.get("usage") or self.usage
        for choice in chunk.get("choices") or []:
            if choice.get("index", 0) != 0:
                continue
            self.finish_reason = choice.get("finish_reason") or self.finish_reason
            delta = choice.get("delta") or {}
            for key in ("content", "reasoning_content", "reasoning"):
                if delta.get(key) is not None:
                    self.message_parts.setdefault(key, []).append(delta[key])
            for detail in delta.get("reasoning_details") or []:
                previous = self.reasoning_details[-1] if self.reasoning_details else {}
                detail_type = detail.get("type")
                content_field = {
                    "reasoning.summary": "summary",
                    "reasoning.text": "text",
                }.get(detail_type)
                if (
                    content_field
                    and detail_type == previous.get("type")
                    and all(
                        previous.get(field_name) is None
                        or detail.get(field_name) is None
                        or previous[field_name] == detail[field_name]
                        for field_name in ("id", "index", "format")
                    )
                ):
                    self.reasoning_detail_parts.setdefault(
                        len(self.reasoning_details) - 1,
                        (content_field, [previous.get(content_field) or ""]),
                    )[1].append(detail.get(content_field) or "")
                    for field_name in ("id", "index", "signature", "format"):
                        value = previous.get(field_name) or detail.get(field_name)
                        if value is not None:
                            previous[field_name] = value
                else:
                    self.reasoning_details.append(detail)
            for tool_call in delta.get("tool_calls") or []:
                index = tool_call.get("index", 0)
                slot = self.tool_calls.setdefault(index, {})
                slot["id"] = tool_call.get("id") or slot.get("id", "")
                kind = tool_call.get("type") or (
                    "custom" if tool_call.get("custom") is not None else "function"
                )
                slot["type"] = kind
                native = slot.setdefault(kind, {"name": ""})
                delta_native = tool_call.get(kind) or {}
                if delta_native.get("name"):
                    native["name"] = delta_native["name"]
                input_field = "input" if kind == "custom" else "arguments"
                self.tool_inputs.setdefault(index, []).append(
                    delta_native.get(input_field) or ""
                )

    def finish(self) -> Response:
        for key, parts in self.message_parts.items():
            if parts:
                self.message[key] = "".join(parts)
        for index, (content_field, parts) in self.reasoning_detail_parts.items():
            self.reasoning_details[index][content_field] = "".join(parts)
        for index, parts in self.tool_inputs.items():
            call = self.tool_calls[index]
            input_field = "input" if call["type"] == "custom" else "arguments"
            call[call["type"]][input_field] = "".join(parts)
        if self.tool_calls:
            self.message["tool_calls"] = [
                self.tool_calls[index] for index in sorted(self.tool_calls)
            ]
        if self.reasoning_details:
            self.message["reasoning_details"] = self.reasoning_details
        head = self.head or {}
        completion = {
            "id": head.get("id", "vf-intercept"),
            "object": "chat.completion",
            "created": head.get("created", int(time.time())),
            "model": head.get("model", ""),
            "choices": [
                {
                    "index": 0,
                    "message": self.message,
                    "finish_reason": self.finish_reason or "stop",
                }
            ],
            "usage": self.usage,
        }
        response = response_from_wire(ModdedChatCompletion.model_validate(completion))
        response.raw = completion
        return response


class ChatDialect(Dialect[ChatCompletion]):
    sampling_fields = frozenset(
        {
            "temperature",
            "top_p",
            "top_k",
            "min_p",
            "max_tokens",
            "max_completion_tokens",
            "reasoning_effort",
            "seed",
            "stop",
            "n",
            "logprobs",
            "top_logprobs",
            "logit_bias",
            "frequency_penalty",
            "presence_penalty",
            "repetition_penalty",
            "response_format",
            "tool_choice",
            "parallel_tool_calls",
            "extra_body",
        }
    )
    routes = ("/v1/chat/completions",)
    upstream_path = "/chat/completions"
    response_type = ModdedChatCompletion

    def mediate_external_capabilities(
        self, body: RawRequest, policy: NetworkPolicyConfig
    ) -> tuple[RawRequest, list[str]]:
        mediated = body
        capabilities: list[str] = []

        if mediated.pop("web_search_options", None) is not None:
            capabilities.append("web_search_options")
        if mediated.pop("plugins", None) is not None:
            capabilities.append("plugins")

        audio = mediated.get("audio")
        voice = audio.get("voice") if isinstance(audio, dict) else None
        if isinstance(voice, dict) and voice.get("id"):
            capabilities.append("audio.voice.id")
            mediated.pop("audio")
            modalities = mediated.get("modalities")
            if isinstance(modalities, list):
                mediated["modalities"] = [
                    item for item in modalities if item != "audio"
                ] or ["text"]

        raw_tools = mediated.get("tools")
        tool_items = raw_tools if isinstance(raw_tools, list) else []
        if raw_tools is not None and not isinstance(raw_tools, list):
            capabilities.append("tools")
        tools = []
        for index, tool in enumerate(tool_items):
            if (
                isinstance(tool, dict)
                and tool.get("type", "function") in _CLIENT_TOOL_TYPES
            ):
                tools.append(tool)
            else:
                capabilities.append(f"tools[{index}].type")
        if "tools" in mediated:
            mediated["tools"] = tools

        choice = mediated.get("tool_choice")
        valid_choice = choice is None or (
            isinstance(choice, str) and choice in ("none", "auto", "required")
        )
        if isinstance(choice, dict):
            kind = choice.get("type", "function")
            valid_choice = any(
                kind == tool.get("type", "function")
                and isinstance(tool.get(kind), dict)
                and isinstance(choice.get(kind), dict)
                and tool[kind].get("name") == choice[kind].get("name")
                for tool in tools
            )
            if kind == "allowed_tools":
                allowed = choice.get("allowed_tools")
                allowed_tools = (
                    allowed.get("tools") if isinstance(allowed, dict) else None
                )
                valid_choice = isinstance(allowed_tools, list) and all(
                    isinstance(tool, dict)
                    and tool.get("type", "function") in _CLIENT_TOOL_TYPES
                    for tool in allowed_tools
                )
        if raw_tools is not None and not tools and choice not in (None, "none"):
            valid_choice = False
        if not valid_choice:
            capabilities.append(
                "tool_choice.type" if isinstance(choice, dict) else "tool_choice"
            )
            mediated.pop("tool_choice", None)

        for message_index, message in enumerate(mediated.get("messages") or []):
            if not isinstance(message, dict):
                continue
            if isinstance(message.get("audio"), dict) and message["audio"].get("id"):
                path = f"messages[{message_index}].audio.id"
                capabilities.append(path)
                message.pop("audio")
                if message.get("content") is None:
                    message["content"] = ""
            content = message.get("content")
            if not isinstance(content, list):
                continue
            safe_content = []
            for part_index, part in enumerate(content):
                path = f"messages[{message_index}].content[{part_index}]"
                capability = None
                kind = part.get("type") if isinstance(part, dict) else None
                if kind not in _SAFE_CONTENT_TYPES:
                    capability = f"{path}.type"
                elif kind == "image_url":
                    image = part.get("image_url") or {}
                    url = image.get("url") if isinstance(image, dict) else image
                    if not isinstance(url, str) or blocked_url(url, policy):
                        capability = f"{path}.image_url.url"
                elif kind == "file":
                    file = part.get("file")
                    if not isinstance(file, dict):
                        capability = f"{path}.file"
                    elif file.get("file_id"):
                        capability = f"{path}.file.file_id"
                    else:
                        file_data = file.get("file_data")
                        if file_data is not None and not isinstance(file_data, str):
                            capability = f"{path}.file.file_data"
                        elif isinstance(file_data, str):
                            try:
                                parsed = urlsplit(file_data)
                            except ValueError:
                                capability = f"{path}.file.file_data"
                            else:
                                if (parsed.scheme or parsed.netloc) and blocked_url(
                                    file_data, policy
                                ):
                                    capability = f"{path}.file.file_data"
                if capability is None:
                    safe_content.append(part)
                else:
                    capabilities.append(capability)
            message["content"] = safe_content or ""

        append_user_notice(mediated.setdefault("messages", []))
        return mediated, capabilities

    def parse_request(self, body: RawRequest) -> Request:
        if body.get("n", 1) != 1:
            raise ValueError("chat completions require n=1")
        messages: Messages = []
        tool_names: dict[str, str] = {}
        for raw in body.get("messages", []):
            message = parse_message(raw)
            if isinstance(message, ToolMessage) and message.name is None:
                name = tool_names.get(message.tool_call_id)
                if name is not None:
                    message = message.model_copy(update={"name": name})
            messages.append(message)
            if isinstance(message, AssistantMessage):
                for call in message.tool_calls or []:
                    tool_names[call.id] = call.name
        return Request(messages=messages, tools=parse_tools(body.get("tools")))

    def parse_sampling(self, body: RawRequest) -> Sampling:
        settings = {k: v for k, v in body.items() if k in self.sampling_fields}
        # Canonicalize the max-tokens alias; when both ride the wire (an eval override
        # on top of a harness's `max_completion_tokens`), the override wins.
        if (mct := settings.pop("max_completion_tokens", None)) is not None:
            settings.setdefault("max_tokens", mct)
        return Sampling.model_validate(settings)

    def parse_response(self, response: ChatCompletion) -> Response:
        return response_from_wire(response)

    def rewrite_request(self, body: dict, before: Request, after: Request) -> None:
        for native, original, rewritten in zip(
            body.get("messages", []), before.messages, after.messages, strict=True
        ):
            if rewritten != original:
                native["content"] = _content_to_wire(rewritten.content)
                if isinstance(rewritten, ToolMessage):
                    if rewritten.name is None:
                        native.pop("name", None)
                    else:
                        native["name"] = rewritten.name

    def rewrite_response(self, raw: dict, text: str) -> None:
        for choice in raw.get("choices") or []:
            if isinstance(choice.get("message"), dict):
                choice["message"] = {"role": "assistant", "content": text}
                choice["finish_reason"] = "stop"
                choice.pop("logprobs", None)

    def stream_events(self, raw: dict) -> list[bytes]:
        choice = (raw.get("choices") or [{}])[0]
        chunk = {
            **{key: value for key, value in raw.items() if key != "choices"},
            "object": "chat.completion.chunk",
            "choices": [
                {
                    "index": 0,
                    "delta": choice.get("message")
                    or {"role": "assistant", "content": ""},
                    "finish_reason": choice.get("finish_reason"),
                }
            ],
        }
        return [f"data: {json.dumps(chunk)}\n\n".encode(), b"data: [DONE]\n\n"]

    def stream_parser(self) -> StreamParser:
        return ChatStreamParser()

    def apply_overrides(
        self, body: RawRequest, model: str, sampling: SamplingConfig
    ) -> RawRequest:
        # Preserve the program's native fields, overlaying only what the eval owns: the model and
        # the sampling knobs it set. The selected model is authoritative even if a permissive
        # sampling config carries an extra field named `model`.
        return {**body, **sampling.model_dump(exclude_none=True), "model": model}
