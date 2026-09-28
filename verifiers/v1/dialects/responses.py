"""The OpenAI Responses dialect (codex and friends).

Request parsing walks the `input` items, folding each run of assistant-side items (reasoning /
assistant message / function or custom tool call) into one typed assistant message; response
parsing reads the `output` items. Relay-only: the eval client forwards the program's bytes to a
`/responses` endpoint and this dialect parses a copy for the trace. Server-side statefulness
(`previous_response_id`) is not emulated — the endpoint owns it.
"""

import json
from collections import deque

from openai.types.responses.response_usage import (
    InputTokensDetails,
    OutputTokensDetails,
    ResponseUsage,
)
from pydantic import BaseModel, ConfigDict

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
from verifiers.v1.errors import model_error
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

# The placeholder response a committed stream's keepalives carry until its turn is ready:
# schema-valid, so strictly validating clients accept the events that carry it.
_KEEPALIVE_RESPONSE = {
    "id": "resp_keepalive",
    "object": "response",
    "created_at": 0,
    "model": "",
    "status": "in_progress",
    "output": [],
    "parallel_tool_calls": True,
    "tool_choice": "auto",
    "tools": [],
}

FINAL_EVENTS = ("response.completed", "response.incomplete", "response.failed")
# Byte markers for the terminal event types above, in both compact and spaced JSON, so the
# interception server can cheaply spot the turn-ending event without parsing each delta.
_TERMINAL_MARKERS = tuple(
    marker.encode()
    for event in FINAL_EVENTS
    for marker in (f'"type":"{event}"', f'"type": "{event}"')
)
# Sampling knobs the eval owns, in this format's shape (Responses uses `max_output_tokens`).
_SAMPLING_KEYS = frozenset({"temperature", "top_p", "max_output_tokens", "max_tokens"})
TEXT_TOOL_OUTPUT_TYPES = ("function_call_output", "custom_tool_call_output")


class ProviderUsageInputTokensDetails(InputTokensDetails):
    """Permissive input token details: OpenAI-compatible providers may omit fields
    the pinned SDK declares required (e.g. ``cache_write_tokens``)."""

    cache_write_tokens: int | None = None
    cached_tokens: int | None = None


class ProviderUsageOutputTokensDetails(OutputTokensDetails):
    """Permissive output token details: providers may omit ``reasoning_tokens``."""

    reasoning_tokens: int | None = None


class ProviderUsage(ResponseUsage):
    """Responses usage with optional detail objects for OpenAI-compatible providers."""

    input_tokens_details: ProviderUsageInputTokensDetails | None = None
    output_tokens_details: ProviderUsageOutputTokensDetails | None = None


class OpenAIResponse(BaseModel):
    """Permissive parse-only view of a Responses object: `extra='allow'` keeps it a plain dict
    for the trace (read via `model_dump`), so a strict SDK model can't crash the rollout on a
    provider/SDK enum skew (e.g. a value the pinned `openai` rejects)."""

    model_config = ConfigDict(extra="allow")
    usage: ProviderUsage | None = None


def parse_content(content) -> str | list[ContentPart]:
    if isinstance(content, str):
        return content
    parts: list[ContentPart] = []
    for part in content or []:
        kind = part.get("type")
        if kind in ("input_text", "output_text"):
            parts.append(TextContentPart(text=part.get("text", "")))
        elif kind == "input_image":
            parts.append(
                ImageUrlContentPart(
                    image_url=ImageUrlSource(url=part.get("image_url", ""))
                )
            )
    return parts


def content_to_wire(content):
    if isinstance(content, str):
        return content
    return [
        {"type": "input_text", "text": part.text}
        if isinstance(part, TextContentPart)
        else {"type": "input_image", "image_url": part.image_url.url}
        for part in content
    ]


def input_to_wire(content):
    encoded = content_to_wire(content)
    return (
        encoded if isinstance(encoded, str) else [{"role": "user", "content": encoded}]
    )


def fold_assistant(items: list[dict] | None) -> AssistantMessage:
    """Assistant-side Responses items -> one typed assistant message."""
    items = [
        {**item, "tools": [provider_declaration(t) for t in item.get("tools") or []]}
        if item.get("type") in ("additional_tools", "tool_search_output")
        else item
        for item in items or []
    ]
    content = ""
    reasoning: list[str] = []
    calls: list[ToolCall] = []
    for item in items or []:
        kind = item.get("type")
        if kind == "reasoning":
            reasoning += [s.get("text", "") for s in item.get("summary") or []]
            reasoning += [c.get("text", "") for c in item.get("content") or []]
        elif kind in ("function_call", "custom_tool_call"):
            calls.append(
                ToolCall(
                    id=item.get("call_id", ""),
                    type="custom" if kind == "custom_tool_call" else "function",
                    name=item.get("name", ""),
                    namespace=item.get("namespace"),
                    arguments=item.get("arguments", item.get("input", "")),
                )
            )
        else:
            raw = item.get("content")
            content += (
                raw
                if isinstance(raw, str)
                else "".join(
                    p.get("text", "")
                    for p in raw or []
                    if p.get("type") in ("input_text", "output_text")
                )
            )
    return with_provider_identity(
        AssistantMessage(
            content=content or None,
            reasoning_content="\n".join(r for r in reasoning if r) or None,
            tool_calls=calls or None,
            provider_state=items,
        )
    )


def response_from_wire(response: OpenAIResponse) -> Response:
    """An OpenAI Responses object -> a vf `Response` (its `output` items folded into one
    assistant message)."""
    # Copy the output snapshot without traversing echoed tools or request settings.
    data = response.model_dump(
        include={"id", "created_at", "model", "status", "error", "output"}
    )
    status = data.get("status")
    if status not in (None, "completed", "incomplete"):
        error = data.get("error") or {}
        code = error.get("code") if isinstance(error, dict) else None
        message = error.get("message") if isinstance(error, dict) else None
        detail = ": ".join(str(value) for value in (status, code, message) if value)
        status_code = (
            429
            if code in ("rate_limit_exceeded", "rate_limit_error")
            else 400
            if code in ("invalid_prompt", "context_length_exceeded")
            else 502
        )
        raise model_error(
            f"upstream Responses request did not complete: {detail}",
            status_code=status_code,
        )
    message = fold_assistant(data.get("output"))
    finish: FinishReason = (
        "length"
        if data.get("status") == "incomplete"
        else ("tool_calls" if message.tool_calls else "stop")
    )
    usage = None
    if response.usage:
        provider_usage = response.usage
        input_details = provider_usage.input_tokens_details
        output_details = provider_usage.output_tokens_details
        cached = input_details.cached_tokens if input_details else None
        # Responses input_tokens includes cache hits; vf keeps the buckets disjoint.
        usage = Usage(
            prompt_tokens=provider_usage.input_tokens - (cached or 0),
            completion_tokens=provider_usage.output_tokens,
            cached_input_tokens=cached,
            reasoning_tokens=output_details.reasoning_tokens
            if output_details
            else None,
            cost=getattr(provider_usage, "cost", None),
        )
    return Response(
        id=data.get("id", ""),
        created=data.get("created_at", 0),
        model=data.get("model", ""),
        message=message,
        finish_reason=finish,
        usage=usage,
    )


class ResponsesStreamParser(StreamParser):
    """Retain only the complete terminal response event and trailing DONE event."""

    def __init__(self) -> None:
        self.events: deque[bytes] = deque(maxlen=2)
        self.feed = self.events.append
        self.terminal_events: tuple[bytes, ...] | None = None

    def on_done(self) -> None:
        # Freeze the terminal tail before later relay chunks can evict it.
        self.terminal_events = tuple(self.events)

    def finish(self) -> Response:
        events = self.terminal_events or self.events
        for raw in reversed(events):
            event = parse_sse_event(raw)
            if event and event.get("type") in FINAL_EVENTS:
                response = response_from_wire(
                    OpenAIResponse.model_validate(event["response"])
                )
                response.raw = event["response"]
                return response
        raise ValueError("Responses stream ended without a terminal event")


class ResponsesDialect(Dialect[OpenAIResponse]):
    sampling_fields = frozenset(
        {
            "temperature",
            "top_p",
            "max_output_tokens",
            "max_tool_calls",
            "reasoning",
            "service_tier",
            "text",
            "tool_choice",
            "parallel_tool_calls",
            "top_logprobs",
            "truncation",
        }
    )
    routes = ("/v1/responses",)
    upstream_path = "/responses"
    response_type = OpenAIResponse

    def is_terminal_event(self, chunk: bytes) -> bool:
        # A Responses client (e.g. codex) ends its turn on `response.completed`, before the
        # trailing `[DONE]`, so the turn-ending event is the final event, not the sentinel.
        return any(marker in chunk for marker in _TERMINAL_MARKERS)

    def parse_sampling(self, body: RawRequest) -> Sampling:
        settings = {k: v for k, v in body.items() if k in self.sampling_fields}
        # Lift `reasoning.effort` onto the typed knob; keep any other reasoning keys
        # (e.g. `summary`) as the wire sent them.
        if isinstance(reasoning := settings.get("reasoning"), dict):
            reasoning = dict(reasoning)
            if reasoning.get("effort"):
                settings["reasoning_effort"] = reasoning.pop("effort")
            if reasoning:
                settings["reasoning"] = reasoning
            else:
                settings.pop("reasoning")
        if "max_output_tokens" in settings:
            settings["max_tokens"] = settings.pop("max_output_tokens")
        return Sampling.model_validate(settings)

    def bind_request(self, body: RawRequest) -> NativeRequest:
        prompt: Messages = []
        targets = {}
        if instructions := body.get("instructions"):
            prompt.append(SystemMessage(content=instructions))
        raw = body.get("input")
        items = (
            [{"role": "user", "content": raw}] if isinstance(raw, str) else raw or []
        )
        run: list[dict] = []  # the current run of assistant-side items
        for item in items:
            role = item.get("role")
            assistant = (
                role == "assistant"
                or role is None
                and not (item.get("type") or "").endswith(("_output", "_response"))
            )
            if run and not assistant:
                prompt.append(fold_assistant(run))
                run = []
            if assistant:
                run.append(item)
            elif item.get("type") in TEXT_TOOL_OUTPUT_TYPES:
                output = item.get("output")
                content = (
                    parse_content(output)
                    if isinstance(output, (str, list))
                    else json.dumps(output)
                )
                targets[len(prompt)] = ContentTarget(item, "output", content_to_wire)
                prompt.append(
                    ToolMessage(
                        tool_call_id=item.get("call_id", ""),
                        content=content,
                    )
                )
            elif item.get("role") in ("system", "developer"):
                prompt.append(SystemMessage(content=parse_content(item.get("content"))))
            else:
                targets[len(prompt)] = (
                    ContentTarget(body, "input", input_to_wire)
                    if isinstance(raw, str)
                    else ContentTarget(item, "content", content_to_wire)
                )
                prompt.append(UserMessage(content=parse_content(item.get("content"))))
        if run:
            prompt.append(fold_assistant(run))
        tools, provider_tools = [], []
        declarations = []
        for item in items:
            if item.get("type") in ("additional_tools", "tool_search_output"):
                declarations.extend(item.get("tools") or [])
        # Current declarations take precedence over definitions replayed in history.
        declarations.extend(body.get("tools") or [])
        for group in declarations:
            namespace = group["name"] if group.get("type") == "namespace" else None
            for tool in group["tools"] if namespace else [group]:
                if tool.get("type") not in ("function", "custom"):
                    declaration = provider_declaration(tool)
                    if namespace:
                        declaration = {
                            "type": "namespace",
                            "name": namespace,
                            "tools": [declaration],
                        }
                    provider_tools.append(declaration)
                    continue
                tools.append(
                    Tool.model_validate(
                        tool
                        | {
                            "name": tool["name"],
                            "namespace": namespace,
                            "description": tool.get("description") or "",
                            "parameters": tool.get("parameters") or {},
                        }
                    )
                )
        tools = list({(t.namespace, t.name, t.type): t for t in tools}.values())
        return NativeRequest(
            body, Request(messages=prompt, tools=tools or None), targets, provider_tools
        )

    def parse_response(self, response: OpenAIResponse) -> Response:
        return response_from_wire(response)

    def rewrite_response(self, raw: dict, text: str) -> None:
        original = next(
            (
                item
                for item in raw.get("output") or []
                if isinstance(item, dict) and item.get("type") == "message"
            ),
            {},
        )
        raw["output"] = [
            {
                "type": "message",
                "id": original.get("id") or "msg_intercepted",
                "role": "assistant",
                "status": "completed",
                "content": [{"type": "output_text", "text": text, "annotations": []}],
            }
        ]
        raw.update(status="completed", error=None, incomplete_details=None)
        raw.pop("required_action", None)
        if "output_text" in raw:
            raw["output_text"] = text

    def stream_keepalive(self, first: bool) -> bytes:
        # Codex's idle timer resets only on data events (it drops comments), and the OpenAI
        # SDK's stream helper rejects any event before `response.created`: open with a
        # placeholder `response.created`, then repeat `response.in_progress`. The turn's own
        # `response.created` follows and supersedes it; clients take the turn's id and final
        # state from its own events, and its sequence numbers restart at 0.
        kind = "response.created" if first else "response.in_progress"
        payload = {"type": kind, "sequence_number": 0, "response": _KEEPALIVE_RESPONSE}
        return f"event: {kind}\ndata: {json.dumps(payload)}\n\n".encode()

    def stream_error(self, error: dict) -> bytes:
        # `response.failed` is what Responses clients (codex) act on; the `error` key is what
        # the OpenAI SDK raises on.
        message = error["error"]["message"]
        failed = {
            **_KEEPALIVE_RESPONSE,
            "status": "failed",
            "error": {"code": "server_error", "message": message},
        }
        payload = {
            "type": "response.failed",
            "sequence_number": 0,
            "response": failed,
            **error,
        }
        return f"event: response.failed\ndata: {json.dumps(payload)}\n\n".encode()

    def stream_events(self, raw: dict) -> list[bytes]:
        item = raw["output"][0]
        part = item["content"][0]
        common = {"output_index": 0, "item_id": item["id"], "content_index": 0}
        logprobs = part.get("logprobs") or []
        head = {
            **raw,
            "status": "in_progress",
            "output": [],
            "completed_at": None,
        }
        events = [
            ("response.created", {"response": head}),
            (
                "response.output_item.added",
                {"output_index": 0, "item": {**item, "content": []}},
            ),
            (
                "response.content_part.added",
                {**common, "part": {**part, "text": ""}},
            ),
            # `logprobs` is required on both text events; carry the part's own.
            (
                "response.output_text.delta",
                {**common, "delta": part["text"], "logprobs": logprobs},
            ),
            (
                "response.output_text.done",
                {**common, "text": part["text"], "logprobs": logprobs},
            ),
            ("response.content_part.done", {**common, "part": part}),
            ("response.output_item.done", {"output_index": 0, "item": item}),
            ("response.completed", {"response": raw}),
        ]
        return [
            *(
                f"data: {json.dumps({'type': kind, 'sequence_number': i, **data})}\n\n".encode()
                for i, (kind, data) in enumerate(events)
            ),
            b"data: [DONE]\n\n",
        ]

    def stream_parser(self) -> StreamParser:
        return ResponsesStreamParser()

    def apply_overrides(
        self, body: RawRequest, model: str, sampling: SamplingConfig
    ) -> RawRequest:
        # Preserve native fields except the eval's model + sampling, mapped to the Responses shape
        # (`max_tokens` -> `max_output_tokens`); sampling is authoritative.
        s = sampling.wire_args()
        max_tokens = s.pop("max_tokens", None)
        reasoning_effort = s.pop("reasoning_effort", None)
        sampling_reasoning = s.pop("reasoning", None)
        name = model.rsplit("/", 1)[-1]
        reasoning_model = (
            name.startswith(("gpt-5", "o1", "o3", "o4"))
            and "-chat" not in name
            and ("/" not in model or model.startswith("openai/"))
        )
        overrides: dict = {**s, "model": model}
        if reasoning_model:
            # Preserve opaque reasoning state so it can be replayed on the next turn.
            include = list(overrides.get("include") or body.get("include") or [])
            if "reasoning.encrypted_content" not in include:
                include.append("reasoning.encrypted_content")
            overrides["include"] = include
        if max_tokens is not None:
            overrides["max_output_tokens"] = max_tokens
        reasoning = {
            **({"summary": "auto"} if reasoning_model else {}),
            **dict(body.get("reasoning") or {}),
            **dict(sampling_reasoning or {}),
        }
        if reasoning_effort is not None:
            reasoning["effort"] = reasoning_effort
        if reasoning:
            overrides["reasoning"] = reasoning
        steered = {
            k: v
            for k, v in body.items()
            if k not in _SAMPLING_KEYS and k not in overrides
        }
        return {**steered, **overrides}
