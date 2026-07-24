"""The OpenAI Responses dialect (codex and friends).

Request parsing walks the `input` items, folding each run of assistant-side items (reasoning /
assistant message / function_call) into one typed assistant message; response parsing reads the
`output` items. Relay-only: the eval client forwards the program's bytes to a `/responses`
endpoint and this dialect parses a copy for the trace. Server-side statefulness
(`previous_response_id`) is not emulated — the endpoint owns it.
"""

import json
from collections import deque

from openai.types.responses import (
    ResponseUsage,
)
from pydantic import BaseModel, ConfigDict

from verifiers.v1.dialects.base import Dialect, StreamParser, iter_sse_reverse
from verifiers.v1.types import (
    AssistantMessage,
    ContentPart,
    FinishReason,
    ImageUrlContentPart,
    ImageUrlSource,
    Messages,
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


class ProviderUsageInputTokensDetails(BaseModel):
    """Permissive input token details: OpenAI-compatible providers may omit fields
    the pinned SDK declares required (e.g. ``cache_write_tokens``)."""

    model_config = ConfigDict(extra="allow")
    cache_write_tokens: int | None = None
    cached_tokens: int | None = None


class ProviderUsageOutputTokensDetails(BaseModel):
    """Permissive output token details: providers may omit ``reasoning_tokens``."""

    model_config = ConfigDict(extra="allow")
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


def fold_assistant(items: list[dict]) -> AssistantMessage:
    """One run of assistant-side items (reasoning / message / function_call) -> one typed
    assistant message."""
    content = ""
    reasoning: list[str] = []
    calls: list[ToolCall] = []
    for item in items:
        if item.get("type") == "reasoning":
            reasoning += [s.get("text", "") for s in item.get("summary") or []]
            reasoning += [c.get("text", "") for c in item.get("content") or []]
        elif item.get("type") == "function_call":
            calls.append(
                ToolCall(
                    id=item.get("call_id", ""),
                    name=item.get("name", ""),
                    arguments=item.get("arguments", ""),
                )
            )
        else:  # an assistant message item
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
    return AssistantMessage(
        content=content or None,
        reasoning_content="\n".join(r for r in reasoning if r) or None,
        tool_calls=calls or None,
        provider_state=items,
    )


def response_from_wire(response: OpenAIResponse) -> Response:
    """An OpenAI Responses object -> a vf `Response` (its `output` items folded into one
    assistant message)."""
    data = response.model_dump()
    content = ""
    reasoning: list[str] = []
    calls: list[ToolCall] = []
    for item in data.get("output") or []:
        kind = item.get("type")
        if kind == "message":
            content += "".join(
                p.get("text", "")
                for p in item.get("content") or []
                if p.get("type") == "output_text"
            )
        elif kind == "reasoning":
            reasoning += [s.get("text", "") for s in item.get("summary") or []]
            reasoning += [c.get("text", "") for c in item.get("content") or []]
        elif kind == "function_call":
            calls.append(
                ToolCall(
                    id=item.get("call_id", ""),
                    name=item.get("name", ""),
                    arguments=item.get("arguments", ""),
                )
            )
    tool_calls = calls or None
    finish: FinishReason = (
        "length"
        if data.get("status") == "incomplete"
        else ("tool_calls" if tool_calls else "stop")
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
        message=AssistantMessage(
            content=content or None,
            reasoning_content="\n".join(r for r in reasoning if r) or None,
            tool_calls=tool_calls,
            provider_state=data.get("output"),
        ),
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
        for event in iter_sse_reverse(b"".join(events)):
            if event.get("type") in FINAL_EVENTS:
                raw = event["response"]
                response = response_from_wire(OpenAIResponse.model_validate(raw))
                response.raw = raw
                return response
        raise ValueError("Responses stream ended without a terminal event")


class ResponsesDialect(Dialect[dict, OpenAIResponse]):
    sampling_fields = frozenset(
        {
            "temperature",
            "top_p",
            "max_output_tokens",
            "max_tool_calls",
            "reasoning",
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

    def parse_sampling(self, body: dict) -> Sampling:
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

    def parse_request(self, body: dict) -> tuple[Messages, list[Tool] | None]:
        prompt: Messages = []
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
            elif item.get("type") == "function_call_output":
                output = item.get("output")
                content = (
                    parse_content(output)
                    if isinstance(output, (str, list))
                    else json.dumps(output)
                )
                prompt.append(
                    ToolMessage(
                        tool_call_id=item.get("call_id", ""),
                        content=content,
                    )
                )
            elif item.get("role") in ("system", "developer"):
                prompt.append(SystemMessage(content=parse_content(item.get("content"))))
            else:
                prompt.append(UserMessage(content=parse_content(item.get("content"))))
        if run:
            prompt.append(fold_assistant(run))
        tools = [
            Tool(
                name=t["name"],
                description=t.get("description") or "",
                parameters=t.get("parameters") or {},
                strict=t.get("strict"),
            )
            for t in body.get("tools") or []
            if t.get("type") == "function"
        ] or None
        return prompt, tools

    def parse_response(self, response: OpenAIResponse) -> Response:
        return response_from_wire(response)

    def rewrite_response(self, raw: dict, text: str) -> None:
        output = raw.get("output") or []
        original = next(
            (
                item
                for item in output
                if isinstance(item, dict) and item.get("type") == "message"
            ),
            {},
        )
        raw["output"] = [
            item
            for item in output
            if isinstance(item, dict) and item.get("type") == "reasoning"
        ] + [
            {
                "type": "message",
                "id": original.get("id") or "msg_intercepted",
                "role": "assistant",
                "status": "completed",
                "content": [{"type": "output_text", "text": text, "annotations": []}],
            }
        ]
        raw["status"] = "completed"
        raw["error"] = None
        raw["incomplete_details"] = None
        raw.pop("required_action", None)
        if "output_text" in raw:
            raw["output_text"] = text

    def rewrite_tool_result(self, body: dict, tool_call_id: str, text: str) -> None:
        for item in body.get("input") or []:
            if (
                isinstance(item, dict)
                and item.get("type") == "function_call_output"
                and item.get("call_id") == tool_call_id
            ):
                item["output"] = text

    def stream_events(self, raw: dict) -> list[bytes]:
        sequence = 0
        events: list[bytes] = []

        def emit(kind: str, **fields) -> None:
            nonlocal sequence
            event = {"type": kind, "sequence_number": sequence, **fields}
            events.append(f"data: {json.dumps(event)}\n\n".encode())
            sequence += 1

        head = {
            **raw,
            "status": "in_progress",
            "output": [],
            "error": None,
            "incomplete_details": None,
            "completed_at": None,
        }
        emit("response.created", response=head)
        for output_index, item in enumerate(raw.get("output") or []):
            if item.get("type") != "message":
                emit(
                    "response.output_item.added",
                    output_index=output_index,
                    item=item,
                )
                emit(
                    "response.output_item.done",
                    output_index=output_index,
                    item=item,
                )
                continue

            item_id = item.get("id") or "msg_intercepted"
            emit(
                "response.output_item.added",
                output_index=output_index,
                item={**item, "status": "in_progress", "content": []},
            )
            for content_index, part in enumerate(item.get("content") or []):
                empty = (
                    {**part, "text": ""} if part.get("type") == "output_text" else part
                )
                emit(
                    "response.content_part.added",
                    output_index=output_index,
                    item_id=item_id,
                    content_index=content_index,
                    part=empty,
                )
                if part.get("type") == "output_text":
                    logprobs = part.get("logprobs") or []
                    text = part.get("text", "")
                    emit(
                        "response.output_text.delta",
                        output_index=output_index,
                        item_id=item_id,
                        content_index=content_index,
                        delta=text,
                        logprobs=logprobs,
                    )
                    emit(
                        "response.output_text.done",
                        output_index=output_index,
                        item_id=item_id,
                        content_index=content_index,
                        text=text,
                        logprobs=logprobs,
                    )
                emit(
                    "response.content_part.done",
                    output_index=output_index,
                    item_id=item_id,
                    content_index=content_index,
                    part=part,
                )
            emit(
                "response.output_item.done",
                output_index=output_index,
                item=item,
            )

        status = raw.get("status")
        terminal = (
            f"response.{status}"
            if status in ("incomplete", "failed")
            else "response.completed"
        )
        emit(terminal, response=raw)
        return [*events, b"data: [DONE]\n\n"]

    def stream_parser(self) -> StreamParser:
        return ResponsesStreamParser()

    def apply_overrides(self, body: dict, model: str, sampling: SamplingConfig) -> dict:
        # Preserve native fields except the eval's model + sampling, mapped to the Responses shape
        # (`max_tokens` -> `max_output_tokens`); sampling is authoritative.
        s = sampling.model_dump(exclude_none=True)
        name = model.rsplit("/", 1)[-1]
        reasoning_model = (
            name.startswith(("gpt-5", "o1", "o3", "o4"))
            and "-chat" not in name
            and ("/" not in model or model.startswith("openai/"))
        )
        overrides: dict = {"model": model}
        if reasoning_model:
            # Preserve opaque reasoning state so it can be replayed on the next turn.
            include = list(body.get("include") or [])
            if "reasoning.encrypted_content" not in include:
                include.append("reasoning.encrypted_content")
            overrides["include"] = include
        if "temperature" in s:
            overrides["temperature"] = s["temperature"]
        if "top_p" in s:
            overrides["top_p"] = s["top_p"]
        if "max_tokens" in s:
            overrides["max_output_tokens"] = s["max_tokens"]
        reasoning = dict(body.get("reasoning") or {})
        if reasoning_model:
            # Summaries provide the trace's readable reasoning text.
            reasoning = {"summary": "auto", **reasoning}
        if "reasoning_effort" in s:
            reasoning["effort"] = s["reasoning_effort"]
        if reasoning:
            overrides["reasoning"] = reasoning
        steered = {
            k: v
            for k, v in body.items()
            if k not in _SAMPLING_KEYS and k not in overrides
        }
        return {**steered, **overrides}
