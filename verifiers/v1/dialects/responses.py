"""The OpenAI Responses dialect (codex and friends).

Request parsing walks the `input` items, folding each run of assistant-side items (reasoning /
assistant message / function_call) into one typed assistant message; response parsing reads the
`output` items. Relay-only: the eval client forwards the program's bytes to a `/responses`
endpoint and this dialect parses a copy for the trace. Server-side statefulness
(`previous_response_id`) is not emulated — the endpoint owns it.
"""

import json
from collections import deque
from typing import Any, cast

from openai.types.responses import (
    EasyInputMessageParam,
    ResponseFunctionToolCallParam,
    ResponseInputImageParam,
    ResponseInputMessageContentListParam,
    ResponseInputParam,
    ResponseInputTextParam,
    ResponseUsage,
)
from openai.types.responses.response_input_param import FunctionCallOutput
from openai.types.responses.response_usage import (
    InputTokensDetails,
    OutputTokensDetails,
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
# Sampling knobs the eval owns, in this format's shape (Responses uses `max_output_tokens`).
_SAMPLING_KEYS = frozenset({"temperature", "top_p", "max_output_tokens", "max_tokens"})


class ProviderUsage(ResponseUsage):
    """Responses usage with optional detail objects for OpenAI-compatible providers."""

    input_tokens_details: InputTokensDetails | None = None
    output_tokens_details: OutputTokensDetails | None = None


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


def messages_to_wire(messages: Messages) -> ResponseInputParam:
    items: ResponseInputParam = []
    for message in messages:
        if isinstance(message, AssistantMessage):
            if message.provider_state:
                items.extend(cast(ResponseInputParam, message.provider_state))
                continue
            if message.content:
                items.append(
                    EasyInputMessageParam(
                        role="assistant",
                        content=message.content,
                    )
                )
            items.extend(
                ResponseFunctionToolCallParam(
                    type="function_call",
                    call_id=call.id,
                    name=call.name,
                    arguments=call.arguments,
                )
                for call in message.tool_calls or []
            )
            continue
        content: str | ResponseInputMessageContentListParam = (
            message.content
            if isinstance(message.content, str)
            else [
                ResponseInputTextParam(type="input_text", text=part.text)
                if isinstance(part, TextContentPart)
                else ResponseInputImageParam(
                    type="input_image",
                    image_url=part.image_url.url,
                    detail="auto",
                )
                for part in message.content
            ]
        )
        if isinstance(message, ToolMessage):
            items.append(
                FunctionCallOutput(
                    type="function_call_output",
                    call_id=message.tool_call_id,
                    output=cast(Any, content),
                )
            )
        else:
            items.append(EasyInputMessageParam(role=message.role, content=content))
    return items


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

    def finish(self) -> Response:
        for event in iter_sse_reverse(b"".join(self.events)):
            if event.get("type") in FINAL_EVENTS:
                return response_from_wire(
                    OpenAIResponse.model_validate(event["response"])
                )
        raise ValueError("Responses stream ended without a terminal event")


class ResponsesDialect(Dialect[dict, OpenAIResponse]):
    """The OpenAI Responses wire format."""

    routes = ("/v1/responses",)
    upstream_path = "/responses"
    response_type = OpenAIResponse

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

    def stream_parser(self) -> StreamParser:
        return ResponsesStreamParser()

    def apply_overrides(self, body: dict, model: str, sampling: SamplingConfig) -> dict:
        # Forward verbatim except the eval's model + sampling, mapped to the Responses shape
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

    def extend(
        self, body: dict, completion: dict | None, user_messages: Messages
    ) -> dict:
        """Append raw model output and the user simulator's reply for the next turn."""
        raw = body.get("input")
        items: ResponseInputParam = (
            [EasyInputMessageParam(role="user", content=raw)]
            if isinstance(raw, str)
            else cast(ResponseInputParam, list(raw or []))
        )
        items.extend(cast(ResponseInputParam, (completion or {}).get("output") or []))
        items.extend(messages_to_wire(user_messages))
        return {**body, "input": items}
