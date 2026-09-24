"""The OpenAI Responses dialect (codex and friends).

Request parsing walks the `input` items, folding each run of assistant-side items (reasoning /
assistant message / function or custom tool call) into one typed assistant message; response
parsing reads the `output` items. Relay-only: the eval client forwards the program's bytes to a
`/responses` endpoint and this dialect parses a copy for the trace. Server-side statefulness
(`previous_response_id`) is not emulated — the endpoint owns it.
"""

import json
from functools import partial

from verifiers.v1.dialects.base import Dialect, RawRequest, Setter, StreamParser
from verifiers.v1.errors import model_error
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


def parse_content(content) -> MessageContent:
    if isinstance(content, str):
        return content
    parts: list[ContentPart] = []
    for part in content or []:
        kind = part.get("type")
        if kind in ("input_text", "output_text"):
            parts.append(TextContentPart(text=part.get("text", "")))
        elif kind == "input_image" and part.get("image_url"):
            parts.append(
                ImageUrlContentPart(image_url=ImageUrlSource(url=part["image_url"]))
            )
        else:
            parts.append(NativeContentPart(native=part))
    return parts


def _content_to_wire(content: MessageContent) -> str | list[dict]:
    """Typed content in this format's input shape."""
    if isinstance(content, str):
        return content
    return [
        {"type": "input_text", "text": part.text}
        if isinstance(part, TextContentPart)
        else {"type": "input_image", "image_url": part.image_url.url}
        if isinstance(part, ImageUrlContentPart)
        else part.native
        for part in content
    ]


def _output_content(output) -> MessageContent:
    """A tool output's content: text, content parts, or one native result object (e.g. a
    computer screenshot)."""
    if isinstance(output, str):
        return output
    if isinstance(output, list):
        return parse_content(output)
    return [NativeContentPart(native=output)]


def _write_input(body: dict, message: Message) -> None:
    content = _content_to_wire(message.content)
    body["input"] = (
        content if isinstance(content, str) else [{"role": "user", "content": content}]
    )


def _write_content(item: dict, message: Message) -> None:
    item["content"] = _content_to_wire(message.content)


def _write_output(item: dict, message: Message) -> None:
    content = _content_to_wire(message.content)
    if (
        isinstance(item["output"], dict)
        and isinstance(content, list)
        and len(content) == 1
    ):
        content = content[0]  # a single result object, sent back in its own shape
    item["output"] = content


def _write_item(item: dict, message: Message) -> None:
    """Replace a protocol item recorded verbatim with its edited native form."""
    content = _content_to_wire(message.content)
    if not (isinstance(content, list) and len(content) == 1):
        raise ValueError(f"a {item.get('type')} item can only become one native part")
    item.clear()
    item.update(content[0])


def fold_assistant(items: list[dict] | None) -> AssistantMessage:
    """Assistant-side Responses items -> one typed assistant message."""
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
    return AssistantMessage(
        content=content or None,
        reasoning_content="\n".join(r for r in reasoning if r) or None,
        tool_calls=calls or None,
        provider_state=items,
    )


def response_from_wire(response: dict) -> Response:
    """An OpenAI Responses object -> a vf `Response` (its `output` items folded into one
    assistant message)."""
    status = response.get("status")
    if status not in (None, "completed", "incomplete"):
        error = response.get("error") or {}
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
    message = fold_assistant(response.get("output"))
    finish: FinishReason = (
        "length"
        if status == "incomplete"
        else ("tool_calls" if message.tool_calls else "stop")
    )
    usage = None
    if provider_usage := response.get("usage"):
        cached = (provider_usage.get("input_tokens_details") or {}).get("cached_tokens")
        # Responses input_tokens includes cache hits; vf keeps the buckets disjoint.
        usage = Usage(
            prompt_tokens=provider_usage["input_tokens"] - (cached or 0),
            completion_tokens=provider_usage["output_tokens"],
            cached_input_tokens=cached,
            reasoning_tokens=(provider_usage.get("output_tokens_details") or {}).get(
                "reasoning_tokens"
            ),
            cost=provider_usage.get("cost"),
        )
    return Response(
        id=response.get("id") or "",
        created=response.get("created_at") or 0,
        model=response.get("model") or "",
        message=message,
        finish_reason=finish,
        usage=usage,
    )


class ResponsesStreamParser(StreamParser):
    """A Responses stream's terminal event carries the whole response; keep only that."""

    def __init__(self) -> None:
        self.response: dict = {}

    def feed(self, event: dict) -> None:
        if event.get("type") in ResponsesDialect.terminal_events:
            self.response = event["response"]

    def finish(self) -> dict:
        return self.response


class ResponsesDialect(Dialect):
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
    max_tokens_keys = ("max_output_tokens",)
    effort_path = ("reasoning", "effort")
    routes = ("/v1/responses",)
    upstream_path = "/responses"
    # A Responses client (e.g. codex) ends its turn on its final event, before the `[DONE]`.
    terminal_events = frozenset(
        {"response.completed", "response.incomplete", "response.failed"}
    )

    def parse_request(self, body: RawRequest) -> tuple[Request, list[Setter | None]]:
        prompt: Messages = []
        setters: list[Setter | None] = []

        def add(message: Message, setter: Setter | None = None) -> None:
            prompt.append(message)
            setters.append(setter)

        if instructions := body.get("instructions"):
            add(SystemMessage(content=instructions))
        raw = body.get("input")
        if isinstance(raw, str):
            add(UserMessage(content=raw), partial(_write_input, body))
        items = raw if isinstance(raw, list) else []
        run: list[dict] = []  # the current run of assistant-side items
        for item in items:
            role = item.get("role")
            assistant = (
                role == "assistant"
                or role is None
                and not (item.get("type") or "").endswith(("_output", "_response"))
            )
            if run and not assistant:
                add(fold_assistant(run))
                run = []
            if assistant:
                run.append(item)
            elif role in ("system", "developer"):
                add(SystemMessage(content=parse_content(item.get("content"))))
            elif item.get("output") is not None:
                add(
                    ToolMessage(
                        tool_call_id=item.get("call_id") or item.get("id") or "",
                        content=_output_content(item["output"]),
                    ),
                    partial(_write_output, item),
                )
            elif "content" in item:
                add(
                    UserMessage(content=parse_content(item["content"])),
                    partial(_write_content, item),
                )
            else:
                # A protocol item with no message body (an MCP approval, a tool search
                # result): recorded verbatim, as the result of its call when it has one.
                native = [NativeContentPart(native=item)]
                call_id = item.get("call_id")
                add(
                    ToolMessage(tool_call_id=call_id, content=native)
                    if call_id
                    else UserMessage(content=native),
                    partial(_write_item, item),
                )
        if run:
            add(fold_assistant(run))
        tools = []
        declarations = []
        for item in items:
            if item.get("type") in ("additional_tools", "tool_search_output"):
                declarations.extend(item.get("tools") or [])
        # Current declarations take precedence over definitions replayed in history.
        declarations.extend(body.get("tools") or [])
        for group in declarations:
            namespace = group["name"] if group.get("type") == "namespace" else None
            for tool in group["tools"] if namespace else [group]:
                if tool.get("type") == "mcp":
                    # Connection credentials belong in the native request, not the trace.
                    tool = {
                        key: value
                        for key, value in tool.items()
                        if key.lower() not in ("authorization", "headers")
                    }
                tools.append(
                    Tool.model_validate(
                        tool
                        | {
                            "name": tool.get("name")
                            or tool.get("server_label")
                            or tool.get("type"),
                            "namespace": namespace,
                            "description": tool.get("description") or "",
                            "parameters": tool.get("parameters") or {},
                        }
                    )
                )
        tools = list({(t.namespace, t.name, t.type): t for t in tools}.values())
        return Request(messages=prompt, tools=tools or None), setters

    def parse_response(self, response: dict) -> Response:
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
