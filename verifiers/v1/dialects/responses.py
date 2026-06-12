"""The OpenAI Responses dialect.

Translates the OpenAI Responses API wire format into vf types: requests (`parse_request`,
the `input` item list -> vf prompt) and responses (`parse_response`, the `output` item list ->
a vf `Response`). Responses clients (codex) only speak streaming, so this dialect sets
`streams = True`: the proxy forces `stream` off upstream (`apply_overrides`) to fetch the whole
response unary, and the interception server replays it as SSE via `stream_events`.

Parsing mirrors the v0 responses client: `output` carries `message` (assistant `output_text`),
`reasoning` (summary/content), and `function_call` items; `input` carries `message`,
`function_call`, and `function_call_output` items.
"""

from collections.abc import Iterable
from typing import Any

from pydantic import BaseModel, ConfigDict

from verifiers.v1.dialects.base import Dialect
from verifiers.v1.types import (
    AssistantMessage,
    FinishReason,
    Message,
    Messages,
    Response,
    SamplingConfig,
    SystemMessage,
    Tool,
    ToolCall,
    ToolMessage,
    Usage,
    UserMessage,
)

# Sampling knobs the eval owns: dropped from the program's request before the eval's are applied
# (see `apply_overrides`). Responses names completion budget `max_output_tokens`; the chat
# aliases are included so a `max_tokens` from the eval's SamplingConfig maps cleanly.
_SAMPLING_KEYS = frozenset(
    {"temperature", "top_p", "max_tokens", "max_completion_tokens", "max_output_tokens"}
)


class ResponsesResponse(BaseModel):
    """A permissive view of a Responses API response — only the fields the trace needs, typed
    loosely (output/usage as dicts) with `extra='allow'` so any provider's extra fields survive.
    The program gets the verbatim raw back (`Response.raw`), so this is just for parsing."""

    model_config = ConfigDict(extra="allow")
    id: str = ""
    created_at: float = 0.0
    model: str = ""
    status: str | None = None
    output: list[dict[str, Any]] = []
    usage: dict[str, Any] | None = None
    incomplete_details: dict[str, Any] | None = None


def _content_text(content: Any) -> str:
    """Flatten Responses message content (a list of `input_text`/`output_text` parts, or a
    plain string) to text."""
    if isinstance(content, list):
        return "".join(
            p.get("text", "") for p in content if isinstance(p, dict) and "text" in p
        )
    return content or ""


def parse_tools(raw: list[dict] | None) -> list[Tool] | None:
    """Responses tools are flat (`{type, name, description, parameters}`), unlike chat's nested
    `function` block."""
    if not raw:
        return None
    return [
        Tool(
            name=t["name"],
            description=t.get("description", ""),
            parameters=t.get("parameters", {}),
            strict=t.get("strict"),
        )
        for t in raw
        if t.get("type", "function") == "function" and "name" in t
    ]


def parse_input_item(item: dict) -> Message | None:
    """One Responses `input` item -> a typed Message (None for items with no prompt role, e.g.
    reasoning)."""
    item_type = item.get("type", "message")
    if item_type == "function_call":
        return AssistantMessage(
            tool_calls=[
                ToolCall(
                    id=item.get("call_id", ""),
                    name=item.get("name", ""),
                    arguments=item.get("arguments", ""),
                )
            ]
        )
    if item_type == "function_call_output":
        return ToolMessage(
            tool_call_id=item.get("call_id", ""),
            content=_content_text(item.get("output")),
        )
    if item_type != "message":
        return None  # reasoning / other items don't map to a prompt message
    role = item.get("role")
    text = _content_text(item.get("content"))
    if role == "system":
        return SystemMessage(content=text)
    if role == "assistant":
        return AssistantMessage(content=text or None)
    return UserMessage(content=text)


class ResponsesDialect(Dialect[dict, ResponsesResponse]):
    """The OpenAI Responses wire format (codex)."""

    routes = ("/v1/responses",)
    upstream_path = "/responses"
    streams = True
    response_type = ResponsesResponse

    def parse_request(self, body: dict) -> tuple[Messages, list[Tool] | None]:
        messages: list[Message] = []
        instructions = body.get("instructions")
        if isinstance(instructions, str) and instructions:
            messages.append(SystemMessage(content=instructions))
        for item in body.get("input", []):
            if isinstance(item, dict) and (message := parse_input_item(item)):
                messages.append(message)
        return messages, parse_tools(body.get("tools"))

    def parse_response(self, response: ResponsesResponse) -> Response:
        content_chunks: list[str] = []
        reasoning_chunks: list[str] = []
        tool_calls: list[ToolCall] = []
        for item in response.output:
            item_type = item.get("type")
            if item_type == "message":
                for part in item.get("content", []) or []:
                    if part.get("type") == "output_text" and part.get("text"):
                        content_chunks.append(part["text"])
                    elif part.get("type") == "refusal" and part.get("refusal"):
                        content_chunks.append(part["refusal"])
            elif item_type == "reasoning":
                for block in (item.get("summary") or []) + (item.get("content") or []):
                    if isinstance(block, dict) and isinstance(block.get("text"), str):
                        reasoning_chunks.append(block["text"])
            elif item_type == "function_call":
                tool_calls.append(
                    ToolCall(
                        id=item.get("call_id", ""),
                        name=item.get("name", ""),
                        arguments=item.get("arguments", ""),
                    )
                )
        usage = None
        if response.usage:
            usage = Usage(
                prompt_tokens=response.usage.get("input_tokens", 0),
                completion_tokens=response.usage.get("output_tokens", 0),
            )
        finish: FinishReason = (
            "tool_calls"
            if tool_calls
            else "length"
            if response.status == "incomplete" or response.incomplete_details
            else "stop"
            if response.status == "completed"
            else None
        )
        return Response(
            id=response.id,
            created=int(response.created_at),
            model=response.model,
            message=AssistantMessage(
                content="".join(content_chunks) or None,
                reasoning_content="\n".join(reasoning_chunks) or None,
                tool_calls=tool_calls or None,
            ),
            finish_reason=finish,
            usage=usage,
        )

    def apply_overrides(self, body: dict, model: str, sampling: SamplingConfig) -> dict:
        # Forward verbatim except what the eval owns (model overlay, sampling authoritative) and
        # `stream`, which is forced off so the provider returns a unary body to fake-stream.
        overrides: dict = {}
        s = sampling.model_dump(exclude_none=True)
        if "temperature" in s:
            overrides["temperature"] = s["temperature"]
        if "top_p" in s:
            overrides["top_p"] = s["top_p"]
        if s.get("max_tokens") is not None:
            overrides["max_output_tokens"] = s["max_tokens"]
        steered = {
            k: v for k, v in body.items() if k not in _SAMPLING_KEYS and k != "stream"
        }
        return {**steered, "model": model, "stream": False, **overrides}

    def serialize_response(self, response: Response, model: str) -> dict:
        """A vf `Response` -> a Responses API response dict (for the renderer / fake-stream when
        there's no raw to relay — the proxy path relays `Response.raw` instead)."""
        output: list[dict] = []
        if response.message.reasoning_content:
            output.append(
                {
                    "id": f"rs_{response.id}",
                    "type": "reasoning",
                    "summary": [
                        {
                            "type": "summary_text",
                            "text": response.message.reasoning_content,
                        }
                    ],
                }
            )
        if response.message.content:
            output.append(
                {
                    "id": f"msg_{response.id}",
                    "type": "message",
                    "role": "assistant",
                    "status": "completed",
                    "content": [
                        {
                            "type": "output_text",
                            "text": response.message.content,
                            "annotations": [],
                        }
                    ],
                }
            )
        for call in response.message.tool_calls or []:
            output.append(
                {
                    "id": call.id,
                    "type": "function_call",
                    "call_id": call.id,
                    "name": call.name,
                    "arguments": call.arguments,
                    "status": "completed",
                }
            )
        usage = (
            {
                "input_tokens": response.usage.prompt_tokens,
                "output_tokens": response.usage.completion_tokens,
                "total_tokens": response.usage.total_tokens,
            }
            if response.usage
            else None
        )
        return {
            "id": response.id or "vf-intercept",
            "object": "response",
            "created_at": float(response.created),
            "status": "completed",
            "model": response.model or model,
            "output": output,
            "usage": usage,
        }

    def extend(self, body: dict, completion: dict, user_messages: Messages) -> dict:
        # Append the model's turn (its verbatim output items, so reasoning survives) and the
        # simulator's user turn(s) to the Responses `input` list.
        items = [*body.get("input", []), *completion.get("output", [])]
        for message in user_messages:
            items.append(
                {
                    "type": "message",
                    "role": message.role,
                    "content": [
                        {"type": "input_text", "text": _content_text(message.content)}
                    ],
                }
            )
        return {**body, "input": items}

    def stream_events(self, completion: dict) -> Iterable[dict]:
        # Replay the buffered Responses object as the SSE sequence codex consumes: `created`,
        # then each output item `added` + `done` (codex reads items from `output_item.done`),
        # then `completed` carrying the whole response (codex reads usage + id from it). The
        # stream MUST end with `response.completed` or codex errors ("stream closed before
        # response.completed").
        skeleton = {
            "id": completion.get("id", ""),
            "object": "response",
            "status": "in_progress",
            "output": [],
        }
        yield {"type": "response.created", "response": skeleton}
        for index, item in enumerate(completion.get("output", [])):
            yield {
                "type": "response.output_item.added",
                "output_index": index,
                "item": item,
            }
            yield {
                "type": "response.output_item.done",
                "output_index": index,
                "item": item,
            }
        yield {"type": "response.completed", "response": completion}
