"""The Anthropic Messages dialect (claude-code and friends).

Request parsing maps Anthropic content blocks onto the typed messages; responses are
parsed with the Anthropic client's `response_from_wire`. The serializers (translate
path) hand a typed `Response` back in Anthropic shape — reasoning is emitted as a
thinking block so the program displays it and echoes it back, where `parse_request`
recovers it into `reasoning_content` (the passback some reasoning models require).
"""

import json
from collections.abc import Mapping

from anthropic.types import Message as AnthropicMessage

from verifiers.v1.clients.anthropic import response_from_wire
from verifiers.v1.dialects.base import Dialect, iter_sse, sse
from verifiers.v1.types import (
    AssistantMessage,
    ContentPart,
    ImageUrlContentPart,
    ImageUrlSource,
    Messages,
    Response,
    SystemMessage,
    TextContentPart,
    Tool,
    ToolCall,
    ToolMessage,
    UserMessage,
)

STOP_REASONS = {"stop": "end_turn", "length": "max_tokens", "tool_calls": "tool_use"}


def parse_content(content) -> str | list[ContentPart]:
    """Anthropic user-side content (text + image blocks) -> typed content parts."""
    if isinstance(content, str):
        return content
    parts: list[ContentPart] = []
    for block in content:
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


def block_text(content) -> str:
    """Flatten a tool_result's content (a string or text blocks) to text."""
    if isinstance(content, str):
        return content
    return "".join(b.get("text", "") for b in content or [] if b.get("type") == "text")


def parse_messages(body: dict) -> Messages:
    """The request's `system` + `messages` -> typed messages. Assistant turns fold their
    blocks into one message (thinking -> reasoning, tool_use -> tool calls); a user turn's
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
                )
            )
            continue
        rest = []
        for block in [] if isinstance(content, str) else content or []:
            if block.get("type") == "tool_result":
                prompt.append(
                    ToolMessage(
                        tool_call_id=block.get("tool_use_id", ""),
                        content=block_text(block.get("content")),
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


class AnthropicDialect(Dialect):
    name = "anthropic"
    route = "/v1/messages"
    aux_routes = ("/v1/messages/count_tokens",)

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

    def parse_response(self, raw: dict) -> Response:
        return response_from_wire(AnthropicMessage.model_validate(raw))

    def parse_stream(self, raw: bytes) -> Response:
        """Assemble message_start / content_block_* / message_delta events into the
        complete message, then parse it."""
        message: dict = {}
        blocks: dict[int, dict] = {}
        partial_json: dict[int, str] = {}
        for event in iter_sse(raw):
            kind = event.get("type")
            if kind == "message_start":
                message = event.get("message") or {}
            elif kind == "content_block_start":
                blocks[event["index"]] = dict(event.get("content_block") or {})
            elif kind == "content_block_delta":
                block = blocks.setdefault(event["index"], {"type": "text", "text": ""})
                delta = event.get("delta") or {}
                if delta.get("type") == "text_delta":
                    block["text"] = block.get("text", "") + delta.get("text", "")
                elif delta.get("type") == "thinking_delta":
                    block["thinking"] = block.get("thinking", "") + delta.get(
                        "thinking", ""
                    )
                elif delta.get("type") == "signature_delta":
                    block["signature"] = block.get("signature", "") + delta.get(
                        "signature", ""
                    )
                elif delta.get("type") == "input_json_delta":
                    partial_json[event["index"]] = partial_json.get(
                        event["index"], ""
                    ) + delta.get("partial_json", "")
            elif kind == "message_delta":
                message.update(
                    {
                        k: v
                        for k, v in (event.get("delta") or {}).items()
                        if v is not None
                    }
                )
                usage = {**(message.get("usage") or {}), **(event.get("usage") or {})}
                message["usage"] = usage
        for index, partial in partial_json.items():
            blocks[index]["input"] = json.loads(partial or "{}")
        message["content"] = [blocks[i] for i in sorted(blocks)]
        return self.parse_response(message)

    def serialize_response(self, response: Response, model: str) -> dict:
        message = response.message
        content: list[dict] = []
        if message.reasoning_content:
            # Emitted so the program shows reasoning and echoes it back next turn,
            # where parse_request recovers it (required passback for some models).
            content.append(
                {
                    "type": "thinking",
                    "thinking": message.reasoning_content,
                    "signature": "",
                }
            )
        if message.content:
            content.append({"type": "text", "text": message.content})
        for call in message.tool_calls or []:
            content.append(
                {
                    "type": "tool_use",
                    "id": call.id,
                    "name": call.name,
                    "input": json.loads(call.arguments or "{}"),
                }
            )
        return {
            "id": response.id or "vf-intercept",
            "type": "message",
            "role": "assistant",
            "model": response.model or model,
            "content": content,
            "stop_reason": STOP_REASONS.get(response.finish_reason or "", "end_turn"),
            "stop_sequence": None,
            "usage": {
                "input_tokens": response.usage.prompt_tokens if response.usage else 0,
                "output_tokens": response.usage.completion_tokens
                if response.usage
                else 0,
            },
        }

    def serialize_stream(self, response: Response, model: str) -> bytes:
        """The completed response as a minimal valid Anthropic event stream: each content
        block as start + one delta + stop, framed by message_start/delta/stop."""
        full = self.serialize_response(response, model)
        deltas = {
            "text": lambda b: {"type": "text_delta", "text": b["text"]},
            "thinking": lambda b: {"type": "thinking_delta", "thinking": b["thinking"]},
            "tool_use": lambda b: {
                "type": "input_json_delta",
                "partial_json": json.dumps(b["input"]),
            },
        }
        out = sse(
            {
                "type": "message_start",
                "message": {**full, "content": [], "stop_reason": None},
            },
            "message_start",
        )
        for index, block in enumerate(full["content"]):
            start = dict(block)
            if block["type"] == "text":
                start["text"] = ""
            elif block["type"] == "thinking":
                start["thinking"] = ""
            else:
                start["input"] = {}
            out += sse(
                {"type": "content_block_start", "index": index, "content_block": start},
                "content_block_start",
            )
            out += sse(
                {
                    "type": "content_block_delta",
                    "index": index,
                    "delta": deltas[block["type"]](block),
                },
                "content_block_delta",
            )
            out += sse(
                {"type": "content_block_stop", "index": index}, "content_block_stop"
            )
        out += sse(
            {
                "type": "message_delta",
                "delta": {"stop_reason": full["stop_reason"], "stop_sequence": None},
                "usage": {"output_tokens": full["usage"]["output_tokens"]},
            },
            "message_delta",
        )
        return out + sse({"type": "message_stop"}, "message_stop")

    def handle_aux(self, path: str, body: dict) -> dict:
        """count_tokens, estimated (translate path only — relayed verbatim otherwise):
        ~4 chars per token over the request's text."""
        chars = len(json.dumps(body.get("messages", []))) + len(
            json.dumps(body.get("system", ""))
        )
        return {"input_tokens": max(1, chars // 4)}
