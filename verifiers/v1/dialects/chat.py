"""The OpenAI chat-completions dialect (the lingua franca every aggregator speaks).

wire -> vf request parsing moved here from the interception server; response parsing
reuses the chat client's `response_from_wire`. The serializers produce the
`chat.completion` (and chunked SSE) shapes the program's OpenAI SDK expects.
"""

import time

from openai.types.chat import ChatCompletion

from verifiers.v1.clients.openai import response_from_wire
from verifiers.v1.dialects.base import Dialect, iter_sse, sse
from verifiers.v1.types import (
    AssistantMessage,
    Message,
    Messages,
    Response,
    SystemMessage,
    Tool,
    ToolCall,
    ToolMessage,
    UserMessage,
    content_to_parts,
)


def content_text(content) -> str:
    """Flatten content to text — for roles that never carry images (assistant, tool)."""
    if isinstance(content, list):
        return "".join(p.get("text", "") for p in content if isinstance(p, dict))
    return content or ""


def parse_message(raw: dict) -> Message:
    """An OpenAI request message dict -> a typed Message. User/system bodies keep their
    image parts (multimodal ingress); assistant/tool bodies flatten to text."""
    role = raw.get("role")
    content = raw.get("content")
    if role == "system":
        return SystemMessage(content=content_to_parts(content))
    if role == "tool":
        return ToolMessage(
            tool_call_id=raw.get("tool_call_id", ""), content=content_text(content)
        )
    if role == "assistant":
        calls = [
            ToolCall(
                id=c["id"],
                name=c["function"]["name"],
                arguments=c["function"]["arguments"],
            )
            for c in (raw.get("tool_calls") or [])
        ] or None
        return AssistantMessage(
            content=content_text(content) or None,
            reasoning_content=raw.get("reasoning_content") or raw.get("reasoning"),
            tool_calls=calls,
            provider_state=raw.get("reasoning_details") or raw.get("provider_state"),
        )
    return UserMessage(content=content_to_parts(content))


def parse_tools(raw: list[dict] | None) -> list[Tool] | None:
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
    ]


def serialize_completion(response: Response, model: str) -> dict:
    """A `Response` -> an OpenAI chat.completion dict the program's SDK expects."""
    message: dict = {"role": "assistant", "content": response.message.content}
    if response.message.reasoning_content is not None:
        message["reasoning_content"] = response.message.reasoning_content
    if response.message.provider_state is not None:
        message["reasoning_details"] = response.message.provider_state
    if response.message.tool_calls:
        message["tool_calls"] = [
            {
                "id": c.id,
                "type": "function",
                "function": {"name": c.name, "arguments": c.arguments},
            }
            for c in response.message.tool_calls
        ]
    usage = (
        {
            "prompt_tokens": response.usage.prompt_tokens,
            "completion_tokens": response.usage.completion_tokens,
            "total_tokens": response.usage.total_tokens,
        }
        if response.usage
        else None
    )
    return {
        "id": response.id or "vf-intercept",
        "object": "chat.completion",
        "created": response.created,
        "model": response.model or model,
        "choices": [
            {
                "index": 0,
                "message": message,
                "finish_reason": response.finish_reason or "stop",
            }
        ],
        "usage": usage,
    }


class ChatDialect(Dialect):
    name = "chat"
    route = "/v1/chat/completions"

    def parse_request(self, body: dict) -> tuple[Messages, list[Tool] | None]:
        return (
            [parse_message(m) for m in body.get("messages", [])],
            parse_tools(body.get("tools")),
        )

    def parse_response(self, raw: dict) -> Response:
        return response_from_wire(ChatCompletion.model_validate(raw))

    def parse_stream(self, raw: bytes) -> Response:
        """Assemble `chat.completion.chunk` deltas into one completion, then parse it."""
        chunks = iter_sse(raw)
        message: dict = {"role": "assistant", "content": None}
        tool_calls: dict[int, dict] = {}
        finish_reason = None
        usage = None
        for chunk in chunks:
            usage = chunk.get("usage") or usage
            for choice in chunk.get("choices") or []:
                if choice.get("index", 0) != 0:
                    continue
                finish_reason = choice.get("finish_reason") or finish_reason
                delta = choice.get("delta") or {}
                for key in ("content", "reasoning_content", "reasoning"):
                    if delta.get(key) is not None:
                        message[key] = (message.get(key) or "") + delta[key]
                for tc in delta.get("tool_calls") or []:
                    slot = tool_calls.setdefault(
                        tc.get("index", 0),
                        {"type": "function", "function": {"name": "", "arguments": ""}},
                    )
                    slot["id"] = tc.get("id") or slot.get("id", "")
                    fn = tc.get("function") or {}
                    if fn.get("name"):
                        slot["function"]["name"] = fn["name"]
                    slot["function"]["arguments"] += fn.get("arguments") or ""
        if tool_calls:
            message["tool_calls"] = [tool_calls[i] for i in sorted(tool_calls)]
        head = chunks[0] if chunks else {}
        return self.parse_response(
            {
                "id": head.get("id", "vf-intercept"),
                "object": "chat.completion",
                "created": head.get("created", int(time.time())),
                "model": head.get("model", ""),
                "choices": [
                    {
                        "index": 0,
                        "message": message,
                        "finish_reason": finish_reason or "stop",
                    }
                ],
                "usage": usage,
            }
        )

    def serialize_response(self, response: Response, model: str) -> dict:
        return serialize_completion(response, model)

    def extend_request(
        self, body: dict, completion: dict, user_messages: Messages
    ) -> dict:
        """Extend a relayed request with the last completion's verbatim assistant
        message plus the user simulator's injected turn(s) — the framework-authored
        continuation request the relay path re-sends."""
        from verifiers.v1.clients.openai import message_to_wire

        return {
            **body,
            "messages": [
                *body.get("messages", []),
                completion["choices"][0]["message"],
                *(message_to_wire(m) for m in user_messages),
            ],
        }

    def serialize_stream(self, response: Response, model: str) -> bytes:
        """The completed response as a minimal valid chunk stream: one delta carrying
        the whole message, a usage chunk, then `[DONE]`."""
        completion = serialize_completion(response, model)
        delta = completion["choices"][0]["message"]
        if "tool_calls" in delta:
            delta["tool_calls"] = [
                {**tc, "index": i} for i, tc in enumerate(delta["tool_calls"])
            ]
        chunk = {
            "id": completion["id"],
            "object": "chat.completion.chunk",
            "created": completion["created"],
            "model": completion["model"],
            "choices": [
                {
                    "index": 0,
                    "delta": delta,
                    "finish_reason": completion["choices"][0]["finish_reason"],
                }
            ],
            "usage": completion["usage"],
        }
        return sse(chunk) + b"data: [DONE]\n\n"
