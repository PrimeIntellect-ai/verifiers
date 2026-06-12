"""The OpenAI Responses dialect (codex and friends).

Request parsing walks the `input` items, grouping each run of assistant-side items
(reasoning / assistant message / function_call) into one typed assistant message whose
`provider_state` carries the raw items (the same continuation state the Responses
egress client replays verbatim). Server-side statefulness (`previous_response_id`)
is not emulated: the relay path's endpoint owns it, the translate path rejects it.
"""

import json
import time

from openai.types.responses import Response as OpenAIResponse

from verifiers.v1.clients.openai_responses import response_from_wire
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

FINAL_EVENTS = ("response.completed", "response.incomplete", "response.failed")
ASSISTANT_ITEMS = ("reasoning", "function_call")


def parse_content(content) -> str | list[ContentPart]:
    if isinstance(content, str):
        return content
    parts: list[ContentPart] = []
    for part in content:
        kind = part.get("type")
        if kind in ("input_text", "output_text"):
            parts.append(TextContentPart(text=part.get("text", "")))
        elif kind == "input_image":
            parts.append(
                ImageUrlContentPart(
                    image_url=ImageUrlSource(
                        url=part.get("image_url", ""),
                        detail=part.get("detail")
                        if part.get("detail") != "auto"
                        else None,
                    )
                )
            )
    return parts


def fold_assistant(items: list[dict]) -> AssistantMessage:
    """One run of assistant-side items -> one typed assistant message, raw items kept
    as `provider_state` (the Responses egress client replays them verbatim)."""
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
            content += (
                item["content"]
                if isinstance(item.get("content"), str)
                else "".join(
                    p.get("text", "")
                    for p in item.get("content") or []
                    if p.get("type") in ("input_text", "output_text")
                )
            )
    return AssistantMessage(
        content=content or None,
        reasoning_content="\n".join(r for r in reasoning if r) or None,
        tool_calls=calls or None,
        provider_state=items,
    )


class ResponsesDialect(Dialect):
    name = "responses"
    route = "/v1/responses"

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
            assistant = item.get("type") in ASSISTANT_ITEMS or (
                item.get("role") == "assistant"
            )
            if run and not assistant:
                prompt.append(fold_assistant(run))
                run = []
            if assistant:
                run.append(item)
            elif item.get("type") == "function_call_output":
                prompt.append(
                    ToolMessage(
                        tool_call_id=item.get("call_id", ""),
                        content=item.get("output", "")
                        if isinstance(item.get("output"), str)
                        else json.dumps(item.get("output")),
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

    def parse_response(self, raw: dict) -> Response:
        return response_from_wire(OpenAIResponse.model_validate(raw))

    def parse_stream(self, raw: bytes) -> Response:
        """The terminal event (`response.completed` / `.incomplete` / `.failed`)
        carries the full response object — parse that."""
        for event in reversed(iter_sse(raw)):
            if event.get("type") in FINAL_EVENTS:
                return self.parse_response(event["response"])
        raise ValueError("Responses stream ended without a terminal event")

    def serialize_response(self, response: Response, model: str) -> dict:
        message = response.message
        output: list[dict] = []
        if message.reasoning_content:
            output.append(
                {
                    "type": "reasoning",
                    "id": "rs_vf-intercept",
                    "summary": [
                        {"type": "summary_text", "text": message.reasoning_content}
                    ],
                }
            )
        if message.content is not None:
            output.append(
                {
                    "type": "message",
                    "id": "msg_vf-intercept",
                    "role": "assistant",
                    "status": "completed",
                    "content": [
                        {
                            "type": "output_text",
                            "text": message.content,
                            "annotations": [],
                        }
                    ],
                }
            )
        for index, call in enumerate(message.tool_calls or []):
            output.append(
                {
                    "type": "function_call",
                    "id": f"fc_vf-intercept-{index}",
                    "call_id": call.id,
                    "name": call.name,
                    "arguments": call.arguments,
                    "status": "completed",
                }
            )
        incomplete = response.finish_reason == "length"
        return {
            "id": response.id or "resp_vf-intercept",
            "object": "response",
            "created_at": response.created or int(time.time()),
            "model": response.model or model,
            "status": "incomplete" if incomplete else "completed",
            "incomplete_details": {"reason": "max_output_tokens"}
            if incomplete
            else None,
            "output": output,
            "parallel_tool_calls": True,
            "tool_choice": "auto",
            "tools": [],
            "error": None,
            "usage": {
                "input_tokens": response.usage.prompt_tokens,
                "output_tokens": response.usage.completion_tokens,
                "total_tokens": response.usage.total_tokens,
                "input_tokens_details": {"cached_tokens": 0},
                "output_tokens_details": {"reasoning_tokens": 0},
            }
            if response.usage
            else None,
        }

    def serialize_stream(self, response: Response, model: str) -> bytes:
        """A minimal valid Responses event stream: `response.created`, then the
        terminal event carrying the full response."""
        full = self.serialize_response(response, model)
        created = sse(
            {
                "type": "response.created",
                "sequence_number": 0,
                "response": {**full, "status": "in_progress", "output": []},
            },
            "response.created",
        )
        final = (
            "response.incomplete"
            if full["status"] == "incomplete"
            else ("response.completed")
        )
        return created + sse(
            {"type": final, "sequence_number": 1, "response": full}, final
        )
