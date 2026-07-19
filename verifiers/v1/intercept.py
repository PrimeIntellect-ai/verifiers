"""Task-authored interception of every model exchange (`@vf.intercept`).

An `@intercept` handler on a `Task` sees each model exchange at the one choke point all
model traffic funnels through (the interception server), in both directions:

- **request** — the harness's native request body on its way upstream (carrying the tool
  results and provider-tool definitions), and
- **response** — the provider's native response object on its way back (carrying the tool
  calls and provider-side tool output such as web-search results).

The handler mutates `exchange.raw` in place and returns an action: a `Message` (block the
response — its tool calls are dropped and the model gets the message's text as the answer
instead, so the exchange continues) or `Terminate` (end the rollout, optionally with a
negative reward). Returning None after mutating `exchange.raw` is auto-detected and logged
as a rewrite. Every action taken is recorded on the trace as an `InterceptRecord`.

The `strip_*`/`drop_*` helpers do the wire surgery across the chat / anthropic / responses
shapes; the matcher receives a tool type/name string (e.g. `"web_search_call"`,
`"web_search_20250305"`) and the returned list holds the matched labels, for logging.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable
from typing import TYPE_CHECKING, Any, Literal

from pydantic import BaseModel

from verifiers.v1.types import AssistantMessage, Messages

if TYPE_CHECKING:
    from verifiers.v1.dialects import Dialect
    from verifiers.v1.trace import Trace

Direction = Literal["request", "response"]


class InterceptAction(BaseModel):
    """Base for what an `@intercept` handler returns (None = no opinion, a `Message` =
    block the response and answer the model with its text)."""

    reason: str = ""


class Terminate(InterceptAction):
    """End the rollout now; optionally record a (negative) reward."""

    reason: str = "terminated by interception"
    reward: float | None = None


class InterceptRecord(BaseModel):
    """One action an `@intercept` handler took on an exchange, recorded on the trace."""

    direction: Direction
    handler: str
    action: str  # "block" | "rewrite" | "terminate"
    target: str = ""
    reason: str = ""
    before: str = ""  # compact snippet of what was removed/changed
    after: str = ""


class InterceptExchange:
    """One intercepted model exchange, handed to each `@intercept` handler in turn.

    `raw` is the native wire JSON — the request body inbound, the response object
    outbound — mutated in place. `prompt`/`message` are read-only typed views re-derived
    from `raw` via the dialect on each access, so they always reflect the current
    (possibly already mutated) wire state."""

    def __init__(
        self,
        direction: Direction,
        raw: dict[str, Any],
        trace: "Trace",
        dialect: "Dialect",
    ) -> None:
        self.direction = direction
        self.raw = raw
        self.trace = trace
        self._dialect = dialect

    def __repr__(self) -> str:
        return f"InterceptExchange(direction={self.direction!r})"

    @property
    def prompt(self) -> Messages | None:
        """Request side: the typed prompt messages parsed from `raw`; None outbound."""
        if self.direction != "request":
            return None
        return self._dialect.parse_request(self.raw)[0]

    @property
    def message(self) -> AssistantMessage | None:
        """Response side: the typed assistant message parsed from `raw`; None inbound."""
        if self.direction != "response":
            return None
        response = self._dialect.parse_response(
            self._dialect.validate_response(self.raw)
        )
        return response.message

    def digest(self) -> bytes:
        """A stable digest of `raw`, compared around a handler to detect a silent rewrite."""
        canonical = json.dumps(
            self.raw, sort_keys=True, separators=(",", ":"), default=str
        )
        return hashlib.blake2b(canonical.encode(), digest_size=16).digest()


def _snippet(value: Any) -> str:
    """A compact one-line JSON snippet of what an interception removed, truncated so a
    record stays small even when the removed payload (a tool result, a web search) is not."""
    return json.dumps(value, separators=(",", ":"), default=str)[:500]


def _matches(matcher: Callable[[str], bool], *labels: Any) -> bool:
    """Match on every label a wire entry carries (its `type` and/or `name`)."""
    return any(matcher(label) for label in labels if isinstance(label, str))


def _response_kind(raw: dict) -> str | None:
    """Sniff which dialect produced a native response object."""
    if isinstance(raw.get("choices"), list):
        return "chat"
    if isinstance(raw.get("output"), list):
        return "responses"
    if isinstance(raw.get("content"), list):
        return "anthropic"
    return None


# Anthropic-only content block types — what tells its `messages` apart from a chat body.
_ANTHROPIC_BLOCKS = frozenset(
    {
        "tool_use",
        "tool_result",
        "thinking",
        "redacted_thinking",
        "server_tool_use",
        "web_search_tool_result",
        "image",
    }
)


def _request_kind(raw: dict) -> str | None:
    """Sniff which dialect produced a native request body. Chat and anthropic both carry
    `messages`; chat gives itself away with system/tool roles, `tool_calls`, or
    function-wrapped tool defs, anthropic with its typed content blocks and tool defs.
    An ambiguous body (plain user/assistant text only) holds no provider items, so
    defaulting to chat is safe — there is nothing to strip either way."""
    if "input" in raw:
        return "responses"
    messages = raw.get("messages")
    if not isinstance(messages, list):
        return None
    for message in messages:
        if not isinstance(message, dict):
            continue
        if (
            message.get("role") in ("system", "tool")
            or "tool_calls" in message
            or "tool_call_id" in message
        ):
            return "chat"
        content = message.get("content")
        for block in content if isinstance(content, list) else []:
            if isinstance(block, dict) and block.get("type") in _ANTHROPIC_BLOCKS:
                return "anthropic"
    for tool in raw.get("tools") or []:
        if isinstance(tool, dict):
            if "function" in tool or "custom" in tool:
                return "chat"
            if "input_schema" in tool or tool.get("type"):
                return "anthropic"
    return "chat"


def drop_response_tool_calls(raw: dict, reason: str) -> list[str]:
    """Remove the model's tool calls from a native RESPONSE object, keeping/appending
    assistant text telling the model the call was blocked with `reason`. Returns the
    dropped tool names."""
    notice = f"Tool call blocked: {reason}"
    dropped: list[str] = []
    kind = _response_kind(raw)
    if kind == "chat":
        for choice in raw.get("choices") or []:
            message = choice.get("message")
            if not isinstance(message, dict):
                continue
            calls = message.pop("tool_calls", None) or []
            if not calls:
                continue
            dropped.extend(
                (call.get("function") or {}).get("name") or ""
                for call in calls
                if isinstance(call, dict)
            )
            content = message.get("content")
            if isinstance(content, str):
                message["content"] = f"{content}\n\n{notice}" if content else notice
            elif isinstance(content, list):
                content.append({"type": "text", "text": notice})
            else:
                message["content"] = notice
            if choice.get("finish_reason") == "tool_calls":
                choice["finish_reason"] = "stop"
    elif kind == "responses":
        kept = []
        for item in raw.get("output") or []:
            if isinstance(item, dict) and item.get("type") in (
                "function_call",
                "custom_tool_call",
            ):
                dropped.append(item.get("name") or "")
            else:
                kept.append(item)
        if dropped:
            kept.append(
                {
                    "type": "message",
                    "id": "msg_blocked",
                    "role": "assistant",
                    "status": "completed",
                    "content": [{"type": "output_text", "text": notice}],
                }
            )
            raw["output"] = kept
    elif kind == "anthropic":
        kept = []
        for block in raw.get("content") or []:
            if isinstance(block, dict) and block.get("type") == "tool_use":
                dropped.append(block.get("name") or "")
            else:
                kept.append(block)
        if dropped:
            kept.append({"type": "text", "text": notice})
            raw["content"] = kept
            if raw.get("stop_reason") == "tool_use":
                raw["stop_reason"] = "end_turn"
    return dropped


def strip_request_tools(raw: dict, matcher: Callable[[str], bool]) -> list[str]:
    """Remove provider-tool DEFINITIONS (web search and friends — the `tools` entries
    that are not model-invoked function/custom tools) whose type/name matches. Returns
    the matched labels."""
    tools = raw.get("tools")
    if not isinstance(tools, list):
        return []
    kept, stripped = [], []
    for tool in tools:
        if isinstance(tool, dict) and not (
            # A model-invoked tool: chat's function/custom wrappers, a bare function or
            # custom entry (responses), or anthropic's input_schema tool.
            "function" in tool
            or "custom" in tool
            or "input_schema" in tool
            or tool.get("type") in ("function", "custom")
        ):
            if _matches(matcher, tool.get("type"), tool.get("name")):
                stripped.append(tool.get("type") or tool.get("name") or "")
                continue
        kept.append(tool)
    if stripped:
        raw["tools"] = kept
    return stripped


def strip_history_items(raw: dict, matcher: Callable[[str], bool]) -> list[str]:
    """Remove provider-tool RESULT/USE items from a request body's conversation history
    (responses: `input` items like `web_search_call`; anthropic: content blocks like
    `server_tool_use`/`web_search_tool_result`). Chat history carries no provider items.
    Returns the matched labels."""
    kind = _request_kind(raw)
    stripped: list[str] = []
    if kind == "responses":
        kept = []
        for item in raw.get("input") or []:
            if isinstance(item, dict) and _matches(
                matcher, item.get("type"), item.get("name")
            ):
                stripped.append(item.get("type") or item.get("name") or "")
            else:
                kept.append(item)
        if stripped:
            raw["input"] = kept
    elif kind == "anthropic":
        messages = []
        for message in raw.get("messages") or []:
            content = message.get("content") if isinstance(message, dict) else None
            if not isinstance(content, list):
                messages.append(message)
                continue
            kept_blocks = []
            for block in content:
                if isinstance(block, dict) and _matches(
                    matcher, block.get("type"), block.get("name")
                ):
                    stripped.append(block.get("type") or block.get("name") or "")
                else:
                    kept_blocks.append(block)
            # Anthropic rejects an empty content array; a message left with none goes.
            if kept_blocks:
                message["content"] = kept_blocks
                messages.append(message)
        if stripped:
            raw["messages"] = messages
    return stripped


def strip_response_items(raw: dict, matcher: Callable[[str], bool]) -> list[str]:
    """Remove provider-tool output items/blocks from a native RESPONSE object
    (responses: `output` items like `web_search_call`; anthropic: content blocks like
    `server_tool_use`/`web_search_tool_result`). Returns the matched labels."""
    kind = _response_kind(raw)
    items = raw.get("output") if kind == "responses" else raw.get("content")
    if kind not in ("responses", "anthropic") or not isinstance(items, list):
        return []
    kept, stripped = [], []
    for item in items:
        if isinstance(item, dict) and _matches(
            matcher, item.get("type"), item.get("name")
        ):
            stripped.append(item.get("type") or item.get("name") or "")
        else:
            kept.append(item)
    if stripped:
        raw["output" if kind == "responses" else "content"] = kept
    return stripped


def _response_tool_call_items(raw: dict) -> list[dict]:
    """The tool-call entries of a native response object (for a record's `before` snippet)."""
    kind = _response_kind(raw)
    if kind == "chat":
        items = []
        for choice in raw.get("choices") or []:
            message = choice.get("message")
            if isinstance(message, dict):
                items.extend(
                    call
                    for call in message.get("tool_calls") or []
                    if isinstance(call, dict)
                )
        return items
    if kind == "responses":
        return [
            item
            for item in raw.get("output") or []
            if isinstance(item, dict)
            and item.get("type") in ("function_call", "custom_tool_call")
        ]
    if kind == "anthropic":
        return [
            block
            for block in raw.get("content") or []
            if isinstance(block, dict) and block.get("type") == "tool_use"
        ]
    return []
