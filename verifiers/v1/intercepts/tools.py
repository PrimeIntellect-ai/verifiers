"""Ready-made checks for `@vf.intercept` handlers: fuzzy tool matching and provider-tool
stripping. Each takes the `exchange` and returns a verdict — the handler decides what to
do with it (answer with a `Message`, `Terminate`, or nothing)."""

from __future__ import annotations

from typing import TYPE_CHECKING

from verifiers.v1.intercept import (
    InterceptRecord,
    strip_history_items,
    strip_request_tools,
    strip_response_items,
)
from verifiers.v1.intercepts.match import match_tool

if TYPE_CHECKING:
    from verifiers.v1.intercept import InterceptExchange


def match_tool_calls(exchange: InterceptExchange, *patterns: str) -> bool:
    """Response side: True if any of the response's tool calls fuzzily matches a pattern
    (`match_tool`). False on the request side or when there are no tool calls."""
    if exchange.direction != "response":
        return False
    message = exchange.message
    return any(
        match_tool(call.name, *patterns)
        for call in (message.tool_calls if message else None) or []
    )


def strip_provider_tools(exchange: InterceptExchange, *patterns: str) -> list[str]:
    """Strip provider tools (web search and friends) matching any pattern, in both
    directions — request-side tool definitions and history items, response-side output
    items — mutating `exchange.raw` in place. Records what was stripped on the trace and
    returns the matched labels (empty = nothing was stripped)."""

    def matcher(tool: str) -> bool:
        return match_tool(tool, *patterns)

    if exchange.direction == "request":
        matched = strip_request_tools(exchange.raw, matcher) + strip_history_items(
            exchange.raw, matcher
        )
    else:
        matched = strip_response_items(exchange.raw, matcher)
    if matched:
        exchange.trace.record_interception(
            InterceptRecord(
                direction=exchange.direction,
                handler="strip_provider_tools",
                action="rewrite",
                target=", ".join(matched),
                reason=f"stripped provider tools: {matched}",
            )
        )
    return matched


__all__ = ["match_tool_calls", "strip_provider_tools"]
