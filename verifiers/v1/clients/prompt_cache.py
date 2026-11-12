"""Gateway-side prompt-cache markers for Anthropic models.

Anthropic has no implicit prompt cache: a request only caches content when a
block carries ``cache_control``. Verifiers relays harness-native bodies, and
many harnesses send no marker, so Claude traffic through Prime Inference or
OpenRouter pays full input price. When the caller sends no marker, mark the
static prefix (last custom tool and last system/developer block) that repeats
across rollouts; multi-turn requests also mark the latest message.

Markers default to the five-minute cache (no ``ttl``). Set
``cache_retention = "long"`` on the eval client, or ``PI_CACHE_RETENTION=long``,
to emit ``ttl: "1h"`` instead; one-hour writes cost twice as much. ``"none"``
disables injection entirely. Any caller marker wins over generation here.
"""

from __future__ import annotations

import os
from collections.abc import Mapping
from typing import Any, Literal
from urllib.parse import urlsplit

from verifiers.v1.dialects.base import Dialect

CacheRetention = Literal["none", "short", "long"]

_CACHE_MARKERS = frozenset({"cache_control", "prompt_cache_breakpoint"})
_SCHEMA_KEYS = frozenset({"input_schema", "parameters", "json_schema"})
_ANTHROPIC_BLOCK_TYPES = frozenset({"text", "image", "tool_result", "document"})


def _has_marker(value: Any) -> bool:
    return isinstance(value, Mapping) and not _CACHE_MARKERS.isdisjoint(value)


def _content_has_marker(content: Any) -> bool:
    if isinstance(content, Mapping):
        return _has_marker(content)
    if not isinstance(content, list):
        return False
    return any(isinstance(block, Mapping) and _has_marker(block) for block in content)


def _has_caller_marker(body: Mapping[str, Any]) -> bool:
    """Only scan positions where Anthropic honors cache markers.

    Tool-use inputs, tool arguments, metadata, and JSON schemas can contain a
    ``cache_control`` key without being prompt-cache markers.
    """
    if _has_marker(body):
        return True
    if _content_has_marker(body.get("system")):
        return True
    messages = body.get("messages")
    if isinstance(messages, list):
        for message in messages:
            if not isinstance(message, Mapping):
                continue
            if _has_marker(message):
                return True
            content = message.get("content")
            if _content_has_marker(content):
                return True
            if isinstance(content, list):
                for block in content:
                    if (
                        isinstance(block, Mapping)
                        and block.get("type") == "tool_result"
                        and _content_has_marker(block.get("content"))
                    ):
                        return True
    tools = body.get("tools")
    if isinstance(tools, list):
        for tool in tools:
            if _has_marker(tool):
                return True
    return False


def resolve_cache_retention(
    cache_retention: CacheRetention | None,
) -> CacheRetention:
    """Explicit setting wins, then ``PI_CACHE_RETENTION=long``, then short."""
    if cache_retention is not None:
        return cache_retention
    env = os.environ.get("PI_CACHE_RETENTION", "").strip().casefold()
    if env in {"none", "short", "long"}:
        return env
    return "short"


def _cache_marker(retention: CacheRetention) -> dict[str, str]:
    marker = {"type": "ephemeral"}
    if retention == "long":
        marker["ttl"] = "1h"
    return marker


def _is_anthropic_model(body: Mapping[str, Any]) -> bool:
    model = body.get("model")
    return isinstance(model, str) and model.casefold().startswith("anthropic/")


def _is_gateway(base_url: str) -> bool:
    hostname = urlsplit(base_url).hostname
    if hostname is None:
        return False
    host = hostname.casefold()
    return host in {"openrouter.ai", "pinference.ai"} or host.endswith(
        (".openrouter.ai", ".pinference.ai")
    )


def _mark_last_tool(
    body: dict[str, Any], *, anthropic: bool, marker: dict[str, str]
) -> bool:
    tools = body.get("tools")
    if not isinstance(tools, list) or not tools:
        return False
    for index in range(len(tools) - 1, -1, -1):
        tool = tools[index]
        if not isinstance(tool, dict):
            continue
        if anthropic:
            # Server tools (web search, computer use, ...) do not accept a
            # cache marker; fall back to the last custom tool.
            if tool.get("type") not in {None, "custom"} or "name" not in tool:
                continue
        elif tool.get("type") != "function":
            continue
        updated = list(tools)
        updated[index] = {**tool, "cache_control": dict(marker)}
        body["tools"] = updated
        return True
    return False


def _mark_last_text_block(
    message: dict[str, Any], marker: dict[str, str]
) -> dict | None:
    content = message.get("content")
    if isinstance(content, str):
        if not content:
            return None
        return {
            **message,
            "content": [
                {
                    "type": "text",
                    "text": content,
                    "cache_control": dict(marker),
                }
            ],
        }
    if not isinstance(content, list):
        return None
    for index in range(len(content) - 1, -1, -1):
        block = content[index]
        if (
            isinstance(block, dict)
            and block.get("type") == "text"
            and isinstance(block.get("text"), str)
            and block["text"]
        ):
            updated = list(content)
            updated[index] = {**block, "cache_control": dict(marker)}
            return {**message, "content": updated}
    return None


def _mark_last_anthropic_block(
    message: dict[str, Any], marker: dict[str, str]
) -> dict | None:
    content = message.get("content")
    if isinstance(content, str):
        if not content:
            return None
        return {
            **message,
            "content": [
                {
                    "type": "text",
                    "text": content,
                    "cache_control": dict(marker),
                }
            ],
        }
    if not isinstance(content, list):
        return None
    for index in range(len(content) - 1, -1, -1):
        block = content[index]
        if not isinstance(block, dict):
            continue
        block_type = block.get("type")
        if not isinstance(block_type, str) or block_type not in _ANTHROPIC_BLOCK_TYPES:
            continue
        if block_type == "text" and not (
            isinstance(block.get("text"), str) and block["text"]
        ):
            continue
        updated = list(content)
        updated[index] = {**block, "cache_control": dict(marker)}
        return {**message, "content": updated}
    return None


def _leading_instruction_indices(messages: list[Any]) -> list[int]:
    indices: list[int] = []
    for index, message in enumerate(messages):
        if not isinstance(message, dict):
            break
        role = message.get("role")
        if not isinstance(role, str) or role not in {"system", "developer"}:
            break
        indices.append(index)
    return indices


def _has_chat_history(messages: list[Any]) -> bool:
    user_turns = 0
    for message in messages:
        if not isinstance(message, dict):
            continue
        role = message.get("role")
        if not isinstance(role, str):
            continue
        if role == "user":
            user_turns += 1
        elif role in {"assistant", "tool"}:
            return True
    return user_turns > 1


def _has_messages_history(messages: list[Any]) -> bool:
    if len([message for message in messages if isinstance(message, dict)]) <= 1:
        return False
    return isinstance(messages[-1], dict) and messages[-1].get("role") == "user"


def _mark_chat(body: dict[str, Any], marker: dict[str, str]) -> bool:
    applied = _mark_last_tool(body, anthropic=False, marker=marker)
    messages = body.get("messages")
    if not isinstance(messages, list) or not messages:
        return applied

    updated_messages = list(messages)
    messages_changed = False
    for index in reversed(_leading_instruction_indices(messages)):
        marked = _mark_last_text_block(messages[index], marker)
        if marked is not None:
            updated_messages[index] = marked
            messages_changed = True
            applied = True
            break

    last_message = messages[-1] if messages else None
    last_role = last_message.get("role") if isinstance(last_message, dict) else None
    if (
        _has_chat_history(messages)
        and isinstance(last_message, dict)
        and isinstance(last_role, str)
        and last_role in {"user", "tool"}
    ):
        marked = _mark_last_text_block(last_message, marker)
        if marked is not None:
            updated_messages[-1] = marked
            messages_changed = True
            applied = True

    if messages_changed:
        body["messages"] = updated_messages
    return applied


def _mark_anthropic_messages(body: dict[str, Any], marker: dict[str, str]) -> bool:
    applied = _mark_last_tool(body, anthropic=True, marker=marker)

    system = body.get("system")
    if isinstance(system, str):
        if system:
            body["system"] = [
                {
                    "type": "text",
                    "text": system,
                    "cache_control": dict(marker),
                }
            ]
            applied = True
    elif isinstance(system, list):
        for index in range(len(system) - 1, -1, -1):
            block = system[index]
            if (
                isinstance(block, dict)
                and block.get("type") == "text"
                and isinstance(block.get("text"), str)
                and block["text"]
            ):
                updated = list(system)
                updated[index] = {**block, "cache_control": dict(marker)}
                body["system"] = updated
                applied = True
                break

    messages = body.get("messages")
    if isinstance(messages, list) and messages and _has_messages_history(messages):
        marked = _mark_last_anthropic_block(messages[-1], marker)
        if marked is not None:
            body["messages"] = [*messages[:-1], marked]
            applied = True
    return applied


def maybe_add_prompt_cache_control(
    dialect: Dialect,
    body: dict[str, Any],
    base_url: str,
    cache_retention: CacheRetention | None = None,
) -> dict[str, Any]:
    """Return a request body with prompt-cache markers when eligible.

    Only Anthropic models on Prime Inference/OpenRouter are touched, and only
    when the caller has not already set any cache marker. The original body is
    left untouched; edited lists and blocks are copied.
    """
    if dialect.name not in {"anthropic_messages", "chat_completions"}:
        return body
    if not _is_gateway(base_url) or not _is_anthropic_model(body):
        return body
    retention = resolve_cache_retention(cache_retention)
    if retention == "none":
        return body
    if _has_caller_marker(body):
        return body

    marker = _cache_marker(retention)
    updated = dict(body)
    if isinstance(body.get("tools"), list):
        updated["tools"] = list(body["tools"])
    if isinstance(body.get("messages"), list):
        updated["messages"] = list(body["messages"])
    if isinstance(body.get("system"), list):
        updated["system"] = list(body["system"])

    applied = (
        _mark_anthropic_messages(updated, marker)
        if dialect.name == "anthropic_messages"
        else _mark_chat(updated, marker)
    )
    return updated if applied else body
