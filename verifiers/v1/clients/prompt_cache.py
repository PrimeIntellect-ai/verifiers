"""Gateway-side prompt-cache markers for Anthropic models.

Anthropic has no implicit prompt cache: a request only caches content when a
block carries ``cache_control``. Verifiers relays harness-native bodies, and
many harnesses send no marker, so Claude traffic through Prime Inference or
OpenRouter pays full input price. When the caller sends no marker, mark the
static prefix (last tool and last system/developer block) that repeats across
rollouts; multi-turn requests also mark the latest message.

Generated markers intentionally omit ``ttl`` (the five-minute default). A
caller that wants a different cache lifetime can send its own markers; any
caller marker disables generation here.
"""

from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
from typing import Any
from urllib.parse import urlsplit

from verifiers.v1.dialects.base import Dialect

_CACHE_MARKERS = frozenset({"cache_control", "prompt_cache_breakpoint"})
_EPHEMERAL = {"type": "ephemeral"}
_GATEWAY_HOSTS = frozenset({"api.pinference.ai", "openrouter.ai"})
_ANTHROPIC_BLOCK_TYPES = frozenset({"text", "image", "tool_result", "document"})


def _has_marker(value: Any) -> bool:
    return isinstance(value, Mapping) and not _CACHE_MARKERS.isdisjoint(value)


def _content_has_marker(content: Any) -> bool:
    if isinstance(content, Mapping):
        return _has_marker(content)
    if not isinstance(content, list):
        return False
    return any(_has_marker(block) for block in content)


def _has_caller_marker(body: Mapping[str, Any]) -> bool:
    if _has_marker(body):
        return True
    if _content_has_marker(body.get("system")):
        return True
    for key in ("messages", "input"):
        items = body.get(key)
        if not isinstance(items, list):
            continue
        for item in items:
            if not isinstance(item, Mapping):
                continue
            if _has_marker(item) or _content_has_marker(item.get("content")):
                return True
    tools = body.get("tools")
    return isinstance(tools, list) and any(_has_marker(tool) for tool in tools)


def _is_anthropic_model(body: Mapping[str, Any]) -> bool:
    model = body.get("model")
    if not isinstance(model, str):
        return False
    normalized = model.casefold()
    return normalized.startswith("anthropic/") or "claude-" in normalized


def _is_gateway(base_url: str) -> bool:
    hostname = urlsplit(base_url).hostname
    return hostname is not None and hostname.casefold() in _GATEWAY_HOSTS


def _mark_last_tool(body: dict[str, Any], *, anthropic: bool) -> bool:
    tools = body.get("tools")
    if not isinstance(tools, list) or not tools:
        return False
    last_tool = tools[-1]
    if not isinstance(last_tool, dict):
        return False
    if anthropic:
        # Server tools (web search, computer use, ...) do not accept a cache
        # marker; only custom tools do.
        if last_tool.get("type") not in {None, "custom"} or "name" not in last_tool:
            return False
    elif last_tool.get("type") != "function":
        return False
    last_tool["cache_control"] = dict(_EPHEMERAL)
    return True


def _mark_last_text_block(message: dict[str, Any]) -> bool:
    content = message.get("content")
    if isinstance(content, str):
        if not content:
            return False
        message["content"] = [
            {
                "type": "text",
                "text": content,
                "cache_control": dict(_EPHEMERAL),
            }
        ]
        return True
    if not isinstance(content, list):
        return False
    for index in range(len(content) - 1, -1, -1):
        block = content[index]
        if (
            isinstance(block, dict)
            and block.get("type") == "text"
            and isinstance(block.get("text"), str)
            and block["text"]
        ):
            content[index] = {**block, "cache_control": dict(_EPHEMERAL)}
            return True
    return False


def _mark_last_anthropic_block(message: dict[str, Any]) -> bool:
    content = message.get("content")
    if isinstance(content, str):
        if not content:
            return False
        message["content"] = [
            {
                "type": "text",
                "text": content,
                "cache_control": dict(_EPHEMERAL),
            }
        ]
        return True
    if not isinstance(content, list):
        return False
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
        content[index] = {**block, "cache_control": dict(_EPHEMERAL)}
        return True
    return False


def _leading_instruction_messages(messages: list[Any]) -> list[dict[str, Any]]:
    instructions: list[dict[str, Any]] = []
    for message in messages:
        if not isinstance(message, dict):
            break
        role = message.get("role")
        if not isinstance(role, str) or role not in {"system", "developer"}:
            break
        instructions.append(message)
    return instructions


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


def _mark_chat(body: dict[str, Any]) -> bool:
    applied = _mark_last_tool(body, anthropic=False)
    messages = body.get("messages")
    if not isinstance(messages, list) or not messages:
        return applied
    for message in reversed(_leading_instruction_messages(messages)):
        if _mark_last_text_block(message):
            applied = True
            break
    if (
        _has_chat_history(messages)
        and isinstance(messages[-1], dict)
        and _mark_last_text_block(messages[-1])
    ):
        applied = True
    return applied


def _mark_anthropic_messages(body: dict[str, Any]) -> bool:
    applied = _mark_last_tool(body, anthropic=True)

    system = body.get("system")
    if isinstance(system, str):
        if system:
            body["system"] = [
                {
                    "type": "text",
                    "text": system,
                    "cache_control": dict(_EPHEMERAL),
                }
            ]
            applied = True
    elif isinstance(system, list):
        for block in reversed(system):
            if (
                isinstance(block, dict)
                and block.get("type") == "text"
                and isinstance(block.get("text"), str)
                and block["text"]
            ):
                block["cache_control"] = dict(_EPHEMERAL)
                applied = True
                break

    messages = body.get("messages")
    if (
        isinstance(messages, list)
        and len(messages) > 1
        and isinstance(messages[-1], dict)
        and _mark_last_anthropic_block(messages[-1])
    ):
        applied = True
    return applied


def maybe_add_prompt_cache_control(
    dialect: Dialect,
    body: dict[str, Any],
    base_url: str,
) -> dict[str, Any]:
    """Return a request body with a static-prefix cache marker when eligible.

    Only Anthropic models on Prime Inference/OpenRouter are touched, and only
    when the caller has not already set any cache marker.
    """
    if dialect.name not in {"anthropic_messages", "chat_completions"}:
        return body
    if not _is_gateway(base_url) or not _is_anthropic_model(body):
        return body
    if _has_caller_marker(body):
        return body

    # Copy only the containers we edit so the caller's request body (and the
    # trace built from it) stays unchanged.
    updated = dict(body)
    for key in ("tools", "messages", "system", "input"):
        if isinstance(body.get(key), list):
            updated[key] = deepcopy(body[key])

    applied = (
        _mark_anthropic_messages(updated)
        if dialect.name == "anthropic_messages"
        else _mark_chat(updated)
    )
    return updated if applied else body
