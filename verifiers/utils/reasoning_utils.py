from __future__ import annotations

import re
from typing import Literal

from verifiers.types import ChatMessage

ReasoningFormat = Literal[
    "auto", "think_tags", "reasoning_content", "reasoning", "none"
]


def extract_reasoning_from_response(message_obj: object) -> str | None:
    """Extract reasoning from response message's provider-specific fields.

    Checks reasoning_content (vLLM/DeepSeek) then reasoning (OpenRouter).
    Returns None if no reasoning found or if the value is empty/whitespace.
    """
    for attr in ("reasoning_content", "reasoning"):
        value = getattr(message_obj, attr, None)
        if isinstance(value, str) and value.strip():
            return value
    return None


def detect_reasoning_format(
    message_obj: object, content: str | None
) -> ReasoningFormat:
    """Detect reasoning format from a response message.

    Returns "reasoning_content", "reasoning", "think_tags", or "none".
    """
    rc = getattr(message_obj, "reasoning_content", None)
    if isinstance(rc, str) and rc.strip():
        return "reasoning_content"

    r = getattr(message_obj, "reasoning", None)
    if isinstance(r, str) and r.strip():
        return "reasoning"

    if isinstance(content, str) and content.lstrip().startswith("<think>"):
        return "think_tags"

    return "none"


def normalize_reasoning_content(reasoning: str | None, content: str | None) -> str:
    """Prepend reasoning as <think> tags to content string.

    Edge cases:
    - reasoning + content → "<think>\\n{reasoning}\\n</think>\\n{content}"
    - reasoning + no content → "<think>\\n{reasoning}\\n</think>"
    - no reasoning → content as-is
    - content already starts with <think> → content as-is (avoid duplication)
    """
    if not reasoning or not reasoning.strip():
        return content or ""

    if isinstance(content, str) and content.lstrip().startswith("<think>"):
        return content

    think_block = f"<think>\n{reasoning}\n</think>"
    if content:
        return f"{think_block}\n{content}"
    return think_block


def strip_reasoning_from_content(content: str) -> tuple[str | None, str]:
    """Split <think>...</think> prefix from content.

    Returns (reasoning_text_or_None, remaining_content).
    Handles: no tags, truncated tags (no </think>), empty reasoning.
    """
    if not content.lstrip().startswith("<think>"):
        return None, content

    match = re.match(r"^\s*<think>(.*?)</think>\s*(.*)", content, re.DOTALL)
    if match is None:
        # Truncated: <think> present but no </think>
        return None, content

    reasoning = match.group(1).strip()
    remaining = match.group(2)

    return (reasoning if reasoning else None), remaining


def prepare_messages_for_provider(
    messages: list[ChatMessage],
    reasoning_format: ReasoningFormat,
) -> list[ChatMessage]:
    """Convert internal messages to provider format before sending.

    For "reasoning_content"/"reasoning" formats: strips <think> tags from
    assistant message content (provider doesn't expect them in input).
    For "think_tags"/"none"/"auto": returns messages unchanged.
    Only modifies assistant messages; user/system/tool messages pass through.
    """
    if reasoning_format not in ("reasoning_content", "reasoning"):
        return messages

    result: list[ChatMessage] = []
    for msg in messages:
        if msg.get("role") != "assistant":
            result.append(msg)
            continue

        content = msg.get("content")
        if not isinstance(content, str) or not content.lstrip().startswith("<think>"):
            result.append(msg)
            continue

        _reasoning, remaining = strip_reasoning_from_content(content)
        new_msg = dict(msg)
        new_msg["content"] = remaining
        result.append(new_msg)  # type: ignore[arg-type]

    return result
