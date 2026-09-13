"""Translate Inspect input messages to Verifiers' serializable message types."""

from __future__ import annotations

import json
from typing import Any
from urllib.parse import urlparse

import verifiers.v1 as vf


def _image_part(content: Any, where: str) -> vf.ImageUrlContentPart:
    image = getattr(content, "image", "")
    scheme = urlparse(image).scheme.lower()
    if scheme not in {"http", "https", "data"}:
        raise ValueError(
            f"{where} has an image that is not an HTTP(S) or data URL; "
            "local Inspect media paths are not portable to a Verifiers worker"
        )
    if getattr(content, "detail", "auto") != "auto":
        raise ValueError(
            f"{where} requests image detail={content.detail!r}, which v1 messages "
            "cannot preserve"
        )
    if getattr(content, "internal", None) is not None:
        raise ValueError(f"{where} contains provider-internal image state")
    return vf.ImageUrlContentPart(image_url=vf.ImageUrlSource(url=image))


def _content_parts(content: Any, where: str) -> vf.MessageContent:
    if isinstance(content, str):
        return content

    parts: list[vf.ContentPart] = []
    for index, part in enumerate(content):
        part_where = f"{where}.content[{index}]"
        part_type = getattr(part, "type", None)
        if part_type == "text":
            if (
                getattr(part, "internal", None) is not None
                or getattr(part, "refusal", None)
                or getattr(part, "citations", None)
            ):
                raise ValueError(
                    f"{part_where} contains text annotations v1 cannot preserve"
                )
            parts.append(vf.TextContentPart(text=part.text))
        elif part_type == "image":
            parts.append(_image_part(part, part_where))
        else:
            raise ValueError(
                f"{part_where} uses unsupported Inspect content type {part_type!r}; "
                "v1 currently accepts text and images"
            )
    return parts


def _assistant_content(content: Any, where: str) -> str:
    if isinstance(content, str):
        return content
    texts: list[str] = []
    for index, part in enumerate(content):
        part_type = getattr(part, "type", None)
        if part_type != "text":
            raise ValueError(
                f"{where}.content[{index}] uses {part_type!r}; prompt-supplied "
                "assistant messages support plain text only"
            )
        if (
            getattr(part, "internal", None) is not None
            or getattr(part, "refusal", None)
            or getattr(part, "citations", None)
        ):
            raise ValueError(
                f"{where}.content[{index}] contains annotations v1 cannot preserve"
            )
        texts.append(part.text)
    return "\n".join(texts)


def _tool_calls(message: Any, where: str) -> list[vf.ToolCall] | None:
    calls = getattr(message, "tool_calls", None)
    if not calls:
        return None
    translated: list[vf.ToolCall] = []
    for index, call in enumerate(calls):
        if getattr(call, "parse_error", None) or getattr(call, "view", None):
            raise ValueError(
                f"{where}.tool_calls[{index}] contains Inspect-only parse/view state"
            )
        translated.append(
            vf.ToolCall(
                id=call.id,
                type=getattr(call, "type", "function"),
                name=call.function,
                arguments=json.dumps(
                    call.arguments,
                    ensure_ascii=False,
                    separators=(",", ":"),
                ),
            )
        )
    return translated


def convert_message(message: Any, index: int) -> vf.Message:
    """Convert one Inspect ``ChatMessage`` without importing Inspect here."""
    role = getattr(message, "role", None)
    where = f"input[{index}]"
    if role == "system":
        return vf.SystemMessage(content=_content_parts(message.content, where))
    if role == "user":
        if getattr(message, "tool_call_id", None):
            raise ValueError(
                f"{where} is a user tool-result payload v1 cannot preserve"
            )
        return vf.UserMessage(content=_content_parts(message.content, where))
    if role == "assistant":
        return vf.AssistantMessage(
            content=_assistant_content(message.content, where),
            tool_calls=_tool_calls(message, where),
        )
    if role == "tool":
        tool_call_id = getattr(message, "tool_call_id", None)
        if not tool_call_id:
            raise ValueError(f"{where} is a tool message without a tool_call_id")
        if getattr(message, "error", None) is not None:
            raise ValueError(f"{where} contains Inspect-only tool error state")
        return vf.ToolMessage(
            tool_call_id=tool_call_id,
            name=getattr(message, "function", None),
            content=_content_parts(message.content, where),
        )
    raise ValueError(f"{where} has unsupported Inspect message role {role!r}")


def convert_input(value: Any) -> str | vf.Messages:
    """Convert ``Sample.input`` while retaining chat role and content order."""
    if isinstance(value, str):
        return value
    return [convert_message(message, index) for index, message in enumerate(value)]
