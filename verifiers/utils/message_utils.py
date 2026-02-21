import base64
import binascii
import json
from collections.abc import Mapping
from enum import Enum
from typing import Any, cast

from rich.text import Text

from verifiers.types import (
    AssistantMessage,
    ContentPart,
    ImageUrlContentPart,
    InputAudioContentPart,
    Message,
    Messages,
    SystemMessage,
    TextContentPart,
    TextMessage,
    ToolMessage,
    UserMessage,
)


class ImageMode(str, Enum):
    PLACEHOLDER = "placeholder"
    BASE64 = "base64"


def coerce_image_mode(
    image_mode: str | ImageMode, *, arg_name: str = "image_mode"
) -> ImageMode:
    """Convert a string to ImageMode with a helpful error."""
    if isinstance(image_mode, ImageMode):
        return image_mode
    try:
        return ImageMode(image_mode)
    except ValueError as exc:
        valid_modes = "', '".join(mode.value for mode in ImageMode)
        raise ValueError(
            f"Invalid {arg_name}: {image_mode}. Expected one of '{valid_modes}'."
        ) from exc


def from_raw_content_part(part: dict[str, Any]) -> ContentPart:
    """Convert a raw content-part dict to a typed content part when possible."""
    part_type = part.get("type")
    if part_type == "text":
        return TextContentPart.model_validate(part)
    if part_type == "image_url":
        return ImageUrlContentPart.model_validate(part)
    if part_type == "input_audio":
        return InputAudioContentPart.model_validate(part)
    return part


def _normalize_raw_message_content(message: dict[str, Any]) -> dict[str, Any]:
    content = message.get("content")
    if isinstance(content, list):
        normalized_parts = []
        for part in content:
            if isinstance(part, dict):
                normalized_parts.append(from_raw_content_part(part))
            else:
                normalized_parts.append(part)
        message = dict(message)
        message["content"] = normalized_parts
    return message


def _normalize_raw_tool_calls(message: dict[str, Any]) -> dict[str, Any]:
    if message.get("role") != "assistant":
        return message

    tool_calls = message.get("tool_calls")
    if not isinstance(tool_calls, list):
        return message

    normalized_tool_calls: list[Any] = []
    for tool_call in tool_calls:
        if not isinstance(tool_call, dict):
            normalized_tool_calls.append(tool_call)
            continue

        if "name" in tool_call and "arguments" in tool_call:
            normalized_tool_calls.append(tool_call)
            continue

        function = tool_call.get("function")
        if not isinstance(function, dict):
            normalized_tool_calls.append(tool_call)
            continue

        name = function.get("name")
        arguments = function.get("arguments")
        if not isinstance(name, str):
            normalized_tool_calls.append(tool_call)
            continue

        if isinstance(arguments, str):
            arguments_str = arguments
        else:
            try:
                arguments_str = json.dumps(arguments if arguments is not None else {})
            except (TypeError, ValueError):
                arguments_str = str(arguments)

        tool_call_id = tool_call.get("id")
        if not isinstance(tool_call_id, str):
            tool_call_id = name

        normalized_tool_calls.append(
            {
                "id": tool_call_id,
                "name": name,
                "arguments": arguments_str,
            }
        )

    message = dict(message)
    message["tool_calls"] = normalized_tool_calls
    return message


def from_raw_message(message: dict) -> Message:
    """Convert a raw dict to the appropriate Pydantic message type."""
    message = _normalize_raw_message_content(message)
    message = _normalize_raw_tool_calls(message)
    if message["role"] == "text":
        return TextMessage.model_validate(message)
    elif message["role"] == "system":
        return SystemMessage.model_validate(message)
    elif message["role"] == "user":
        return UserMessage.model_validate(message)
    elif message["role"] == "assistant":
        return AssistantMessage.model_validate(message)
    elif message["role"] == "tool":
        return ToolMessage.model_validate(message)
    else:
        raise ValueError(f"Unknown role: {message['role']}")


def normalize_messages(
    value: Messages | str, *, field_name: str = "messages"
) -> Messages:
    """Normalize raw/string message inputs into provider-agnostic Message objects."""
    if isinstance(value, str):
        return [TextMessage(content=value)]
    normalized: Messages = []
    for message in value:
        if isinstance(message, dict):
            normalized.append(from_raw_message(dict(message)))
            continue
        if hasattr(message, "role") and hasattr(message, "content"):
            normalized.append(cast(Message, message))
            continue
        raise TypeError(
            f"Invalid {field_name} item type: {type(message).__name__}. "
            "Expected vf.Message-like objects."
        )
    return normalized


def concat_messages(messages_list: list[Messages]) -> Messages:
    """Concatenate multiple Messages lists into one."""
    result = []
    for messages in messages_list:
        result.extend(messages)
    return result


def _extract_data_uri_base64(url: str) -> tuple[str, str]:
    if not url.startswith("data:"):
        raise ValueError(
            f"Image URLs must be data URIs when image_mode='base64'. Got: {url[:64]}"
        )
    if "," not in url:
        raise ValueError("Invalid data URI: missing comma separator")
    header, payload = url.split(",", 1)
    if ";base64" not in header:
        raise ValueError("Data URI must include ';base64' when image_mode='base64'")
    media_type = header.removeprefix("data:").split(";", 1)[0]
    if not media_type.startswith("image/"):
        raise ValueError(f"Expected image/* media type in data URI, got: {media_type}")
    if payload == "":
        raise ValueError("Data URI payload is empty")
    try:
        base64.b64decode(payload, validate=True)
    except binascii.Error as exc:
        raise ValueError("Data URI payload is not valid base64") from exc
    return media_type, payload


def _extract_image_payload(
    part: dict[str, Any], max_image_base64_chars: int | None
) -> dict[str, str | int]:
    image_url_obj = part.get("image_url")
    if isinstance(image_url_obj, dict):
        url = image_url_obj.get("url")
    else:
        url = getattr(image_url_obj, "url", None)
    if not isinstance(url, str):
        raise ValueError("image_url content block must contain a string URL")

    media_type, payload = _extract_data_uri_base64(url)
    payload_size = len(payload)
    if max_image_base64_chars is not None and payload_size > max_image_base64_chars:
        raise ValueError(
            f"Image base64 payload exceeds max_image_base64_chars: {payload_size} > {max_image_base64_chars}"
        )
    return {
        "media_type": media_type,
        "base64": payload,
        "base64_chars": payload_size,
    }


def message_to_printable(
    message: Any,
    image_mode: str | ImageMode = ImageMode.PLACEHOLDER,
    max_image_base64_chars: int | None = None,
) -> Any:
    """Convert message content into log/save-friendly text placeholders.

    - text parts are preserved
    - input_audio parts are rendered as [audio]
    - image_url parts are rendered as [image]
    - in base64 mode, extracted image payloads are emitted under `images`
    """
    image_mode = coerce_image_mode(image_mode)

    if isinstance(message, dict):
        role = message.get("role")
        content = message.get("content")
        reasoning_content = message.get("reasoning_content")
        tool_calls = message.get("tool_calls")

        if isinstance(content, list):
            chunks: list[str] = []
            images: list[dict[str, str | int]] = []
            for part in content:
                if not isinstance(part, dict):
                    chunks.append(str(part))
                    continue

                part_type = part.get("type")
                if part_type == "text":
                    text = part.get("text")
                    if isinstance(text, str):
                        chunks.append(text)
                elif part_type in {"input_audio", "audio"}:
                    chunks.append("[audio]")
                elif part_type == "image_url":
                    chunks.append("[image]")
                    if image_mode == ImageMode.BASE64:
                        images.append(
                            _extract_image_payload(part, max_image_base64_chars)
                        )

            printable: dict[str, Any] = {
                "role": role,
                "content": "\n\n".join(chunks).strip(),
            }
            if images:
                printable["images"] = images
            if isinstance(reasoning_content, str):
                printable["reasoning_content"] = reasoning_content
            if tool_calls is not None:
                printable["tool_calls"] = tool_calls
            return printable

        return message

    content = getattr(message, "content", None)
    if isinstance(content, list):
        raw = (
            message.model_dump()
            if hasattr(message, "model_dump")
            else {"content": content}
        )
        printable = message_to_printable(
            raw,
            image_mode=image_mode,
            max_image_base64_chars=max_image_base64_chars,
        )
        if hasattr(message, "model_copy"):
            return message.model_copy(update={"content": printable.get("content", "")})
        return printable
    return message


def messages_to_printable(
    messages: Any,
    image_mode: str | ImageMode = ImageMode.PLACEHOLDER,
    max_image_base64_chars: int | None = None,
) -> Any:
    """Convert messages to printable/saveable form."""
    if isinstance(messages, str):
        return messages
    return [
        message_to_printable(
            m,
            image_mode=image_mode,
            max_image_base64_chars=max_image_base64_chars,
        )
        for m in messages or []
    ]


def strip_nones_from_content(messages: list[Any]) -> list[Any]:
    """Strip None-valued keys from content parts (HF map schema normalization helper)."""
    result: list[Any] = []
    for msg in messages:
        if not isinstance(msg, dict):
            result.append(msg)
            continue
        content = msg.get("content")
        if isinstance(content, list):
            new_msg = dict(msg)
            new_msg["content"] = [
                {k: v for k, v in c.items() if v is not None}
                if isinstance(c, dict)
                else c
                for c in content
            ]
            result.append(new_msg)
        else:
            result.append(msg)
    return result


# --- Legacy utilities (still used by save_utils, trainer, logging) ---


def format_messages(messages: Any) -> Text:
    """Format messages for display. Works with both Pydantic messages and legacy dicts."""

    def _attr_or_key(obj: Any, key: str, default: Any = None) -> Any:
        val = getattr(obj, key, None)
        if val is not None:
            return val
        if isinstance(obj, Mapping):
            return obj.get(key, default)
        return default

    def _normalize_tool_call(tc: Any) -> dict[str, str]:
        if isinstance(tc, str):
            tc = json.loads(tc)
        src = _attr_or_key(tc, "function") or tc
        name = _attr_or_key(src, "name", "") or ""
        args = _attr_or_key(src, "arguments", {}) or {}
        if not isinstance(args, str):
            try:
                args = json.dumps(args)
            except Exception:
                args = str(args)
        return {"name": name, "args": args}

    if isinstance(messages, str):
        return Text(messages)

    out = Text()
    for idx, msg in enumerate(messages):
        if idx:
            out.append("\n\n")

        role = _attr_or_key(msg, "role", "")
        content = _attr_or_key(msg, "content", "")
        style = "bright_cyan" if role == "assistant" else "bright_magenta"

        out.append(f"{role}: ", style="bold")

        reasoning_content = _attr_or_key(msg, "reasoning_content")
        if isinstance(reasoning_content, str) and reasoning_content.strip():
            out.append("\n")
            out.append("[reasoning]\n", style="dim")
            out.append(reasoning_content, style="dim")
            out.append("\n")

        if content:
            if isinstance(reasoning_content, str) and reasoning_content.strip():
                out.append("\n")
            out.append(str(content), style=style)

        tool_calls = _attr_or_key(msg, "tool_calls")
        for tc in tool_calls or []:
            payload = _normalize_tool_call(tc)
            out.append(
                "\n\n[tool call]\n" + json.dumps(payload, indent=2, ensure_ascii=False),
                style=style,
            )

    return out


def sanitize_tool_calls(messages: Messages):
    """Sanitize tool calls from messages for serialization.

    Used by save_utils and trainer to convert tool call objects to JSON strings.
    Works with both Pydantic message objects and legacy dicts.
    """
    if not isinstance(messages, list):
        return messages
    sanitized_messages = []
    for m in messages:
        # Support both Pydantic message objects and legacy dicts
        if isinstance(m, dict):
            tool_calls = m.get("tool_calls")
            reasoning_content = m.get("reasoning_content")
        else:
            tool_calls = getattr(m, "tool_calls", None)
            reasoning_content = getattr(m, "reasoning_content", None)

        if tool_calls:
            tool_calls_json = []
            for tc in tool_calls:
                if isinstance(tc, str):
                    tool_calls_json.append(tc)
                    continue
                if isinstance(tc, dict):
                    tc_dict = tc
                else:
                    model_dump = getattr(tc, "model_dump", None)
                    assert model_dump is not None
                    tc_dict = model_dump(exclude_none=True)
                tool_calls_json.append(json.dumps(tc_dict))
            if isinstance(m, dict):
                new_m = dict(m)
                new_m["tool_calls"] = tool_calls_json
            else:
                new_m = {
                    "role": m.role,
                    "content": m.content or "",
                    "tool_calls": tool_calls_json,
                }
                if isinstance(reasoning_content, str):
                    new_m["reasoning_content"] = reasoning_content
            sanitized_messages.append(new_m)
        else:
            sanitized_messages.append(m)
    return sanitized_messages
