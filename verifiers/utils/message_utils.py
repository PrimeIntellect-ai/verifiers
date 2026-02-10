import base64
import binascii
import json
from collections.abc import Mapping
from enum import Enum
from typing import Any, cast

from openai.types.chat import (
    ChatCompletionAssistantMessageParam,
)
from rich.text import Text

from verifiers.types import ChatMessage, Messages


class ImageMode(str, Enum):
    PLACEHOLDER = "placeholder"
    BASE64 = "base64"


def coerce_image_mode(
    image_mode: str | ImageMode, *, arg_name: str = "image_mode"
) -> ImageMode:
    """Convert a string to ImageMode, raising ValueError with a helpful message on invalid input."""
    if isinstance(image_mode, ImageMode):
        return image_mode
    try:
        return ImageMode(image_mode)
    except ValueError as exc:
        valid_modes = "', '".join(mode.value for mode in ImageMode)
        raise ValueError(
            f"Invalid {arg_name}: {image_mode}. Expected one of '{valid_modes}'."
        ) from exc


def _extract_data_uri_base64(url: str) -> tuple[str, str]:
    """Extract media type and raw base64 payload from a data URI."""
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


def _extract_image_url(c_dict: dict) -> str:
    """Get the URL string from an image_url content block (handles both dict and object forms)."""
    image_url = c_dict.get("image_url")
    if isinstance(image_url, dict):
        url = image_url.get("url")
    else:
        url = getattr(image_url, "url", None)
    if not isinstance(url, str):
        raise ValueError("image_url content block must contain a string URL")
    return url


def _build_base64_image_payload(
    c_dict: dict, max_image_base64_chars: int | None
) -> dict[str, str | int]:
    """Extract and validate a base64 image payload from a content block dict."""
    image_url = _extract_image_url(c_dict)
    media_type, payload = _extract_data_uri_base64(image_url)
    payload_size = len(payload)
    if max_image_base64_chars is not None and payload_size > max_image_base64_chars:
        raise ValueError(
            f"Image base64 payload exceeds max_image_base64_chars: {payload_size} > {max_image_base64_chars}"
        )
    return {"media_type": media_type, "base64": payload, "base64_chars": payload_size}


def strip_nones_from_content(messages: list[ChatMessage]) -> list[ChatMessage]:
    """Return messages with None values stripped from content dicts (fixes HF Dataset schema unification)."""
    result: list[ChatMessage] = []
    for msg in messages:
        content = msg.get("content")
        if isinstance(content, list):
            new_msg = dict(msg)
            new_msg["content"] = [
                {k: v for k, v in c.items() if v is not None}
                if isinstance(c, dict)
                else c
                for c in content
            ]
            result.append(cast(ChatMessage, new_msg))
        else:
            result.append(msg)
    return result


def concat_messages(messages_list: list[Messages | ChatMessage]) -> Messages:
    all_str = all(isinstance(m, str) for m in messages_list)
    if all_str:
        out = ""
        for m in messages_list:
            assert isinstance(m, str)
            out += str(m)
        return out
    else:
        out = []
        for m in messages_list:
            if isinstance(m, list):
                out.extend(m)
            else:
                out.append(m)
        return out


def message_to_printable(
    message: ChatMessage,
    image_mode: str | ImageMode = ImageMode.PLACEHOLDER,
    max_image_base64_chars: int | None = None,
) -> ChatMessage:
    """
    Removes image_url objects from message content.
    """
    image_mode = coerce_image_mode(image_mode)
    new_message: dict[str, object] = {}
    new_message["role"] = message["role"]
    content_parts: list[str] = []
    images: list[dict[str, str | int]] = []
    if "tool_calls" in message:
        assistant_msg = cast(ChatCompletionAssistantMessageParam, message)
        new_message["tool_calls"] = assistant_msg.get("tool_calls")
    content = message.get("content")
    if content is None:
        new_message["content"] = ""
        return cast(ChatMessage, new_message)
    if isinstance(content, str):
        content_parts.append(content)
    else:
        for c in content:
            if isinstance(c, str):
                content_parts.append(c)
            else:
                c_dict = dict(c)
                if c_dict["type"] == "text":
                    content_parts.append(c_dict["text"])
                elif c_dict["type"] == "image_url":
                    content_parts.append("[image]")
                    if image_mode == ImageMode.BASE64:
                        images.append(
                            _build_base64_image_payload(c_dict, max_image_base64_chars)
                        )
                elif str(c_dict.get("type", "")).startswith("input_audio"):
                    content_parts.append("[audio]")
    new_message["content"] = "\n\n".join(content_parts)
    if images:
        new_message["images"] = images
    return cast(ChatMessage, new_message)


def messages_to_printable(
    messages: Messages,
    image_mode: str | ImageMode = ImageMode.PLACEHOLDER,
    max_image_base64_chars: int | None = None,
) -> Messages:
    """
    Removes image_url objects from messages.
    """
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


def format_messages(messages: Any) -> Text:
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

        assert isinstance(msg, dict)
        role = msg.get("role", "")
        content = msg.get("content", "")
        style = "bright_cyan" if role == "assistant" else "bright_magenta"

        out.append(f"{role}: ", style="bold")
        out.append(str(content) if content else "", style=style)

        for tc in msg.get("tool_calls") or []:
            payload = _normalize_tool_call(tc)
            out.append(
                "\n\n[tool call]\n" + json.dumps(payload, indent=2, ensure_ascii=False),
                style=style,
            )

    return out


def sanitize_tool_calls(messages: Messages):
    """
    Sanitize tool calls from messages.
    """
    if not isinstance(messages, list):
        return messages
    sanitized_messages = []
    for m in messages:
        if "tool_calls" in m:
            assistant_msg = cast(ChatCompletionAssistantMessageParam, m)
            tool_calls_json = []
            for tc in assistant_msg.get("tool_calls", []):
                if isinstance(tc, str):
                    tc_json = tc
                else:
                    model_dump = getattr(tc, "model_dump", None)
                    if callable(model_dump):
                        tc_json = json.dumps(model_dump())
                    else:
                        tc_json = json.dumps(tc)
                tool_calls_json.append(tc_json)
            new_m = dict(m)
            new_m["tool_calls"] = tool_calls_json
            sanitized_messages.append(new_m)
        else:
            sanitized_messages.append(m)
    return sanitized_messages
