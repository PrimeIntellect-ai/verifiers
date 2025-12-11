import json
from typing import cast

from openai.types.chat import (
    ChatCompletionAssistantMessageParam,
)

from verifiers.types import ChatMessage, Messages


def strip_nones_from_content(messages: list[ChatMessage]) -> list[ChatMessage]:
    """Return messages with None values stripped from content dicts (fixes HF Dataset schema unification)."""
    result: list[ChatMessage] = []
    for msg in messages:
        content = msg.get("content")
        if isinstance(content, list):
            new_msg = dict(msg)
            new_msg["content"] = [  # type: ignore[typeddict-item]
                {k: v for k, v in c.items() if v is not None}
                if isinstance(c, dict)
                else c
                for c in content
            ]
            result.append(new_msg)  # type: ignore[arg-type]
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


def message_to_printable(message: ChatMessage) -> ChatMessage:
    """
    Removes image_url objects from message content.
    """
    new_message: dict[str, object] = {}
    new_message["role"] = message["role"]
    new_message["content"] = []
    if "tool_calls" in message:
        assistant_msg = cast(ChatCompletionAssistantMessageParam, message)
        new_message["tool_calls"] = assistant_msg.get("tool_calls")
    content = message.get("content")
    if content is None:
        return cast(ChatMessage, new_message)
    if isinstance(content, str):
        new_message["content"].append(content)
    else:
        for c in content:
            if isinstance(c, str):
                new_message["content"].append(c)
            else:
                c_dict = dict(c)
                if c_dict["type"] == "text":
                    new_message["content"].append(c_dict["text"])
                elif c_dict["type"] == "image_url":
                    new_message["content"].append("[image]")
                elif str(c_dict.get("type", "")).startswith("input_audio"):
                    new_message["content"].append("[audio]")
    new_message["content"] = "\n\n".join(new_message["content"])
    return cast(ChatMessage, new_message)


def messages_to_printable(messages: Messages) -> Messages:
    """
    Removes image_url objects from messages.
    """
    if isinstance(messages, str):
        return messages
    return [message_to_printable(m) for m in messages or []]


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
                if isinstance(tc, dict):
                    tc_dict = tc
                else:
                    model_dump = getattr(tc, "model_dump", None)
                    assert model_dump is not None
                    tc_dict = model_dump()
                tool_calls_json.append(json.dumps(tc_dict))
            new_m = {
                "role": m["role"],
                "content": m.get("content", ""),
                "tool_calls": tool_calls_json,
            }
            sanitized_messages.append(new_m)
        else:
            sanitized_messages.append(m)
    return sanitized_messages
