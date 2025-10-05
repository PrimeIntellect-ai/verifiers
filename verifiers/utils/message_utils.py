import json
from typing import cast

from verifiers.types import ChatMessage, Messages


def message_to_printable(message: ChatMessage) -> ChatMessage:
    """
    Removes image_url objects from message content.
    """
    new_message = {}
    new_message["role"] = message["role"]
    new_message["content"] = []
    if "tool_calls" in message:
        new_message["tool_calls"] = message["tool_calls"]
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
    return [message_to_printable(m) for m in messages]


def cleanup_message(message: ChatMessage) -> ChatMessage:
    new_message = {
        "role": message["role"],
        "content": []
    }

    if "tool_calls" in message:
        new_message["tool_calls"] = message["tool_calls"]
    if "tool_call_id" in message:
        new_message["tool_call_id"] = message["tool_call_id"]

    content = message.get("content")
    if content is None:
        return cast(ChatMessage, new_message)

    if isinstance(content, str):
        new_message["content"] = content
        return cast(ChatMessage, new_message)

    for c in content:
        c_dict = dict(c)
        c_type = c_dict.get("type")

        if c_type == "text":
            if c_dict.get("text") is not None:
                new_message["content"].append({
                    "type": "text",
                    "text": c_dict["text"]
                })

        elif c_type == "image_url":
            if "image_url" in c_dict:
                new_message["content"].append({
                    "type": "image_url",
                    "image_url": c_dict["image_url"]
                })

        elif c_type == "input_audio":
            if "input_audio" in c_dict:
                new_message["content"].append({
                    "type": "input_audio",
                    "input_audio": c_dict["input_audio"]
                })

        else:
            new_message["content"].append(c_dict)

    return cast(ChatMessage, new_message)

def cleanup_messages(messages: Messages) -> Messages:
    if isinstance(messages, str):
        return messages
    new_messages = []
    for m in messages:
        new_messages.append(cleanup_message(m))
    return new_messages


def sanitize_tool_calls(messages: Messages):
    """
    Sanitize tool calls from messages.
    """
    if not isinstance(messages, list):
        return messages
    sanitized_messages = []
    for m in messages:
        if "tool_calls" in m:
            new_m = {
                "role": m["role"],
                "content": m.get("content", ""),
                "tool_calls": [
                    json.dumps(tc.model_dump())  # type: ignore
                    for tc in m.get("tool_calls", [])
                ],
            }
            sanitized_messages.append(new_m)
        else:
            sanitized_messages.append(m)
    return sanitized_messages
