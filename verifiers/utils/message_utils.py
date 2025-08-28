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
    new_message = {}
    new_message["role"] = message["role"]
    if "tool_calls" in message:
        new_message["tool_calls"] = message["tool_calls"]
    new_message["content"] = []
    content = message.get("content")
    if content is None:
        return cast(ChatMessage, new_message)
    if isinstance(content, str):
        new_message["content"] = content
    else:
        for c in content:
            new_c = c.copy()
            c_dict = dict(c)
            if "image_url" in c_dict and "type" in c_dict and c_dict["type"] == "text":
                new_c.pop("image_url")
                new_message["content"].append(new_c)
            elif (
                "image_url" in c_dict
                and "type" in c_dict
                and c_dict["type"] == "image_url"
            ):
                new_c.pop("text")
                new_message["content"].append(new_c)
            else:
                new_message["content"].append(new_c)
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
