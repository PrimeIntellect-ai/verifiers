import json
from typing import cast

from openai.types.chat import (
    ChatCompletion,
    ChatCompletionAssistantMessageParam,
    ChatCompletionMessage,
    ChatCompletionToolMessageParam,
)
from openai.types.chat.chat_completion import Choice
from openai.types.completion import Completion
from openai.types.completion_choice import CompletionChoice

from verifiers.types import ChatMessage, Messages, MessageType, ModelResponse


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


def cleanup_message(message: ChatMessage) -> ChatMessage:
    new_message: dict[str, object] = {}
    new_message["role"] = message["role"]
    if "tool_calls" in message:
        assistant_msg = cast(ChatCompletionAssistantMessageParam, message)
        new_message["tool_calls"] = assistant_msg.get("tool_calls")

    if "tool_call_id" in message:
        tool_msg = cast(ChatCompletionToolMessageParam, message)
        new_message["tool_call_id"] = tool_msg.get("tool_call_id")

    new_message["content"] = []
    content = message.get("content")
    if content is None:
        return cast(ChatMessage, new_message)
    if isinstance(content, str):
        new_message["content"] = content
    else:
        content_list = cast(list[object], new_message["content"])
        for c in content:
            new_c = dict(c)
            c_dict = dict(c)
            if "image_url" in c_dict and "type" in c_dict and c_dict["type"] == "text":
                new_c.pop("image_url")
                content_list.append(new_c)
            elif (
                "image_url" in c_dict
                and "type" in c_dict
                and c_dict["type"] == "image_url"
            ):
                new_c.pop("text")
                content_list.append(new_c)
            elif str(c_dict.get("type", "")).startswith("input_audio"):
                clean_c = {
                    "type": "input_audio",
                    "input_audio": c_dict.get("input_audio", {}),
                }
                content_list.append(clean_c)
            else:
                content_list.append(new_c)
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


def get_overlong_prompt_dummy_response(message_type: MessageType) -> ModelResponse:
    if message_type == "chat":
        return ChatCompletion(
            id="overlong-prompt",
            created=0,
            model="",
            object="chat.completion",
            choices=[
                Choice(
                    index=0,
                    message=ChatCompletionMessage(
                        role="assistant",
                        content="Prompt too long.",
                    ),
                    finish_reason="length",
                )
            ],
        )
    elif message_type == "completion":
        return Completion(
            id="overlong-prompt",
            created=0,
            model="",
            object="text_completion",
            choices=[
                CompletionChoice(
                    index=0,
                    text="Prompt too long.",
                    finish_reason="length",
                )
            ],
        )
    else:
        raise ValueError(f"Invalid message type: {message_type}")
