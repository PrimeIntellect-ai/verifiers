from verifiers.types import (
    AssistantMessage,
    Message,
    Messages,
    SystemMessage,
    TextMessage,
    ToolMessage,
    UserMessage,
)


def concat_messages(messages_list: list[Messages]) -> Messages:
    concat_messages = []
    for messages in messages_list:
        concat_messages.extend(messages)
    return concat_messages


def from_raw_message(message: dict) -> Message:
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


def to_raw_message(message: Message) -> dict:
    return message.model_dump()


def message_to_printable(message: Message) -> Message:
    """
    Removes image_url objects from message content.
    """
    return message
    # new_message: dict[str, object] = {}
    # message = to_raw_message(message)
    # new_message["role"] = message["role"]
    # new_message["content"] = []
    # if "tool_calls" in message:
    #     assistant_msg = cast(ChatCompletionAssistantMessageParam, message)
    #     new_message["tool_calls"] = assistant_msg.get("tool_calls")
    # content = message.get("content")
    # if content is None:
    #     return cast(Message, new_message)
    # if isinstance(content, str):
    #     new_message["content"].append(content)
    # else:
    #     for c in content:
    #         if isinstance(c, str):
    #             new_message["content"].append(c)
    #         else:
    #             c_dict = dict(c)
    #             if c_dict["type"] == "text":
    #                 new_message["content"].append(c_dict["text"])
    #             elif c_dict["type"] == "image_url":
    #                 new_message["content"].append("[image]")
    #             elif str(c_dict.get("type", "")).startswith("input_audio"):
    #                 new_message["content"].append("[audio]")
    # new_message["content"] = "\n\n".join(new_message["content"])
    # return cast(Message, new_message)


def messages_to_printable(messages: Messages) -> Messages:
    """
    Removes image_url objects from messages.
    """
    return [message_to_printable(m) for m in messages or []]
