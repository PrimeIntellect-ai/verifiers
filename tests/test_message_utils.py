from verifiers.types import AssistantMessage, UserMessage
from verifiers.utils.message_utils import (
    from_raw_message,
    get_messages,
    normalize_messages,
)


def test_from_raw_message_normalizes_oai_tool_calls():
    raw = {
        "role": "assistant",
        "content": "calling tool",
        "tool_calls": [
            {
                "id": "call_1",
                "type": "function",
                "function": {
                    "name": "echo",
                    "arguments": '{"x": 1}',
                },
            }
        ],
    }

    message = from_raw_message(raw)

    assert isinstance(message, AssistantMessage)
    assert message.tool_calls is not None
    assert len(message.tool_calls) == 1
    assert message.tool_calls[0].id == "call_1"
    assert message.tool_calls[0].name == "echo"
    assert message.tool_calls[0].arguments == '{"x": 1}'


def test_normalize_messages_accepts_oai_tool_call_dicts():
    messages = normalize_messages(
        [
            {
                "role": "assistant",
                "content": "calling tool",
                "tool_calls": [
                    {
                        "id": "call_2",
                        "type": "function",
                        "function": {
                            "name": "lookup",
                            "arguments": {"q": "hello"},
                        },
                    }
                ],
            }
        ]
    )

    assert len(messages) == 1
    assistant = messages[0]
    assert isinstance(assistant, AssistantMessage)
    assert assistant.tool_calls is not None
    assert assistant.tool_calls[0].id == "call_2"
    assert assistant.tool_calls[0].name == "lookup"
    assert assistant.tool_calls[0].arguments == '{"q": "hello"}'


def test_get_messages_returns_typed_messages():
    messages = get_messages(
        [
            {"role": "user", "content": "question"},
            {"role": "assistant", "content": "answer"},
        ]
    )

    assert isinstance(messages[0], UserMessage)
    assert isinstance(messages[1], AssistantMessage)
    assert messages[-1].content == "answer"


def test_get_messages_filters_by_role_with_typed_return():
    messages = get_messages(
        [
            {"role": "user", "content": "question"},
            {"role": "assistant", "content": "answer"},
        ],
        role="assistant",
    )

    assert len(messages) == 1
    assert isinstance(messages[0], AssistantMessage)
    assert messages[0].content == "answer"
