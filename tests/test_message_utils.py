from verifiers.types import AssistantMessage
from verifiers.utils.message_utils import from_raw_message, normalize_messages


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
