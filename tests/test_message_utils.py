from pydantic import TypeAdapter

from verifiers.types import AssistantMessage, Messages, ToolCall

MESSAGES_ADAPTER = TypeAdapter(Messages)


def test_tool_call_accepts_openai_shape():
    raw = {
        "id": "call_1",
        "type": "function",
        "function": {
            "name": "echo",
            "arguments": '{"x": 1}',
        },
    }

    tool_call = ToolCall.model_validate(raw)

    assert tool_call.id == "call_1"
    assert tool_call.name == "echo"
    assert tool_call.arguments == '{"x": 1}'


def test_messages_adapter_accepts_openai_tool_call_dicts():
    messages = MESSAGES_ADAPTER.validate_python(
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
