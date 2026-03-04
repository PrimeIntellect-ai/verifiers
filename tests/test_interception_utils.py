from verifiers.types import (
    Response,
    ResponseMessage,
    TextContentPart,
    ToolCall,
    Usage,
)
from verifiers.utils.interception_utils import (
    create_empty_completion,
    serialize_intercept_response,
)


def test_serialize_intercept_response_from_vf_response_uses_chat_completion_shape():
    response = Response(
        id="resp_1",
        created=123,
        model="test-model",
        usage=Usage(
            prompt_tokens=10,
            reasoning_tokens=0,
            completion_tokens=5,
            total_tokens=15,
        ),
        message=ResponseMessage(
            content=[TextContentPart(text="hello "), {"type": "text", "text": "world"}],
            reasoning_content=None,
            tool_calls=[
                ToolCall(id="call_1", name="echo", arguments='{"x": 1}'),
            ],
            finish_reason="tool_calls",
            is_truncated=False,
            tokens=None,
        ),
    )

    payload = serialize_intercept_response(response)

    assert payload["id"] == "resp_1"
    assert payload["object"] == "chat.completion"
    assert payload["model"] == "test-model"
    assert payload["choices"][0]["message"]["role"] == "assistant"
    assert payload["choices"][0]["message"]["content"] == "hello world"
    assert payload["choices"][0]["message"]["tool_calls"] == [
        {
            "id": "call_1",
            "type": "function",
            "function": {"name": "echo", "arguments": '{"x": 1}'},
        }
    ]
    assert payload["choices"][0]["finish_reason"] == "tool_calls"
    assert payload["usage"]["prompt_tokens"] == 10
    assert payload["usage"]["completion_tokens"] == 5
    assert payload["usage"]["total_tokens"] == 15


def test_serialize_intercept_response_passthrough_native_chat_completion():
    native = create_empty_completion("native-model")
    payload = serialize_intercept_response(native)

    assert payload["object"] == "chat.completion"
    assert payload["model"] == "native-model"
    assert len(payload["choices"]) == 1
