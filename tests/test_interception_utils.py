from verifiers.errors import InfraError
from verifiers.types import (
    Response,
    ResponseMessage,
    TextContentPart,
    ToolCall,
    Usage,
)
from verifiers.utils.interception_utils import (
    InterceptionServer,
    StreamInterrupted,
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


def test_set_rollout_error_attaches_stream_interrupted_to_state():
    server = InterceptionServer(port=0)
    state: dict = {}
    server.register_rollout("r1", state=state)

    err = StreamInterrupted("tunnel died")
    server._set_rollout_error("r1", err)

    assert state["error"] is err
    assert isinstance(state["error"], InfraError)


def test_set_rollout_error_is_noop_without_state():
    # Covers the "state=None" path — upstream callers may not always wire it.
    server = InterceptionServer(port=0)
    server.register_rollout("r1")
    server._set_rollout_error("r1", StreamInterrupted("nope"))
    # No raise, and no state to read back.


def test_set_rollout_error_does_not_clobber_existing_error():
    # First error wins — later write failures must not hide the original cause.
    server = InterceptionServer(port=0)
    original = InfraError("original")
    state: dict = {"error": original}
    server.register_rollout("r1", state=state)

    server._set_rollout_error("r1", StreamInterrupted("later"))

    assert state["error"] is original


def test_set_rollout_error_is_noop_for_unknown_rollout():
    server = InterceptionServer(port=0)
    # No registration — must not raise.
    server._set_rollout_error("missing", StreamInterrupted("x"))
