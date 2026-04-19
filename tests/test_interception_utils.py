import asyncio
from unittest.mock import AsyncMock, MagicMock

from verifiers.errors import InfraError
from verifiers.types import (
    Response,
    ResponseMessage,
    TextContentPart,
    ToolCall,
    Usage,
)
from verifiers.utils import interception_utils
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


def test_set_rollout_error_does_not_clobber_existing_error():
    # First error wins — later write failures must not hide the original cause.
    server = InterceptionServer(port=0)
    original = InfraError("original")
    state: dict = {"error": original}
    server.register_rollout("r1", state=state)

    server._set_rollout_error("r1", StreamInterrupted("later"))

    assert state["error"] is original


async def test_streaming_write_failure_surfaces_to_state(monkeypatch):
    """The real failure path: a mid-SSE transport close on the client side
    raises out of ``response.write(...)``. The except branch must funnel
    that into ``state["error"]`` so the rollout halts via ``has_error``."""
    server = InterceptionServer(port=0)
    state: dict = {}
    server.register_rollout("r1", state=state)

    # Mock StreamResponse whose second write raises (first write succeeds
    # to prove we're in the streaming loop, not failing at prepare()).
    writes: list[bytes] = []

    async def fake_write(data: bytes) -> None:
        writes.append(data)
        if len(writes) >= 2:
            raise ConnectionResetError("client closed transport")

    fake_response = MagicMock()
    fake_response.prepare = AsyncMock()
    fake_response.write = AsyncMock(side_effect=fake_write)
    fake_response.write_eof = AsyncMock()
    monkeypatch.setattr(
        interception_utils.web, "StreamResponse", lambda **_: fake_response
    )

    chunk_queue: asyncio.Queue = asyncio.Queue()
    await chunk_queue.put({"choices": [{"delta": {"content": "hi"}}]})
    await chunk_queue.put({"choices": [{"delta": {"content": " there"}}]})
    intercept = {
        "chunk_queue": chunk_queue,
        "response_future": asyncio.Future(),
    }

    await server._handle_streaming_response(MagicMock(), "r1", intercept)

    assert isinstance(state["error"], StreamInterrupted)
    assert "ConnectionResetError" in str(state["error"])


async def test_streaming_response_future_failure_surfaces_to_state(monkeypatch):
    """If the model call underlying the stream fails (e.g. vLLM raised and
    ``synthesize_stream(error=X)`` was called), the ``response_future`` await
    at the end of ``_handle_streaming_response`` raises. Previously that was
    only logged at debug, letting the agent see a clean ``data: [DONE]`` and
    exit 0 with an empty trajectory. Now it must funnel into ``state['error']``
    as ``StreamInterrupted`` so the rollout halts visibly."""
    server = InterceptionServer(port=0)
    state: dict = {}
    server.register_rollout("r1", state=state)

    writes: list[bytes] = []

    async def fake_write(data: bytes) -> None:
        writes.append(data)

    fake_response = MagicMock()
    fake_response.prepare = AsyncMock()
    fake_response.write = AsyncMock(side_effect=fake_write)
    fake_response.write_eof = AsyncMock()
    monkeypatch.setattr(
        interception_utils.web, "StreamResponse", lambda **_: fake_response
    )

    chunk_queue: asyncio.Queue = asyncio.Queue()
    await chunk_queue.put(None)

    response_future: asyncio.Future = asyncio.Future()
    response_future.set_exception(RuntimeError("vLLM raised"))

    intercept = {
        "chunk_queue": chunk_queue,
        "response_future": response_future,
    }

    await server._handle_streaming_response(MagicMock(), "r1", intercept)

    assert isinstance(state["error"], StreamInterrupted), (
        f"expected StreamInterrupted, got {type(state.get('error'))}"
    )
    msg = str(state["error"])
    assert "RuntimeError" in msg
    assert "vLLM raised" in msg
    assert any(w == b"data: [DONE]\n\n" for w in writes), writes
    fake_response.write_eof.assert_awaited()
