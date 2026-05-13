import asyncio
from unittest.mock import AsyncMock, MagicMock

from verifiers.errors import InfraError, OverlongPromptError
from verifiers.types import (
    Response,
    ResponseMessage,
    TextContentPart,
    ToolCall,
    Usage,
)
from verifiers.utils import interception_utils
from verifiers.utils.interception_utils import (
    InterceptionError,
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


def test_interception_server_authorizes_bearer_and_x_api_key():
    server = InterceptionServer(port=0, secret="test-secret")
    request = MagicMock()

    request.headers = {"Authorization": "Bearer test-secret"}
    assert server._authorized(request)
    request.headers = {"x-api-key": "test-secret"}
    assert server._authorized(request)
    request.headers = {}
    assert not server._authorized(request)


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


async def test_keepalive_emitted_during_idle(monkeypatch):
    """During the idle window (no chunks on chunk_queue) the handler must
    emit SSE keepalive comments so upstream idle-timeouts don't fire."""
    monkeypatch.setattr(interception_utils, "KEEPALIVE_INTERVAL_SECONDS", 0.05)
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

    chunk_queue: asyncio.Queue = asyncio.Queue()  # starts empty
    response_future: asyncio.Future = asyncio.Future()
    intercept = {
        "chunk_queue": chunk_queue,
        "response_future": response_future,
    }

    task = asyncio.create_task(
        server._handle_streaming_response(MagicMock(), "r1", intercept)
    )
    await asyncio.sleep(0.2)  # enough for a few keepalive cycles

    # Close the loop cleanly: EOF sentinel + resolved future → handler returns.
    response_future.set_result(None)
    await chunk_queue.put(None)
    await task

    assert any(w == b": keepalive\n\n" for w in writes), (
        f"expected at least one keepalive write, got writes={writes}"
    )


async def test_keepalive_write_failure_surfaces_to_state(monkeypatch):
    """A failed keepalive write (upstream already cut the TCP connection)
    must funnel into ``state["error"]`` with elapsed-time instrumentation."""
    monkeypatch.setattr(interception_utils, "KEEPALIVE_INTERVAL_SECONDS", 0.05)
    server = InterceptionServer(port=0)
    state: dict = {}
    server.register_rollout("r1", state=state)

    fake_response = MagicMock()
    fake_response.prepare = AsyncMock()
    fake_response.write = AsyncMock(side_effect=ConnectionResetError("tunnel died"))
    fake_response.write_eof = AsyncMock()
    monkeypatch.setattr(
        interception_utils.web, "StreamResponse", lambda **_: fake_response
    )

    chunk_queue: asyncio.Queue = asyncio.Queue()  # never produces
    intercept = {
        "chunk_queue": chunk_queue,
        "response_future": asyncio.Future(),
    }

    await server._handle_streaming_response(MagicMock(), "r1", intercept)

    assert isinstance(state["error"], StreamInterrupted)
    msg = str(state["error"])
    assert "Keepalive write failed" in msg
    assert "ConnectionResetError" in msg


async def test_non_streaming_overlong_prompt_surfaces_as_stop_condition(monkeypatch):
    """``OverlongPromptError`` is an expected stop condition the multi-turn
    rollout loop translates into ``state['prompt_too_long']``. The interception
    handler must not also wrap it as ``state['error']`` — otherwise the
    rollout reports both ``stop=prompt_too_long`` and a spurious
    ``InterceptionError`` in metrics."""
    server = InterceptionServer(port=0)
    state: dict = {}
    server.register_rollout("r1", state=state)

    request = MagicMock()
    request.match_info = {"rollout_id": "r1"}
    request.json = AsyncMock(
        return_value={"stream": False, "messages": [], "model": "test"}
    )
    request.headers = {"Authorization": f"Bearer {server.secret}"}

    def fake_json_response(data, status=200):
        return MagicMock(_body=data, status=status)

    monkeypatch.setattr(interception_utils.web, "json_response", fake_json_response)

    handler_task = asyncio.create_task(server._handle_request(request))

    for _ in range(50):
        if server.intercepts:
            break
        await asyncio.sleep(0.01)
    assert server.intercepts, "handler did not register intercept"
    intercept = next(iter(server.intercepts.values()))
    interception_utils.deliver_response(
        intercept, None, error=OverlongPromptError("ctx exceeded")
    )

    response = await handler_task

    assert response.status == 500
    assert state.get("error") is None, (
        f"OverlongPromptError must not surface as state['error']; got {state.get('error')!r}"
    )
    assert state.get("prompt_too_long") is True
    assert state.get("is_truncated") is True


async def test_streaming_overlong_prompt_surfaces_as_stop_condition(monkeypatch):
    """Same as the non-streaming variant: ``OverlongPromptError`` raised on
    the streaming response_future must mark the rollout as
    ``prompt_too_long`` instead of becoming a ``StreamInterrupted`` error."""
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
    response_future.set_exception(OverlongPromptError("ctx exceeded"))
    intercept = {
        "chunk_queue": chunk_queue,
        "response_future": response_future,
    }

    await server._handle_streaming_response(MagicMock(), "r1", intercept)

    assert state.get("error") is None, (
        f"OverlongPromptError must not surface as state['error']; got {state.get('error')!r}"
    )
    assert state.get("prompt_too_long") is True
    assert state.get("is_truncated") is True


async def test_non_streaming_response_future_failure_surfaces_to_state(monkeypatch):
    """Non-streaming counterpart: if the model call fails and
    ``deliver_response`` sets the future's exception, the non-streaming
    branch of ``_handle_request`` re-raises when awaiting it. That failure
    must funnel into ``state['error']`` as ``InterceptionError`` so the
    rollout halts visibly (HTTP 500 still returned to the client)."""
    server = InterceptionServer(port=0)
    state: dict = {}
    server.register_rollout("r1", state=state)

    request = MagicMock()
    request.match_info = {"rollout_id": "r1"}
    request.json = AsyncMock(
        return_value={"stream": False, "messages": [], "model": "test"}
    )
    request.headers = {"Authorization": f"Bearer {server.secret}"}

    def fake_json_response(data, status=200):
        return MagicMock(_body=data, status=status)

    monkeypatch.setattr(interception_utils.web, "json_response", fake_json_response)

    handler_task = asyncio.create_task(server._handle_request(request))

    for _ in range(50):
        if server.intercepts:
            break
        await asyncio.sleep(0.01)
    assert server.intercepts, "handler did not register intercept"
    intercept = next(iter(server.intercepts.values()))
    interception_utils.deliver_response(
        intercept, None, error=RuntimeError("vLLM raised")
    )

    response = await handler_task

    assert response.status == 500
    assert isinstance(state["error"], InterceptionError), (
        f"expected InterceptionError, got {type(state.get('error'))}"
    )
    msg = str(state["error"])
    assert "Intercepted request failed" in msg
    assert "RuntimeError" in msg
    assert "vLLM raised" in msg
