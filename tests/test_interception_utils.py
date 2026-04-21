import asyncio
import json
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
    CHUNK_TRACE_DIR_ENV,
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
    assert "keepalive write failed" in msg
    assert "ConnectionResetError" in msg


async def test_chunk_trace_writes_when_env_var_set(monkeypatch, tmp_path):
    """With VF_CHUNK_TRACE_DIR set, every SSE chunk + a DONE sentinel are
    persisted as NDJSON under the trace dir; absent the env var, nothing is
    written."""
    server = InterceptionServer(port=0)
    state: dict = {}
    server.register_rollout("r_trace", state=state)

    async def fake_write(data: bytes) -> None:
        pass

    fake_response = MagicMock()
    fake_response.prepare = AsyncMock()
    fake_response.write = AsyncMock(side_effect=fake_write)
    fake_response.write_eof = AsyncMock()
    monkeypatch.setattr(
        interception_utils.web, "StreamResponse", lambda **_: fake_response
    )

    chunk_queue: asyncio.Queue = asyncio.Queue()
    await chunk_queue.put(
        {"id": "c1", "choices": [{"delta": {"content": "hi"}, "finish_reason": None}]}
    )
    await chunk_queue.put(
        {"id": "c1", "choices": [{"delta": {}, "finish_reason": "stop"}]}
    )
    await chunk_queue.put(None)
    future: asyncio.Future = asyncio.Future()
    future.set_result(None)
    intercept = {"chunk_queue": chunk_queue, "response_future": future}

    monkeypatch.setenv(CHUNK_TRACE_DIR_ENV, str(tmp_path))
    await server._handle_streaming_response(MagicMock(), "r_trace", intercept)

    trace_file = tmp_path / "r_trace.ndjson"
    assert trace_file.exists(), f"expected trace file at {trace_file}"
    lines = [json.loads(ln) for ln in trace_file.read_text().splitlines() if ln.strip()]
    events = [rec["event"] for rec in lines]
    assert events[0] == "open"
    assert events[-1] == "done"
    chunk_records = [r for r in lines if r["event"] == "chunk"]
    assert len(chunk_records) == 2
    assert chunk_records[0]["delta_content"] == "hi"
    assert chunk_records[1]["finish_reason"] == "stop"


async def test_chunk_trace_noop_when_env_var_unset(monkeypatch, tmp_path):
    """Without VF_CHUNK_TRACE_DIR set, no NDJSON file is created."""
    monkeypatch.delenv(CHUNK_TRACE_DIR_ENV, raising=False)
    server = InterceptionServer(port=0)
    state: dict = {}
    server.register_rollout("r_none", state=state)

    async def fake_write(data: bytes) -> None:
        pass

    fake_response = MagicMock()
    fake_response.prepare = AsyncMock()
    fake_response.write = AsyncMock(side_effect=fake_write)
    fake_response.write_eof = AsyncMock()
    monkeypatch.setattr(
        interception_utils.web, "StreamResponse", lambda **_: fake_response
    )

    chunk_queue: asyncio.Queue = asyncio.Queue()
    await chunk_queue.put({"id": "c1", "choices": [{"delta": {"content": "hi"}}]})
    await chunk_queue.put(None)
    future: asyncio.Future = asyncio.Future()
    future.set_result(None)
    intercept = {"chunk_queue": chunk_queue, "response_future": future}

    await server._handle_streaming_response(MagicMock(), "r_none", intercept)

    assert list(tmp_path.iterdir()) == [], (
        "no trace file should exist when env var is unset"
    )
