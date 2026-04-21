"""Utilities for intercepting API calls from agents running in sandboxes."""

# VF_CHUNK_TRACE_DIR: per-rollout NDJSON dump of every SSE chunk emitted to
# the agent (with timestamp, delta, finish_reason, and a sentinel line on
# normal close / stream interruption). Default: disabled. See
# ``_write_chunk_trace`` for format.

import asyncio
import json
import logging
import os
import socket
import struct
import time
import uuid
from pathlib import Path
from typing import Any, cast

from aiohttp import web
from openai.types.chat import (
    ChatCompletion,
    ChatCompletionMessage,
)
from openai.types.chat.chat_completion import Choice
from openai.types.chat.chat_completion_chunk import (
    ChatCompletionChunk,
    ChoiceDelta,
    ChoiceDeltaToolCall,
    ChoiceDeltaToolCallFunction,
)
from openai.types.chat.chat_completion_chunk import (
    Choice as ChunkChoice,
)

from verifiers.errors import InfraError
from verifiers.types import Response
from verifiers.utils.logging_utils import truncate

logger = logging.getLogger(__name__)


KEEPALIVE_INTERVAL_SECONDS = 10.0
CHUNK_TRACE_DIR_ENV = "VF_CHUNK_TRACE_DIR"
# Hard cap on per-rollout chunk-trace files; oldest pruned on open.
_CHUNK_TRACE_FILE_CAP = 20000
# Sample the server-side TCP_INFO state every N seconds during a streaming
# response so we can see the socket transition (e.g. to CLOSE_WAIT) before
# the write that surfaces the failure. 0 disables.
TCP_SAMPLE_INTERVAL_SECONDS = float(
    os.environ.get("VF_TCP_SAMPLE_INTERVAL_SECONDS", "2")
)

# TCP_INFO sockopt (Linux). Value is 11; Python may not expose a constant.
_TCP_INFO = getattr(socket, "TCP_INFO", 11)

# linux/include/net/tcp_states.h
_TCP_STATE_NAMES = {
    1: "ESTABLISHED",
    2: "SYN_SENT",
    3: "SYN_RECV",
    4: "FIN_WAIT1",
    5: "FIN_WAIT2",
    6: "TIME_WAIT",
    7: "CLOSE",
    8: "CLOSE_WAIT",
    9: "LAST_ACK",
    10: "LISTEN",
    11: "CLOSING",
}


def _read_tcp_state(transport: Any) -> tuple[int | None, int]:
    """Return (state_id, retransmits) from TCP_INFO, or (None, 0) if unavailable."""
    try:
        if transport is None:
            return (None, 0)
        sock = transport.get_extra_info("socket")
        if sock is None:
            return (None, 0)
        buf = sock.getsockopt(socket.IPPROTO_TCP, _TCP_INFO, 7)
        state, _ca, retx = struct.unpack("BBB", buf[:3])
        return (int(state), int(retx))
    except Exception:
        return (None, 0)


async def _sample_tcp_state(
    transport: Any,
    rollout_id: str,
    start: float,
    interval: float,
    state_ref: dict,
) -> None:
    """Periodically sample TCP_INFO on the stream socket; log state transitions.

    Runs as a side-task alongside the streaming-response write loop. Stores
    the latest observed state + timestamp in `state_ref` so the error path
    can include it in the StreamInterrupted message even when the socket is
    already closed at error time (the normal case).

    Also logs every transition at WARNING (important + rare — only fires on
    a real state change) so signal is visible regardless of log-level
    config in the host process.
    """
    last_state: int | None = None
    while True:
        try:
            await asyncio.sleep(interval)
        except asyncio.CancelledError:
            return
        state, retx = _read_tcp_state(transport)
        if state is None:
            continue
        elapsed = time.monotonic() - start
        state_ref["state"] = state
        state_ref["retx"] = retx
        state_ref["elapsed"] = elapsed
        if state != last_state:
            prev = (
                _TCP_STATE_NAMES.get(last_state, f"state{last_state}")
                if last_state is not None
                else "init"
            )
            curr = _TCP_STATE_NAMES.get(state, f"state{state}")
            logger.warning(
                f"[{rollout_id}] tcp_state {prev} -> {curr} at {elapsed:.1f}s (retx={retx})"
            )
            last_state = state


def _tcp_diag(
    transport: Any,
    peer: tuple | None = None,
    state_ref: dict | None = None,
) -> str:
    """Return a short one-line diagnostic about the TCP socket.

    Pulls peer/local addr (if not cached) plus TCP_INFO.state and
    retransmits from the underlying socket. If the socket is already gone
    at call time (typical on a cut), falls back to ``state_ref`` which the
    periodic sampler keeps populated. All lookups are best-effort —
    failures return an empty string so the error logging path is never
    broken by instrumentation.
    """
    parts: list[str] = []
    try:
        if peer is None and transport is not None:
            peer = transport.get_extra_info("peername")
        if peer:
            parts.append(f"peer={peer[0]}:{peer[1]}")
    except Exception:
        pass
    # Live read first (authoritative) — fall back to sampler's cache on failure
    state: int | None = None
    retx: int = 0
    try:
        if transport is not None:
            sock = transport.get_extra_info("socket")
            if sock is not None:
                buf = sock.getsockopt(socket.IPPROTO_TCP, _TCP_INFO, 7)
                state, _ca, retx = struct.unpack("BBB", buf[:3])
    except Exception:
        pass
    if state is None and state_ref:
        s = state_ref.get("state")
        if isinstance(s, int):
            state = s
            retx = int(state_ref.get("retx", 0))
            parts.append("tcp_src=cached")
    if state is not None:
        parts.append(f"tcp_state={_TCP_STATE_NAMES.get(state, f'state{state}')}")
        if retx:
            parts.append(f"retx={retx}")
    return " ".join(parts)


def _open_chunk_trace(rollout_id: str) -> Any:
    """Open a per-rollout NDJSON trace file if VF_CHUNK_TRACE_DIR is set.

    Returns an open file handle (line-buffered, write mode) or None if
    tracing is disabled or opening the file failed. Prunes oldest files
    when the directory exceeds the cap so disk usage stays bounded.
    """
    trace_dir = os.environ.get(CHUNK_TRACE_DIR_ENV)
    if not trace_dir:
        return None
    try:
        dir_path = Path(trace_dir)
        dir_path.mkdir(parents=True, exist_ok=True)
        try:
            files = [
                p for p in dir_path.iterdir() if p.is_file() and p.suffix == ".ndjson"
            ]
            if len(files) > _CHUNK_TRACE_FILE_CAP:
                files.sort(key=lambda p: p.stat().st_mtime)
                for p in files[: len(files) - _CHUNK_TRACE_FILE_CAP]:
                    p.unlink(missing_ok=True)
        except Exception:
            pass
        path = dir_path / f"{rollout_id}.ndjson"
        return path.open("a", buffering=1, encoding="utf-8")
    except Exception as e:
        logger.debug(f"[{rollout_id}] chunk-trace open failed: {e}")
        return None


def _write_chunk_trace(fh: Any, record: dict) -> None:
    """Best-effort write of one NDJSON record; never raises."""
    if fh is None:
        return
    try:
        fh.write(json.dumps(record, default=str) + "\n")
    except Exception:
        pass


def _close_chunk_trace(fh: Any) -> None:
    if fh is None:
        return
    try:
        fh.close()
    except Exception:
        pass


def _chunk_trace_record(chunk_dict: dict) -> dict:
    """Extract the useful fields from an SSE chunk for the trace log."""
    record: dict[str, Any] = {"ts": time.time(), "event": "chunk"}
    try:
        choices = chunk_dict.get("choices") or []
        if choices:
            choice = choices[0]
            delta = choice.get("delta") or {}
            record["delta_content"] = delta.get("content")
            record["delta_tool_calls"] = delta.get("tool_calls")
            reasoning = delta.get("reasoning_content")
            if reasoning:
                record["delta_reasoning_content"] = reasoning
            record["finish_reason"] = choice.get("finish_reason")
        record["id"] = chunk_dict.get("id")
    except Exception:
        pass
    return record


class StreamInterrupted(InfraError):
    """Raised when the intercepted streaming response to the agent is cut short.

    Without this, a mid-stream transport failure would be swallowed here and
    the agent would observe a truncated (but syntactically valid) SSE stream,
    often exiting with code 0 and an empty trajectory — bypassing the
    non-zero-exit error capture in `CliAgentEnv.poll_job_completion`.
    """


class InterceptionServer:
    """
    HTTP server that intercepts API requests from agents.

    Requests are queued for processing, and responses are delivered back
    to the agent once the actual model response is obtained.
    """

    def __init__(self, port: int):
        self.port = port
        self._app: Any = None
        self._runner: Any = None
        self._site: Any = None
        self._lock = asyncio.Lock()

        # Track active rollouts and their request queues
        self.active_rollouts: dict[str, dict[str, Any]] = {}
        # Track individual intercepts (request_id -> intercept data)
        self.intercepts: dict[str, dict[str, Any]] = {}

    async def start(self) -> None:
        async with self._lock:
            if self._app is not None:
                return

            app = web.Application()
            app.router.add_post(
                "/rollout/{rollout_id}/v1/chat/completions",
                self._handle_request,
            )
            app.router.add_get(
                "/health",
                lambda _: web.json_response({"status": "ok"}),
            )

            runner = web.AppRunner(app)
            await runner.setup()
            site = web.TCPSite(runner, "0.0.0.0", self.port)
            await site.start()

            self._app = app
            self._runner = runner
            self._site = site

            # OS-assigned port if port=0
            if self.port == 0:
                server = getattr(site, "_server", None)
                sockets = getattr(server, "sockets", None) if server else None
                if sockets:
                    self.port = sockets[0].getsockname()[1]
            if self.port == 0:
                raise RuntimeError("Failed to resolve OS-assigned port")

            logger.debug(f"Started interception server on port {self.port}")

    async def stop(self) -> None:
        async with self._lock:
            if self._runner is not None:
                try:
                    await self._runner.cleanup()
                    logger.debug("Stopped HTTP interception server")
                except RuntimeError as e:
                    if "Event loop is closed" not in str(e):
                        raise
                    logger.debug("HTTP server cleanup skipped (event loop closed)")
                finally:
                    self._runner = None
                    self._site = None
                    self._app = None

    def _set_rollout_error(self, rollout_id: str, error: BaseException) -> None:
        """Attach `error` to the rollout's state if one is registered and
        unset. First error wins — later failures (e.g. the downstream
        `response_future` raising too) should not clobber the original cause.
        """
        context = self.active_rollouts.get(rollout_id)
        if context is None:
            return
        state = context.get("state")
        if state is None or state.get("error"):
            return
        state["error"] = error

    def register_rollout(
        self, rollout_id: str, state: dict[str, Any] | None = None
    ) -> asyncio.Queue:
        request_queue: asyncio.Queue = asyncio.Queue()
        self.active_rollouts[rollout_id] = {
            "request_id_queue": request_queue,
            "state": state,
        }
        return request_queue

    def unregister_rollout(self, rollout_id: str) -> None:
        # Cancel any pending intercepts for this rollout
        for request_id in list(self.intercepts.keys()):
            intercept = self.intercepts.get(request_id)
            if intercept and intercept.get("rollout_id") == rollout_id:
                # Signal chunk queue to exit for streaming requests
                chunk_queue = intercept.get("chunk_queue")
                if chunk_queue is not None:
                    try:
                        chunk_queue.put_nowait(None)
                    except asyncio.QueueFull:
                        pass
                # Cancel pending future to unblock HTTP handler
                future = intercept.get("response_future")
                if future and not future.done():
                    future.cancel()
                del self.intercepts[request_id]

        if rollout_id in self.active_rollouts:
            del self.active_rollouts[rollout_id]

    async def _handle_request(self, request: Any) -> Any:
        rollout_id = request.match_info["rollout_id"]
        context = self.active_rollouts.get(rollout_id)
        if not context:
            return web.json_response({"error": "Rollout not found"}, status=404)

        try:
            request_body = await request.json()
        except Exception as e:
            return web.json_response({"error": f"Invalid JSON: {e}"}, status=400)

        _log_request(rollout_id, request_body)

        is_streaming = request_body.get("stream", False)
        request_id = f"req_{uuid.uuid4().hex[:8]}"

        chunk_queue: asyncio.Queue[dict | None] | None = (
            asyncio.Queue() if is_streaming else None
        )

        intercept = {
            "request_id": request_id,
            "rollout_id": rollout_id,
            "messages": request_body["messages"],
            "model": request_body.get("model"),
            "tools": request_body.get("tools"),
            "stream": is_streaming,
            "chunk_queue": chunk_queue,
            "response_future": asyncio.Future(),
            "headers": {k.lower(): v for k, v in request.headers.items()},
        }

        self.intercepts[request_id] = intercept
        await context["request_id_queue"].put(request_id)

        if is_streaming:
            return await self._handle_streaming_response(request, rollout_id, intercept)
        else:
            try:
                response_future = cast(
                    asyncio.Future[Any], intercept["response_future"]
                )
                response = await response_future
            except asyncio.CancelledError:
                return web.json_response({"error": "Rollout cancelled"}, status=499)
            except Exception as e:
                logger.debug(
                    f"[{rollout_id}] Rollout error surfaced in non-streaming request: {type(e).__name__}: {e}"
                )
                return web.json_response({"error": str(e)}, status=500)

            response_dict = serialize_intercept_response(response)

            _log_response(rollout_id, response_dict)
            return web.json_response(response_dict)

    async def _handle_streaming_response(
        self, http_request: Any, rollout_id: str, intercept: dict
    ) -> Any:
        chunk_queue = cast(asyncio.Queue[dict | None], intercept["chunk_queue"])
        response_future = cast(asyncio.Future[Any], intercept["response_future"])

        response = web.StreamResponse(
            status=200,
            headers={
                "Content-Type": "text/event-stream",
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
            },
        )

        # Cache peer address + start timer before prepare() so pre-loop
        # failures have something to log.
        start = time.monotonic()
        peer: tuple | None = None
        try:
            peer = http_request.transport.get_extra_info("peername")
        except Exception:
            pass

        # Per-rollout SSE chunk-trace file (NDJSON). Env-gated by
        # VF_CHUNK_TRACE_DIR. Each chunk, a DONE sentinel, and every error
        # branch writes one line, so the tail of the file always tells you
        # how the stream ended even on crash.
        chunk_trace = _open_chunk_trace(rollout_id)
        _write_chunk_trace(
            chunk_trace,
            {"ts": time.time(), "event": "open", "rollout_id": rollout_id},
        )

        # prepare() sends the response headers and attaches the writer to the
        # transport. If the transport is already half-open (e.g. the peer
        # closed between accept and our first write), prepare itself can
        # raise, and the old code path let that exception escape silently.
        # Surface it as StreamInterrupted so the rollout is rescheduled.
        try:
            await response.prepare(http_request)
        except Exception as e:
            diag = _tcp_diag(http_request.transport, peer)
            logger.warning(
                f"[{rollout_id}] Streaming response.prepare failed {diag}: "
                f"{type(e).__name__}: {e}"
            )
            self._set_rollout_error(
                rollout_id,
                StreamInterrupted(f"prepare failed {diag}: {type(e).__name__}: {e}"),
            )
            _write_chunk_trace(
                chunk_trace,
                {
                    "ts": time.time(),
                    "event": "prepare_failed",
                    "error": f"{type(e).__name__}: {e}",
                    "diag": diag,
                },
            )
            _close_chunk_trace(chunk_trace)
            return response
        # Start the periodic TCP-state sampler (logs state transitions during
        # the stream so we catch CLOSE_WAIT / FIN_WAIT before the write RST).
        # Uses a shared dict so the error path can read the last-seen state
        # even when getsockopt(TCP_INFO) fails at error time (socket closed).
        tcp_state_ref: dict = {}
        tcp_sampler: asyncio.Task | None = None
        if TCP_SAMPLE_INTERVAL_SECONDS > 0:
            tcp_sampler = asyncio.create_task(
                _sample_tcp_state(
                    http_request.transport,
                    rollout_id,
                    start,
                    TCP_SAMPLE_INTERVAL_SECONDS,
                    tcp_state_ref,
                )
            )
        # Reuse a single get() task across keepalive cycles instead of
        # recreating it each iteration. ``asyncio.wait_for`` on Python
        # 3.10/3.11 has a race where a timeout cancels an inner task that
        # may have already dequeued an item, silently dropping it.
        # ``asyncio.wait`` does not cancel its tasks on timeout, so a
        # pending ``get()`` task carries forward safely.
        get_task: asyncio.Task | None = None
        try:
            while True:
                if get_task is None:
                    get_task = asyncio.create_task(chunk_queue.get())
                done, _ = await asyncio.wait(
                    {get_task}, timeout=KEEPALIVE_INTERVAL_SECONDS
                )
                if get_task not in done:
                    # Idle window — emit SSE keepalive comment to keep
                    # intermediaries (tunnel, LB, kube-proxy) from closing
                    # the connection during the long vLLM wait.
                    try:
                        await response.write(b": keepalive\n\n")
                    except Exception as e:
                        waited_s = time.monotonic() - start
                        diag = _tcp_diag(http_request.transport, peer, tcp_state_ref)
                        logger.debug(
                            f"[{rollout_id}] Streaming error during keepalive "
                            f"after {waited_s:.1f}s {diag}: {e}"
                        )
                        self._set_rollout_error(
                            rollout_id,
                            StreamInterrupted(
                                f"keepalive write failed after {waited_s:.1f}s "
                                f"{diag}: {type(e).__name__}: {e}"
                            ),
                        )
                        _write_chunk_trace(
                            chunk_trace,
                            {
                                "ts": time.time(),
                                "event": "keepalive_failed",
                                "waited_s": waited_s,
                                "error": f"{type(e).__name__}: {e}",
                                "diag": diag,
                            },
                        )
                        _close_chunk_trace(chunk_trace)
                        return response
                    continue

                chunk_dict = get_task.result()
                get_task = None

                if chunk_dict is None:
                    await response.write(b"data: [DONE]\n\n")
                    _write_chunk_trace(
                        chunk_trace,
                        {"ts": time.time(), "event": "done"},
                    )
                    break

                chunk_json = json.dumps(chunk_dict)
                await response.write(f"data: {chunk_json}\n\n".encode())
                _write_chunk_trace(chunk_trace, _chunk_trace_record(chunk_dict))

        except asyncio.CancelledError:
            logger.debug(f"[{rollout_id}] Streaming cancelled")
            _write_chunk_trace(chunk_trace, {"ts": time.time(), "event": "cancelled"})
        except Exception as e:
            waited_s = time.monotonic() - start
            diag = _tcp_diag(http_request.transport, peer, tcp_state_ref)
            logger.warning(
                f"[{rollout_id}] Streaming error after {waited_s:.1f}s {diag}: {e}"
            )
            self._set_rollout_error(
                rollout_id,
                StreamInterrupted(
                    f"stream write failed after {waited_s:.1f}s "
                    f"{diag}: {type(e).__name__}: {e}"
                ),
            )
            _write_chunk_trace(
                chunk_trace,
                {
                    "ts": time.time(),
                    "event": "stream_interrupted",
                    "waited_s": waited_s,
                    "error": f"{type(e).__name__}: {e}",
                    "diag": diag,
                },
            )
            _close_chunk_trace(chunk_trace)
            return response
        finally:
            if get_task is not None and not get_task.done():
                get_task.cancel()
            if tcp_sampler is not None and not tcp_sampler.done():
                tcp_sampler.cancel()

        try:
            await response_future
        except BaseException as e:
            logger.debug(
                f"[{rollout_id}] Rollout error surfaced in stream: {type(e).__name__}: {e}"
            )

        # write_eof sends the final empty chunk and flushes the transport.
        # A failure here means the client/peer went away between the last
        # chunk and the trailing newline. Prior code caught only
        # ConnectionResetError at DEBUG — surface any failure as
        # StreamInterrupted so the rollout is rescheduled instead of being
        # accepted as a clean completion with 0 turns.
        try:
            await response.write_eof()
        except Exception as e:
            waited_s = time.monotonic() - start
            diag = _tcp_diag(http_request.transport, peer, tcp_state_ref)
            logger.warning(
                f"[{rollout_id}] write_eof failed after {waited_s:.1f}s {diag}: "
                f"{type(e).__name__}: {e}"
            )
            self._set_rollout_error(
                rollout_id,
                StreamInterrupted(
                    f"write_eof failed after {waited_s:.1f}s "
                    f"{diag}: {type(e).__name__}: {e}"
                ),
            )
            _write_chunk_trace(
                chunk_trace,
                {
                    "ts": time.time(),
                    "event": "write_eof_failed",
                    "waited_s": waited_s,
                    "error": f"{type(e).__name__}: {e}",
                    "diag": diag,
                },
            )
        _close_chunk_trace(chunk_trace)
        return response


def deliver_response(
    intercept: dict,
    response: Response | ChatCompletion | None,
    error: BaseException | None = None,
) -> None:
    future = intercept.get("response_future")
    if future and not future.done():
        if error is not None:
            future.set_exception(error)
        elif response is not None:
            future.set_result(response)


async def synthesize_stream(
    intercept: dict, response: Response | None, error: BaseException | None = None
) -> None:
    """Deliver a complete ChatCompletion as synthetic SSE chunks to the agent.

    Allows the base-class get_model_response (non-streaming, TITO-aware) to be
    used for the vLLM call while still satisfying agents that request streaming.

    Protocol (must match _handle_streaming_response):
      put chunk(s) on chunk_queue → put None (EOF) → resolve response_future.
    """
    chunk_queue = cast(
        asyncio.Queue[dict | None] | None,
        intercept.get("chunk_queue"),
    )
    future = cast(asyncio.Future[Any] | None, intercept.get("response_future"))

    # Error / no-response: unblock queue reader, fail/resolve future
    if error is not None or response is None:
        if chunk_queue is not None:
            try:
                chunk_queue.put_nowait(None)
            except asyncio.QueueFull:
                pass
        if future and not future.done():
            if error is not None:
                future.set_exception(error)
            else:
                future.set_result(None)
        return

    if chunk_queue is None:
        raise RuntimeError("Missing chunk_queue for streaming interception")

    message = response.message

    # Chunk 1: content + tool_calls in delta
    delta_tool_calls = None
    if message.tool_calls:
        delta_tool_calls = [
            ChoiceDeltaToolCall(
                index=i,
                id=tc.id,
                type="function",
                function=ChoiceDeltaToolCallFunction(
                    name=tc.name,
                    arguments=tc.arguments,
                ),
            )
            for i, tc in enumerate(message.tool_calls)
        ]

    delta_content: str | None
    if isinstance(message.content, str):
        delta_content = message.content
    elif isinstance(message.content, list):
        text_parts: list[str] = []
        for part in message.content:
            text = (
                part.get("text")
                if isinstance(part, dict)
                else getattr(part, "text", None)
            )
            if isinstance(text, str):
                text_parts.append(text)
        delta_content = "".join(text_parts) if text_parts else None
    else:
        delta_content = None

    content_chunk = ChatCompletionChunk(
        id=response.id,
        choices=[
            ChunkChoice(
                index=0,
                delta=ChoiceDelta(
                    role="assistant",
                    content=delta_content,
                    tool_calls=delta_tool_calls,
                ),
                finish_reason=None,
            )
        ],
        created=response.created,
        model=response.model,
        object="chat.completion.chunk",
    )
    content_chunk_dict = content_chunk.model_dump()
    if message.reasoning_content:
        content_chunk_dict["choices"][0]["delta"]["reasoning_content"] = (
            message.reasoning_content
        )
    await chunk_queue.put(content_chunk_dict)

    # Chunk 2: finish_reason only
    finish_chunk = ChatCompletionChunk(
        id=response.id,
        choices=[
            ChunkChoice(
                index=0,
                delta=ChoiceDelta(),
                finish_reason=message.finish_reason,
            )
        ],
        created=response.created,
        model=response.model,
        object="chat.completion.chunk",
    )
    finish_chunk_dict = finish_chunk.model_dump()
    await chunk_queue.put(finish_chunk_dict)

    # EOF sentinel + resolve future
    await chunk_queue.put(None)
    if future and not future.done():
        future.set_result(response)


def create_empty_completion(model: str) -> ChatCompletion:
    return ChatCompletion(
        id="agent-completed",
        choices=[
            Choice(
                finish_reason="stop",
                index=0,
                message=ChatCompletionMessage(role="assistant", content=""),
            )
        ],
        created=int(time.time()),
        model=model,
        object="chat.completion",
    )


# Logging helpers


def _response_content_to_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        text_parts: list[str] = []
        for part in content:
            if isinstance(part, dict):
                text = part.get("text")
            else:
                text = getattr(part, "text", None)
            if isinstance(text, str):
                text_parts.append(text)
        return "".join(text_parts)
    return ""


def serialize_intercept_response(response: Any) -> dict[str, Any]:
    """Serialize intercepted responses to OpenAI ChatCompletion JSON shape."""
    if isinstance(response, Response):
        message = response.message
        tool_calls = []
        for tc in message.tool_calls or []:
            tool_calls.append(
                {
                    "id": tc.id,
                    "type": "function",
                    "function": {
                        "name": tc.name,
                        "arguments": tc.arguments,
                    },
                }
            )

        message_payload: dict[str, Any] = {
            "role": "assistant",
            "content": _response_content_to_text(message.content),
        }
        if tool_calls:
            message_payload["tool_calls"] = tool_calls
        if message.reasoning_content is not None:
            message_payload["reasoning_content"] = message.reasoning_content

        choice: dict[str, Any] = {
            "index": 0,
            "message": message_payload,
            "finish_reason": message.finish_reason,
        }

        output = {
            "id": response.id,
            "object": "chat.completion",
            "created": response.created,
            "model": response.model,
            "choices": [choice],
        }

        if response.usage is not None:
            output["usage"] = response.usage.model_dump(exclude_none=True)

        return output

    if hasattr(response, "model_dump"):
        return response.model_dump()
    return dict(response)


def _log_request(rollout_id: str, body: dict) -> None:
    """Log an intercepted request."""
    if not logger.isEnabledFor(logging.DEBUG):
        return
    log_msg = f"[{rollout_id}] <- INTERCEPTED REQUEST"
    tools = body.get("tools", [])
    log_msg += f" ({len(tools)} tool(s))"
    if tools:
        log_msg += f"\n[tools] {', '.join([tool.get('function', {}).get('name', '?') for tool in tools])}"
    for msg in body.get("messages", []):
        content = msg.get("content", "")
        if isinstance(content, str):
            log_msg += f"\n[{msg.get('role', '?')}] {truncate(content)}"
        else:
            log_msg += f"\n[{msg.get('role', '?')}] <complex content>"
        for tc in msg.get("tool_calls") or []:
            func = tc.get("function", {})
            log_msg += f"\n[tool_call]\n{func.get('name')}({truncate(func.get('arguments', ''), 100)})"
    logger.debug(log_msg)


def _log_response(rollout_id: str, response: dict) -> None:
    """Log the response from the model."""
    if not logger.isEnabledFor(logging.DEBUG):
        return
    log_msg = f"[{rollout_id}] -> RESPONSE"
    msg = response.get("choices", [{}])[0].get("message", {})
    if msg.get("content"):
        log_msg += f"\n[assistant]\n{truncate(msg['content'])}"
    for tc in msg.get("tool_calls") or []:
        func = tc.get("function", {})
        log_msg += f"\n[tool_call]\n{func.get('name')}({truncate(func.get('arguments', ''), 100)})"
    logger.debug(log_msg)
