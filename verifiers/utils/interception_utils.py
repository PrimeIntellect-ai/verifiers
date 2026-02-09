"""Utilities for intercepting API calls from agents running in sandboxes."""

import asyncio
import json
import logging
import time
import uuid
from typing import Any, cast

from aiohttp import web
from openai.types.chat import (
    ChatCompletion,
    ChatCompletionMessage,
)
from openai.types.chat.chat_completion import Choice
from openai.types.chat.chat_completion_chunk import (
    ChatCompletionChunk,
    Choice as ChunkChoice,
    ChoiceDelta,
    ChoiceDeltaToolCall,
    ChoiceDeltaToolCallFunction,
)

from verifiers.types import ModelResponse

logger = logging.getLogger(__name__)


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

    def register_rollout(self, rollout_id: str) -> asyncio.Queue:
        request_queue: asyncio.Queue = asyncio.Queue()
        self.active_rollouts[rollout_id] = {
            "request_id_queue": request_queue,
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

        chunk_queue: asyncio.Queue | None = asyncio.Queue() if is_streaming else None

        intercept = {
            "request_id": request_id,
            "rollout_id": rollout_id,
            "messages": request_body["messages"],
            "model": request_body.get("model"),
            "tools": request_body.get("tools"),
            "stream": is_streaming,
            "chunk_queue": chunk_queue,
            "response_future": asyncio.Future(),
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
                logger.error(f"Error processing intercepted request: {e}")
                return web.json_response({"error": str(e)}, status=500)

            response_dict = (
                response.model_dump()
                if hasattr(response, "model_dump")
                else dict(response)
            )

            _log_response(rollout_id, response_dict)
            return web.json_response(response_dict)

    async def _handle_streaming_response(
        self, http_request: Any, rollout_id: str, intercept: dict
    ) -> Any:
        chunk_queue = cast(asyncio.Queue, intercept["chunk_queue"])
        response_future = cast(asyncio.Future[Any], intercept["response_future"])

        response = web.StreamResponse(
            status=200,
            headers={
                "Content-Type": "text/event-stream",
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
            },
        )
        await response.prepare(http_request)

        try:
            while True:
                chunk = await chunk_queue.get()

                if chunk is None:
                    await response.write(b"data: [DONE]\n\n")
                    break

                chunk_dict = (
                    chunk.model_dump() if hasattr(chunk, "model_dump") else dict(chunk)
                )
                chunk_json = json.dumps(chunk_dict)
                await response.write(f"data: {chunk_json}\n\n".encode())

            await response_future

        except asyncio.CancelledError:
            logger.debug(f"[{rollout_id}] Streaming cancelled")
        except Exception as e:
            logger.error(f"[{rollout_id}] Streaming error: {e}")

        try:
            await response.write_eof()
        except ConnectionResetError:
            logger.debug(f"[{rollout_id}] Client disconnected before write_eof")
        return response


def deliver_response(
    intercept: dict, response: ModelResponse | None, error: BaseException | None = None
) -> None:
    future = intercept.get("response_future")
    if future and not future.done():
        if error is not None:
            future.set_exception(error)
        elif response is not None:
            future.set_result(response)


async def synthesize_stream(
    intercept: dict, response: ModelResponse | None, error: BaseException | None = None
) -> None:
    """Deliver a complete ChatCompletion as synthetic SSE chunks to the agent.

    Allows the base-class get_model_response (non-streaming, TITO-aware) to be
    used for the vLLM call while still satisfying agents that request streaming.

    Protocol (must match _handle_streaming_response):
      put chunk(s) on chunk_queue → put None (EOF) → resolve response_future.
    """
    chunk_queue = intercept.get("chunk_queue")
    future = intercept.get("response_future")

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

    choice = response.choices[0]
    message = choice.message

    # Chunk 1: content + tool_calls in delta
    delta_tool_calls = None
    if message.tool_calls:
        delta_tool_calls = [
            ChoiceDeltaToolCall(
                index=i,
                id=tc.id,
                type="function",
                function=ChoiceDeltaToolCallFunction(
                    name=tc.function.name,
                    arguments=tc.function.arguments,
                ),
            )
            for i, tc in enumerate(message.tool_calls)
        ]

    content_chunk = ChatCompletionChunk(
        id=response.id,
        choices=[
            ChunkChoice(
                index=0,
                delta=ChoiceDelta(
                    role="assistant",
                    content=message.content,
                    tool_calls=delta_tool_calls,
                ),
                finish_reason=None,
            )
        ],
        created=response.created,
        model=response.model,
        object="chat.completion.chunk",
    )
    await chunk_queue.put(content_chunk)

    # Chunk 2: finish_reason only
    finish_chunk = ChatCompletionChunk(
        id=response.id,
        choices=[
            ChunkChoice(
                index=0,
                delta=ChoiceDelta(),
                finish_reason=choice.finish_reason,
            )
        ],
        created=response.created,
        model=response.model,
        object="chat.completion.chunk",
    )
    await chunk_queue.put(finish_chunk)

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


def _truncate(s: str, limit: int = 200) -> str:
    return (s[:limit] + "...") if len(s) > limit else s


def _log_request(rollout_id: str, body: dict) -> None:
    logger.debug(f"[{rollout_id}] <- INTERCEPTED REQUEST")
    for msg in body.get("messages", []):
        content = msg.get("content", "")
        if isinstance(content, str):
            logger.debug(f"  [{msg.get('role', '?')}] {_truncate(content)}")
        else:
            logger.debug(f"  [{msg.get('role', '?')}] <complex content>")
    if body.get("tools"):
        logger.debug(f"  [tools] {len(body['tools'])} tool(s)")


def _log_response(rollout_id: str, response: dict) -> None:
    logger.debug(f"[{rollout_id}] -> RESPONSE")
    msg = response.get("choices", [{}])[0].get("message", {})
    if msg.get("content"):
        logger.debug(f"  [assistant] {_truncate(msg['content'])}")
    for tc in msg.get("tool_calls") or []:
        func = tc.get("function", {})
        logger.debug(
            f"  [tool_call] {func.get('name')}({_truncate(func.get('arguments', ''), 100)})"
        )
