"""Utilities for intercepting API calls from agents running in sandboxes."""

import asyncio
import json
import logging
import time
import uuid
from typing import Any, cast

from aiohttp import web
from openai import AsyncOpenAI
from openai.types.chat import (
    ChatCompletion,
    ChatCompletionMessage,
    ChatCompletionMessageToolCall,
)
from openai.types.chat.chat_completion import Choice
from openai.types.chat.chat_completion_message_tool_call import Function

from verifiers.types import (
    ChatCompletionToolParam,
    Messages,
    ModelResponse,
    SamplingArgs,
    State,
)

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

        await response.write_eof()
        return response


async def get_streaming_model_response(
    state: State,
    prompt: Messages,
    intercept: dict,
    client: AsyncOpenAI | None = None,
    model: str | None = None,
    oai_tools: list[ChatCompletionToolParam] | None = None,
    sampling_args: SamplingArgs | None = None,
) -> ChatCompletion:
    """
    Handle streaming API call, forwarding chunks and accumulating response.

    This function makes a streaming API call to the model, forwards each chunk
    to the waiting HTTP handler via the chunk queue, and accumulates the full
    response to return.
    """
    chunk_queue = cast(asyncio.Queue, intercept["chunk_queue"])

    client = client or state["client"]
    model = model or state["model"]
    sampling_args = sampling_args or state.get("sampling_args") or {}

    # Convert max_tokens to max_completion_tokens for chat
    if "max_tokens" in sampling_args:
        sampling_args = dict(sampling_args)
        max_tokens = sampling_args.pop("max_tokens")
        if "max_completion_tokens" not in sampling_args:
            sampling_args["max_completion_tokens"] = max_tokens

    create_kwargs: dict[str, Any] = {
        "model": model,
        "messages": prompt,
        "stream": True,
    }
    if oai_tools:
        create_kwargs["tools"] = oai_tools
    create_kwargs.update(sampling_args)

    stream = await client.chat.completions.create(**create_kwargs)

    accumulated_content = ""
    accumulated_tool_calls: dict[int, dict] = {}
    finish_reason = None
    completion_id = None
    created_time = int(time.time())
    stream_ended = False

    try:
        async for chunk in stream:
            await chunk_queue.put(chunk)

            if not completion_id and chunk.id:
                completion_id = chunk.id
            if chunk.created:
                created_time = chunk.created

            if chunk.choices:
                choice = chunk.choices[0]
                if choice.finish_reason:
                    finish_reason = choice.finish_reason

                delta = choice.delta
                if delta:
                    if delta.content:
                        accumulated_content += delta.content

                    if delta.tool_calls:
                        for tc in delta.tool_calls:
                            idx = tc.index
                            if idx not in accumulated_tool_calls:
                                accumulated_tool_calls[idx] = {
                                    "id": tc.id or "",
                                    "type": tc.type or "function",
                                    "function": {"name": "", "arguments": ""},
                                }
                            if tc.id:
                                accumulated_tool_calls[idx]["id"] = tc.id
                            if tc.function:
                                if tc.function.name:
                                    accumulated_tool_calls[idx]["function"]["name"] = (
                                        tc.function.name
                                    )
                                if tc.function.arguments:
                                    accumulated_tool_calls[idx]["function"][
                                        "arguments"
                                    ] += tc.function.arguments

        await chunk_queue.put(None)
        stream_ended = True
    finally:
        if not stream_ended:
            try:
                chunk_queue.put_nowait(None)
            except asyncio.QueueFull:
                pass

    tool_calls_list = None
    if accumulated_tool_calls:
        tool_calls_list = [
            ChatCompletionMessageToolCall(
                id=tc_data["id"],
                type="function",
                function=Function(
                    name=tc_data["function"]["name"],
                    arguments=tc_data["function"]["arguments"],
                ),
            )
            for idx, tc_data in sorted(accumulated_tool_calls.items())
        ]

    message = ChatCompletionMessage(
        role="assistant",
        content=accumulated_content if accumulated_content else None,
        tool_calls=tool_calls_list,
    )

    result = ChatCompletion(
        id=completion_id or f"chatcmpl-{uuid.uuid4().hex[:8]}",
        choices=[
            Choice(
                finish_reason=finish_reason or "stop",
                index=0,
                message=message,
            )
        ],
        created=created_time,
        model=model,
        object="chat.completion",
    )

    rollout_id = intercept.get("rollout_id", "?")
    _log_response(rollout_id, result.model_dump())

    return result


def deliver_response(
    intercept: dict, response: ModelResponse | None, error: BaseException | None = None
) -> None:
    future = intercept.get("response_future")
    if future and not future.done():
        if error is not None:
            future.set_exception(error)
        elif response is not None:
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
