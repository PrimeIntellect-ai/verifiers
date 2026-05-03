"""Utilities for intercepting API calls from agents running in sandboxes."""

import asyncio
import hmac
import inspect
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
    ChoiceDelta,
    ChoiceDeltaToolCall,
    ChoiceDeltaToolCallFunction,
)
from openai.types.chat.chat_completion_chunk import (
    Choice as ChunkChoice,
)

from verifiers.errors import InfraError
from verifiers.types import Response
from verifiers.utils.logging_utils import print_time, truncate

logger = logging.getLogger(__name__)


KEEPALIVE_INTERVAL_SECONDS = 10.0


class StreamInterrupted(InfraError):
    """Raised when the intercepted streaming response to the agent is cut short.

    Without this, a mid-stream transport failure would be swallowed here and
    the agent would observe a truncated (but syntactically valid) SSE stream,
    often exiting with code 0 and an empty trajectory — bypassing the
    non-zero-exit error capture in `CliAgentEnv.poll_job_completion`.
    """


def protocol_from_path(path: str) -> str:
    if path.endswith("/v1/messages"):
        return "anthropic_messages"
    if path.endswith("/v1/responses"):
        return "openai_responses"
    return "openai_chat_completions"


class InterceptionServer:
    """
    HTTP server that intercepts API requests from agents.

    Requests are queued for processing, and responses are delivered back
    to the agent once the actual model response is obtained.
    """

    def __init__(self, port: int, secret: str | None = None):
        self.port = port
        self.secret = secret or None  # treat empty string as no secret
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
            app.router.add_post(
                "/rollout/{rollout_id}/v1/responses",
                self._handle_request,
            )
            app.router.add_post(
                "/rollout/{rollout_id}/v1/messages",
                self._handle_request,
            )
            app.router.add_post(
                "/rollout/{rollout_id}/vf/tools/{tool_name}",
                self._handle_tool_request,
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
        self,
        rollout_id: str,
        state: dict[str, Any] | None = None,
        tool_handler: Any | None = None,
    ) -> asyncio.Queue:
        request_queue: asyncio.Queue = asyncio.Queue()
        self.active_rollouts[rollout_id] = {
            "request_id_queue": request_queue,
            "state": state,
            "tool_handler": tool_handler,
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
        if self.secret:
            auth = request.headers.get("Authorization", "")
            if not hmac.compare_digest(auth, f"Bearer {self.secret}"):
                return web.json_response({"error": "Unauthorized"}, status=401)

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

        protocol = protocol_from_path(str(request.path))
        intercept = {
            "request_id": request_id,
            "rollout_id": rollout_id,
            "protocol": protocol,
            "messages": request_body.get("messages"),
            "input": request_body.get("input"),
            "system": request_body.get("system"),
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

            response_dict = serialize_intercept_response(
                response,
                protocol=str(intercept["protocol"]),
            )

            _log_response(rollout_id, response_dict)
            return web.json_response(response_dict)

    async def _handle_tool_request(self, request: Any) -> Any:
        if self.secret:
            auth = request.headers.get("Authorization", "")
            if not hmac.compare_digest(auth, f"Bearer {self.secret}"):
                return web.json_response({"error": "Unauthorized"}, status=401)

        rollout_id = request.match_info["rollout_id"]
        context = self.active_rollouts.get(rollout_id)
        if not context:
            return web.json_response({"error": "Rollout not found"}, status=404)
        tool_handler = context.get("tool_handler")
        if tool_handler is None:
            return web.json_response({"error": "Tool proxy unavailable"}, status=404)

        try:
            request_body = await request.json()
        except Exception as e:
            return web.json_response({"error": f"Invalid JSON: {e}"}, status=400)
        arguments = request_body.get("arguments") or {}
        if not isinstance(arguments, dict):
            return web.json_response(
                {"error": "Tool arguments must be an object"}, status=400
            )

        try:
            result = tool_handler(request.match_info["tool_name"], arguments)
            if inspect.isawaitable(result):
                result = await result
        except Exception as e:
            return web.json_response({"error": str(e)}, status=500)
        result = json.loads(json.dumps(result, default=str))
        return web.json_response({"result": result})

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

        start = time.monotonic()

        # Half-open transport at accept raises here; surface it so the
        # rollout reschedules instead of looking like a clean empty stream.
        try:
            await response.prepare(http_request)
        except Exception as e:
            logger.warning(
                f"[{rollout_id}] Streaming response.prepare failed: "
                f"{type(e).__name__}: {e}"
            )
            self._set_rollout_error(
                rollout_id,
                StreamInterrupted(f"prepare failed: {type(e).__name__}: {e}"),
            )
            return response
        # Reuse one get() task across keepalive cycles; asyncio.wait_for on
        # Py 3.10/3.11 can silently drop an item when its timeout cancels.
        get_task: asyncio.Task | None = None
        try:
            while True:
                if get_task is None:
                    get_task = asyncio.create_task(chunk_queue.get())
                done, _ = await asyncio.wait(
                    {get_task}, timeout=KEEPALIVE_INTERVAL_SECONDS
                )
                if get_task not in done:
                    # SSE comment keeps the TCP path warm across the vLLM wait
                    # so idle-timeouts in any intermediary don't reap it.
                    try:
                        await response.write(b": keepalive\n\n")
                    except Exception as e:
                        waited_s = time.monotonic() - start
                        logger.debug(
                            f"[{rollout_id}] Streaming error during keepalive "
                            f"after {print_time(waited_s)}: {e}"
                        )
                        self._set_rollout_error(
                            rollout_id,
                            StreamInterrupted(
                                f"keepalive write failed after {print_time(waited_s)}: "
                                f"{type(e).__name__}: {e}"
                            ),
                        )
                        return response
                    continue

                chunk_dict = get_task.result()
                get_task = None

                if chunk_dict is None:
                    await response.write(b"data: [DONE]\n\n")
                    break

                chunk_json = json.dumps(chunk_dict)
                await response.write(f"data: {chunk_json}\n\n".encode())
                # Force a loop yield so the transport flushes before close;
                # otherwise burst contention can truncate the final chunk.
                await asyncio.sleep(0)

        except asyncio.CancelledError:
            logger.debug(f"[{rollout_id}] Streaming cancelled")
        except Exception as e:
            waited_s = time.monotonic() - start
            logger.debug(
                f"[{rollout_id}] Streaming error after {print_time(waited_s)}: {e}"
            )
            self._set_rollout_error(
                rollout_id,
                StreamInterrupted(
                    f"stream write failed after {print_time(waited_s)}: "
                    f"{type(e).__name__}: {e}"
                ),
            )
            return response
        finally:
            if get_task is not None and not get_task.done():
                get_task.cancel()

        try:
            await response_future
        except BaseException as e:
            logger.debug(
                f"[{rollout_id}] Rollout error surfaced in stream: {type(e).__name__}: {e}"
            )

        # Surface any write_eof failure so a tail truncation becomes a
        # reschedulable error instead of a silent zero-turn completion.
        try:
            await response.write_eof()
        except Exception as e:
            waited_s = time.monotonic() - start
            logger.warning(
                f"[{rollout_id}] write_eof failed after {print_time(waited_s)}: "
                f"{type(e).__name__}: {e}"
            )
            self._set_rollout_error(
                rollout_id,
                StreamInterrupted(
                    f"write_eof failed after {print_time(waited_s)}: "
                    f"{type(e).__name__}: {e}"
                ),
            )
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

    Allows the base-class get_model_response (non-streaming) to be
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


def serialize_intercept_response(
    response: Any, protocol: str = "openai_chat_completions"
) -> dict[str, Any]:
    """Serialize intercepted responses to the requested endpoint protocol shape."""
    if isinstance(response, Response):
        if protocol == "anthropic_messages":
            return serialize_anthropic_message_response(response)
        if protocol == "openai_responses":
            return serialize_openai_responses_response(response)
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


def serialize_anthropic_message_response(response: Response) -> dict[str, Any]:
    content: list[dict[str, Any]] = []
    message = response.message
    if message.content:
        content.append(
            {"type": "text", "text": _response_content_to_text(message.content)}
        )
    for tool_call in message.tool_calls or []:
        try:
            tool_input = json.loads(tool_call.arguments)
        except json.JSONDecodeError:
            tool_input = {"arguments": tool_call.arguments}
        content.append(
            {
                "type": "tool_use",
                "id": tool_call.id,
                "name": tool_call.name,
                "input": tool_input,
            }
        )
    if not content:
        content.append({"type": "text", "text": ""})
    usage = {}
    if response.usage is not None:
        usage = {
            "input_tokens": response.usage.prompt_tokens,
            "output_tokens": response.usage.completion_tokens,
        }
    return {
        "id": response.id,
        "type": "message",
        "role": "assistant",
        "model": response.model,
        "content": content,
        "stop_reason": "tool_use" if message.tool_calls else "end_turn",
        "stop_sequence": None,
        "usage": usage,
    }


def serialize_openai_responses_response(response: Response) -> dict[str, Any]:
    output: list[dict[str, Any]] = []
    message = response.message
    if message.content:
        output.append(
            {
                "id": f"msg_{response.id}",
                "type": "message",
                "role": "assistant",
                "status": "completed",
                "content": [
                    {
                        "type": "output_text",
                        "text": _response_content_to_text(message.content),
                        "annotations": [],
                    }
                ],
            }
        )
    for tool_call in message.tool_calls or []:
        output.append(
            {
                "id": tool_call.id,
                "type": "function_call",
                "call_id": tool_call.id,
                "name": tool_call.name,
                "arguments": tool_call.arguments,
                "status": "completed",
            }
        )
    usage = None
    if response.usage is not None:
        usage = {
            "input_tokens": response.usage.prompt_tokens,
            "output_tokens": response.usage.completion_tokens,
            "total_tokens": response.usage.total_tokens,
        }
    return {
        "id": response.id,
        "object": "response",
        "created_at": float(response.created),
        "status": "completed",
        "model": response.model,
        "output": output,
        "parallel_tool_calls": True,
        "tool_choice": "auto",
        "tools": [],
        "usage": usage,
    }


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
