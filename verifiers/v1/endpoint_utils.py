from __future__ import annotations

import asyncio
import logging
import os
import time
import uuid
from collections.abc import Callable
from typing import Any, cast

from openai import AsyncOpenAI
from openai.types.chat import ChatCompletion

from verifiers.errors import Error, TunnelError
from verifiers.types import Messages, Tool
from verifiers.utils.async_utils import maybe_call_with_named_args
from verifiers.utils.error_utils import error_info
from verifiers.utils.interception_utils import (
    InterceptionServer,
    deliver_response,
    synthesize_stream,
)
from verifiers.utils.message_utils import normalize_messages
from verifiers.utils.serve_utils import get_free_port

from .runtime import Runtime
from .state import State
from .task import Task


class Endpoint:
    TUNNEL_CHECK_INTERVAL = 60.0

    def __init__(
        self,
        port: int | None = None,
        secret: str | None = None,
        use_tunnel: bool = False,
        logger: logging.Logger | None = None,
    ):
        self.port = get_free_port() if port is None else port
        self.secret = secret or os.environ.get("ENDPOINT_SECRET")
        self.use_tunnel = use_tunnel
        self.logger = logger or logging.getLogger(__name__)
        self.server = InterceptionServer(self.port, secret=self.secret)
        self._tunnel: object | None = None
        self._tunnel_lock = asyncio.Lock()
        self._tunnel_last_checked = 0.0
        self._request_queues: dict[str, asyncio.Queue[str]] = {}

    async def start(self) -> None:
        await self.server.start()

    async def register_rollout(self, state: State) -> str:
        await self.start()
        request_key = f"rollout_{uuid.uuid4().hex[:8]}"
        request_queue = self.server.register_rollout(request_key, state=state)
        self._request_queues[request_key] = cast(asyncio.Queue[str], request_queue)
        state["endpoint_request_key"] = request_key
        state["endpoint_base_url"] = f"{await self.url_base()}/rollout/{request_key}/v1"
        return state["endpoint_base_url"]

    def client(self, state: State) -> AsyncOpenAI:
        return AsyncOpenAI(
            api_key=self.secret or "intercepted",
            base_url=str(state["endpoint_base_url"]),
        )

    def unregister_rollout(self, request_key: str) -> None:
        self._request_queues.pop(request_key, None)
        self.server.unregister_rollout(request_key)

    def request_queue(self, request_key: str) -> asyncio.Queue[str]:
        return self._request_queues[request_key]

    def get_request(self, request_id: str) -> dict[str, object]:
        return cast(dict[str, object], self.server.intercepts[request_id])

    async def url_base(self) -> str:
        if self.use_tunnel:
            return await self.get_tunnel_url()
        return f"http://127.0.0.1:{self.port}"

    async def get_tunnel_url(self) -> str:
        from prime_tunnel import Tunnel

        async with self._tunnel_lock:
            tunnel = cast(Any, self._tunnel)
            if tunnel is not None and not tunnel.is_running:
                tunnel.sync_stop()
                self._tunnel = None

            tunnel = cast(Any, self._tunnel)
            if tunnel is not None:
                now = time.time()
                if now - self._tunnel_last_checked > self.TUNNEL_CHECK_INTERVAL:
                    self._tunnel_last_checked = now
                    if not await tunnel.check_registered():
                        tunnel.sync_stop()
                        self._tunnel = None

            if self._tunnel is None:
                tunnel = Tunnel(local_port=self.port)
                url = await tunnel.start()
                self._tunnel = tunnel
                self._tunnel_last_checked = time.time()
                return str(url)

            tunnel = cast(Any, self._tunnel)
            if tunnel.url is None:
                raise TunnelError("Tunnel started but URL is unavailable.")
            return str(tunnel.url)

    async def check_tunnel(self) -> None:
        tunnel = cast(Any, self._tunnel)
        if tunnel is not None and not tunnel.is_running:
            raise TunnelError("Tunnel process died during rollout.")

    async def teardown(self) -> None:
        async with self._tunnel_lock:
            tunnel = cast(Any, self._tunnel)
            if tunnel is not None:
                tunnel.sync_stop()
                self._tunnel = None
        await self.server.stop()


async def run_intercepted_program(
    program: Callable[..., object],
    endpoint: Endpoint,
    runtime: Runtime,
    task: Task,
    state: State,
) -> object:
    await endpoint.register_rollout(state)
    client = endpoint.client(state)
    execution = asyncio.create_task(
        maybe_call_with_named_args(program, task=task, state=state, client=client)
    )
    request_key = str(state["endpoint_request_key"])
    queue = endpoint.request_queue(request_key)
    try:
        while True:
            request_id = await next_request(queue, execution, endpoint)
            if request_id is None:
                break
            await forward_request(endpoint, runtime, task, state, request_id)
        return await execution
    finally:
        if not execution.done():
            execution.cancel()
            await asyncio.gather(execution, return_exceptions=True)
        endpoint.unregister_rollout(request_key)
        await client.close()


async def next_request(
    queue: asyncio.Queue[str], execution: asyncio.Task[object], endpoint: Endpoint
) -> str | None:
    while True:
        if execution.done():
            if queue.empty():
                return None
            return queue.get_nowait()
        queue_task = asyncio.create_task(queue.get())
        try:
            done, _ = await asyncio.wait(
                {queue_task, execution},
                timeout=1.0,
                return_when=asyncio.FIRST_COMPLETED,
            )
            if queue_task in done:
                return queue_task.result()
            if execution in done:
                if queue.empty():
                    return None
                return queue.get_nowait()
            await endpoint.check_tunnel()
        finally:
            if not queue_task.done():
                queue_task.cancel()
                await asyncio.gather(queue_task, return_exceptions=True)


async def forward_request(
    endpoint: Endpoint,
    runtime: Runtime,
    task: Task,
    state: State,
    request_id: str,
) -> None:
    request = endpoint.get_request(request_id)
    prompt = normalize_endpoint_messages(request["messages"])
    tool_defs = normalize_endpoint_tools(request.get("tools"))
    response = None
    error: BaseException | None = None
    try:
        response = await runtime.submit_model_request(
            prompt,
            task,
            state,
            tool_defs=tool_defs,
            extras={"endpoint": True, "endpoint_request_id": request_id},
        )
    except BaseException as e:
        error = e
        if isinstance(e, Error):
            state["error"] = error_info(e)
        raise
    finally:
        if bool(request.get("stream")):
            await synthesize_stream(request, response, error)
        else:
            deliver_response(request, response, error)


def normalize_endpoint_messages(messages: object) -> Messages:
    if isinstance(messages, str):
        return normalize_messages(messages, field_name="endpoint.messages")
    if isinstance(messages, list):
        return normalize_messages(
            cast(Messages, messages), field_name="endpoint.messages"
        )
    raise TypeError("Endpoint messages must be vf.Messages or str.")


def normalize_endpoint_tools(tools: object) -> list[Tool] | None:
    if tools is None:
        return None
    if not isinstance(tools, list):
        raise TypeError("Endpoint tools must be a list.")
    normalized: list[Tool] = []
    for raw_tool in tools:
        if isinstance(raw_tool, Tool):
            normalized.append(raw_tool)
            continue
        if not isinstance(raw_tool, dict):
            raise TypeError("Endpoint tool definitions must be dicts.")
        raw_tool = cast(dict[str, Any], raw_tool)
        function_payload = raw_tool.get("function")
        if raw_tool.get("type") == "function" and isinstance(function_payload, dict):
            normalized.append(
                Tool(
                    name=str(function_payload.get("name", "")),
                    description=str(function_payload.get("description", "")),
                    parameters=cast(
                        dict[str, Any], function_payload.get("parameters") or {}
                    ),
                    strict=cast(bool | None, function_payload.get("strict")),
                )
            )
        else:
            normalized.append(Tool.model_validate(raw_tool))
    return normalized


def append_openai_message(
    messages: list[dict[str, object]], response: ChatCompletion
) -> None:
    choice = response.choices[0]
    message = choice.message
    payload = message.model_dump(exclude_none=True)
    messages.append(cast(dict[str, object], payload))


def openai_tool_calls(response: ChatCompletion) -> list[Any]:
    choice = response.choices[0]
    message = choice.message
    return list(message.tool_calls or [])


def assistant_completion_from_messages(
    prompt: list[dict[str, object]], messages: list[dict[str, object]]
) -> list[dict[str, object]]:
    return messages[len(prompt) :]
