from __future__ import annotations

import asyncio
import json
import logging
import os
import time
import uuid
from collections.abc import Callable, Mapping
from typing import Any, cast

from anthropic import AsyncAnthropic
from openai import AsyncOpenAI

from verifiers.errors import Error, TunnelError
from verifiers.types import (
    AssistantMessage,
    Messages,
    SystemMessage,
    Tool,
    ToolCall,
    ToolMessage,
    UserMessage,
)
from verifiers.utils.async_utils import maybe_call_with_named_args
from verifiers.utils.error_utils import error_info
from verifiers.utils.interception_utils import (
    InterceptionServer,
    deliver_response,
    synthesize_stream,
)
from verifiers.utils.message_utils import normalize_messages
from verifiers.utils.serve_utils import get_free_port

from ..runtime import Runtime
from ..state import State
from ..task import Task


def openai_endpoint_config(
    state: State, client: object | None = None
) -> dict[str, str]:
    api_key = getattr(client, "api_key", None) or "intercepted"
    return {
        "model": str(state["runtime"]["model"]),
        "api_base": str(state["endpoint_base_url"]),
        "api_key": str(api_key),
    }


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

    async def register_rollout(
        self,
        state: State,
        tool_handler: object | None = None,
        tool_defs: object | None = None,
        user_handler: object | None = None,
        stop_handler: object | None = None,
    ) -> str:
        await self.start()
        request_key = f"rollout_{uuid.uuid4().hex[:8]}"
        request_queue = self.server.register_rollout(
            request_key,
            state=state,
            tool_handler=tool_handler,
            tool_defs=tool_defs,
            user_handler=user_handler,
            stop_handler=stop_handler,
        )
        self._request_queues[request_key] = cast(asyncio.Queue[str], request_queue)
        endpoint_root_url = f"{await self.url_base()}/rollout/{request_key}"
        state["endpoint_request_key"] = request_key
        state["endpoint_root_url"] = endpoint_root_url
        state["endpoint_base_url"] = f"{endpoint_root_url}/v1"
        return state["endpoint_base_url"]

    def client(self, state: State) -> AsyncOpenAI | AsyncAnthropic:
        client_type = state.get("runtime", {}).get("client_type")
        if client_type == "anthropic_messages":
            return AsyncAnthropic(
                api_key=self.secret or "intercepted",
                base_url=str(state["endpoint_root_url"]),
            )
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
    async def call_tool(name: str, arguments: Mapping[str, object]) -> object:
        return await runtime.call_tool(name, task, state, **dict(arguments))

    async def call_user(transcript: list[object]) -> object:
        return await runtime.user_messages(task, state, transcript=transcript)

    async def check_stop() -> object:
        return {
            "done": await runtime.is_completed(task, state),
            "stop_condition": state.get("stop_condition"),
        }

    await endpoint.register_rollout(
        state,
        tool_handler=call_tool,
        tool_defs=runtime.tool_defs(state),
        user_handler=call_user,
        stop_handler=check_stop,
    )
    client = endpoint.client(state)
    execution = asyncio.create_task(
        maybe_call_with_named_args(
            program,
            task=task,
            state=state,
            client=client,
            runtime=runtime,
        )
    )
    request_key = str(state["endpoint_request_key"])
    queue = endpoint.request_queue(request_key)
    pending: set[asyncio.Task[None]] = set()
    try:
        while True:
            await raise_finished_forward_errors(pending)
            if execution.done():
                await raise_execution_error(execution)
                if not queue.empty():
                    request_id = queue.get_nowait()
                    pending.add(
                        asyncio.create_task(
                            forward_request(endpoint, runtime, task, state, request_id)
                        )
                    )
                    continue
                if not pending:
                    break
                await asyncio.wait(
                    pending,
                    timeout=1.0,
                    return_when=asyncio.FIRST_COMPLETED,
                )
                await endpoint.check_tunnel()
                continue
            queue_task = asyncio.create_task(queue.get())
            wait_set = {queue_task, execution, *pending}
            try:
                done, _ = await asyncio.wait(
                    wait_set,
                    timeout=1.0,
                    return_when=asyncio.FIRST_COMPLETED,
                )
                if queue_task in done:
                    request_id = queue_task.result()
                    pending.add(
                        asyncio.create_task(
                            forward_request(endpoint, runtime, task, state, request_id)
                        )
                    )
                    continue
                if execution in done:
                    continue
                if pending.intersection(done):
                    continue
                await endpoint.check_tunnel()
            finally:
                if not queue_task.done():
                    queue_task.cancel()
                    await asyncio.gather(queue_task, return_exceptions=True)
            if execution.done() and queue.empty() and not pending:
                break
        await raise_finished_forward_errors(pending)
        return await execution
    finally:
        if not execution.done():
            execution.cancel()
            await asyncio.gather(execution, return_exceptions=True)
        for task_ in pending:
            if not task_.done():
                task_.cancel()
        if pending:
            await asyncio.gather(*pending, return_exceptions=True)
        endpoint.unregister_rollout(request_key)
        await client.close()


async def raise_finished_forward_errors(pending: set[asyncio.Task[None]]) -> None:
    finished = {task for task in pending if task.done()}
    for task in finished:
        pending.remove(task)
        await task


async def raise_execution_error(execution: asyncio.Task[object]) -> None:
    if execution.cancelled():
        await execution
    error = execution.exception()
    if error is not None:
        raise error


async def forward_request(
    endpoint: Endpoint,
    runtime: Runtime,
    task: Task,
    state: State,
    request_id: str,
) -> None:
    request = endpoint.get_request(request_id)
    prompt = normalize_endpoint_prompt(request)
    tool_defs = normalize_endpoint_tools(
        request.get("tools"), str(request.get("protocol"))
    )
    response = None
    error: BaseException | None = None
    try:
        response = await runtime.submit_model_request(
            prompt,
            task,
            state,
            tool_defs=tool_defs,
            extras={
                "endpoint": True,
                "endpoint_request_id": request_id,
                "headers": request.get("headers") or {},
            },
        )
    except BaseException as e:
        error = e
        if isinstance(e, Error):
            state["error"] = error_info(e)
        raise
    finally:
        if bool(request.get("stream")):
            if request.get("protocol") != "openai_chat_completions":
                raise NotImplementedError(
                    "Streaming interception is currently supported for OpenAI Chat Completions."
                )
            await synthesize_stream(request, response, error)
        else:
            deliver_response(request, response, error)


def normalize_endpoint_prompt(request: dict[str, object]) -> Messages:
    protocol = request.get("protocol")
    if protocol == "anthropic_messages":
        return normalize_anthropic_messages(request)
    if protocol == "openai_responses":
        return normalize_openai_responses_input(request.get("input"))
    return normalize_endpoint_messages(request.get("messages"))


def normalize_endpoint_messages(messages: object) -> Messages:
    if isinstance(messages, str):
        return normalize_messages(messages, field_name="endpoint.messages")
    if isinstance(messages, list):
        return normalize_messages(
            cast(Messages, messages), field_name="endpoint.messages"
        )
    raise TypeError("Endpoint messages must be vf.Messages or str.")


def normalize_anthropic_messages(request: dict[str, object]) -> Messages:
    messages: Messages = []
    system = request.get("system")
    if isinstance(system, str) and system:
        messages.append(SystemMessage(content=system))
    raw_messages = request.get("messages")
    if not isinstance(raw_messages, list):
        raise TypeError("Anthropic endpoint messages must be a list.")
    for raw_message in raw_messages:
        if not isinstance(raw_message, Mapping):
            raise TypeError("Anthropic endpoint message entries must be dicts.")
        raw_message = cast(Mapping[str, object], raw_message)
        role = raw_message.get("role")
        content = raw_message.get("content")
        if role == "user":
            messages.extend(normalize_anthropic_user_message(content))
        elif role == "assistant":
            messages.append(normalize_anthropic_assistant_message(content))
        else:
            raise ValueError(f"Unsupported Anthropic message role: {role!r}")
    return messages


def normalize_anthropic_user_message(content: object) -> Messages:
    if isinstance(content, str):
        return [UserMessage(content=content)]
    if not isinstance(content, list):
        return [UserMessage(content=str(content))]
    messages: Messages = []
    text_parts: list[str] = []
    for block in content:
        if not isinstance(block, Mapping):
            continue
        block = cast(Mapping[str, object], block)
        block_type = block.get("type")
        if block_type == "text" and isinstance(block.get("text"), str):
            text_parts.append(str(block["text"]))
        elif block_type == "tool_result":
            tool_use_id = block.get("tool_use_id")
            if not isinstance(tool_use_id, str):
                continue
            messages.append(
                ToolMessage(
                    tool_call_id=tool_use_id,
                    content=anthropic_block_content_text(block.get("content")),
                )
            )
    if text_parts:
        messages.insert(0, UserMessage(content="\n".join(text_parts)))
    return messages


def normalize_anthropic_assistant_message(content: object) -> AssistantMessage:
    if isinstance(content, str):
        return AssistantMessage(content=content)
    if not isinstance(content, list):
        return AssistantMessage(content=str(content))
    text_parts: list[str] = []
    tool_calls: list[ToolCall] = []
    for block in content:
        if not isinstance(block, Mapping):
            continue
        block = cast(Mapping[str, object], block)
        block_type = block.get("type")
        if block_type == "text" and isinstance(block.get("text"), str):
            text_parts.append(str(block["text"]))
        elif block_type == "tool_use":
            tool_id = block.get("id")
            name = block.get("name")
            if isinstance(tool_id, str) and isinstance(name, str):
                tool_calls.append(
                    ToolCall(
                        id=tool_id,
                        name=name,
                        arguments=json.dumps(block.get("input") or {}),
                    )
                )
    return AssistantMessage(
        content="\n".join(text_parts) if text_parts else None,
        tool_calls=tool_calls or None,
    )


def anthropic_block_content_text(content: object) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        text_parts: list[str] = []
        for block in content:
            if not isinstance(block, Mapping):
                continue
            block = cast(Mapping[str, object], block)
            text = block.get("text")
            if isinstance(text, str):
                text_parts.append(text)
        return "\n".join(text_parts)
    return str(content)


def normalize_openai_responses_input(raw_input: object) -> Messages:
    if isinstance(raw_input, str):
        return [UserMessage(content=raw_input)]
    if not isinstance(raw_input, list):
        raise TypeError("OpenAI Responses input must be a string or list.")
    messages: Messages = []
    for item in raw_input:
        if not isinstance(item, Mapping):
            raise TypeError("OpenAI Responses input entries must be dicts.")
        item = cast(Mapping[str, object], item)
        item_type = item.get("type")
        if item_type == "function_call":
            call_id = item.get("call_id") or item.get("id")
            name = item.get("name")
            arguments = item.get("arguments")
            if (
                isinstance(call_id, str)
                and isinstance(name, str)
                and isinstance(arguments, str)
            ):
                messages.append(
                    AssistantMessage(
                        tool_calls=[
                            ToolCall(id=call_id, name=name, arguments=arguments)
                        ]
                    )
                )
            continue
        if item_type == "function_call_output":
            call_id = item.get("call_id")
            if isinstance(call_id, str):
                messages.append(
                    ToolMessage(
                        tool_call_id=call_id,
                        content=responses_content_text(item.get("output")),
                    )
                )
            continue
        role = item.get("role")
        content = responses_content_text(item.get("content"))
        if role == "system":
            messages.append(SystemMessage(content=content))
        elif role == "assistant":
            messages.append(AssistantMessage(content=content))
        else:
            messages.append(UserMessage(content=content))
    return messages


def responses_content_text(content: object) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        text_parts: list[str] = []
        for part in content:
            if isinstance(part, Mapping):
                part = cast(Mapping[str, object], part)
                text = part.get("text")
                if isinstance(text, str):
                    text_parts.append(text)
        return "\n".join(text_parts)
    return "" if content is None else str(content)


def normalize_endpoint_tools(tools: object, protocol: str) -> list[Tool] | None:
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
        if protocol == "anthropic_messages":
            normalized.append(
                Tool(
                    name=str(raw_tool.get("name", "")),
                    description=str(raw_tool.get("description", "")),
                    parameters=cast(dict[str, Any], raw_tool.get("input_schema") or {}),
                )
            )
            continue
        if protocol == "openai_responses":
            normalized.append(
                Tool(
                    name=str(raw_tool.get("name", "")),
                    description=str(raw_tool.get("description", "")),
                    parameters=cast(dict[str, Any], raw_tool.get("parameters") or {}),
                    strict=cast(bool | None, raw_tool.get("strict")),
                )
            )
            continue
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


def assistant_completion_from_messages(
    prompt: list[dict[str, object]], messages: list[dict[str, object]]
) -> list[dict[str, object]]:
    return messages[len(prompt) :]
