import asyncio
import json
import logging
import os
import time
import uuid
from collections.abc import Awaitable, Callable
from typing import Literal, Protocol, TypeAlias, cast

from anthropic import Anthropic, AsyncAnthropic
from openai import AsyncOpenAI, OpenAI

from verifiers.errors import Error, OverlongPromptError, TunnelError
from verifiers.types import (
    AssistantMessage,
    ClientType,
    ContentPart,
    EndpointApi,
    EndpointClient,
    EndpointConfig,
    MessageContent,
    Messages,
    Response,
    SystemMessage,
    Tool,
    ToolCall,
    ToolMessage,
    UserMessage,
)
from verifiers.utils.interception_utils import (
    InterceptionServer,
    deliver_response,
    synthesize_stream,
)
from verifiers.utils.message_utils import normalize_messages
from verifiers.utils.response_utils import parse_response_message

from ..runtime import ModelRequestContext, Runtime, TrajectoryVisibility
from ..state import State
from ..task import Task
from ..types import JsonData, PromptMessage, RuntimeObject, ToolParameters
from .serialization_utils import serializable

VF_TRAJECTORY_VISIBILITY_HEADER = "x-verifiers-trajectory"
VF_ENDPOINT_API_KEY_VAR = "VF_ENDPOINT_API_KEY"
NormalizedEndpointApi: TypeAlias = Literal[
    "chat_completions",
    "completions",
    "responses",
    "messages",
]
EndpointInterceptData: TypeAlias = dict[str, RuntimeObject]


class TunnelHandle(Protocol):
    is_running: bool
    url: str | None

    async def start(self) -> str: ...

    async def check_registered(self) -> bool: ...

    def sync_stop(self) -> None: ...


def client_from_state(
    state: State,
    api: EndpointApi | ClientType = "chat_completions",
    *,
    sync: bool = False,
) -> EndpointClient:
    endpoint = endpoint_from_state(state)
    return endpoint.client(state, api=api, sync=sync)


def endpoint_config_from_state(
    state: State,
    api: EndpointApi | ClientType = "chat_completions",
) -> EndpointConfig:
    endpoint = endpoint_from_state(state)
    return endpoint.config(state, api=api)


def endpoint_from_state(state: State) -> "Endpoint":
    runtime = state._runtime()
    harness = runtime.harness
    if harness is None:
        raise RuntimeError("State does not have an active model endpoint.")
    endpoint = harness.endpoint
    if not isinstance(endpoint, Endpoint):
        raise RuntimeError("State does not have an active model endpoint.")
    return endpoint


def endpoint_api_client_type(
    api: NormalizedEndpointApi,
) -> Literal[
    "openai_chat_completions",
    "openai_completions",
    "openai_responses",
    "anthropic_messages",
]:
    if api == "chat_completions":
        return "openai_chat_completions"
    if api == "completions":
        return "openai_completions"
    if api == "responses":
        return "openai_responses"
    return "anthropic_messages"


def normalize_endpoint_api(
    api: EndpointApi | ClientType,
) -> NormalizedEndpointApi:
    if api in {
        "chat_completions",
        "openai_chat_completions",
        "chat",
    }:
        return "chat_completions"
    if api in {"responses", "openai_responses"}:
        return "responses"
    if api in {"messages", "anthropic_messages"}:
        return "messages"
    if api in {"completions", "openai_completions"}:
        return "completions"
    if api == "openai_chat_completions_token":
        raise ValueError(
            "state.get_client(...) does not expose token-level chat completions clients."
        )
    if api == "renderer":
        raise ValueError("state.get_client(...) does not expose renderer clients.")
    if api == "nemorl_chat_completions":
        raise ValueError(
            "state.get_client(...) does not expose NeMoRL chat completions clients."
        )
    raise ValueError(f"Unknown endpoint API {api!r}.")


class Endpoint:
    TUNNEL_CHECK_INTERVAL = 60.0

    def __init__(
        self,
        port: int | None = None,
        secret: str | None = None,
        use_tunnel: bool = False,
        logger: logging.Logger | None = None,
        tunnel_labels: list[str] | None = None,
    ):
        self.use_tunnel = use_tunnel
        self.logger = logger or logging.getLogger(__name__)
        self.server = InterceptionServer(
            port if port is not None else 0,
            secret=secret or os.environ.get("ENDPOINT_SECRET"),
        )
        self.secret = self.server.secret
        self.tunnel_labels = list(tunnel_labels) if tunnel_labels else []
        self._tunnel: TunnelHandle | None = None
        self._tunnel_lock = asyncio.Lock()
        self._tunnel_last_checked = 0.0
        self._rollout_queues: dict[str, asyncio.Queue[str]] = {}

    async def start(self) -> None:
        await self.server.start()

    async def register_rollout(
        self,
        state: State,
        tool_handler: object | None = None,
        tool_defs: list[Tool] | None = None,
        user_handler: object | None = None,
        stop_handler: object | None = None,
        model_handler: object | None = None,
    ) -> str:
        await self.start()
        rollout_key = f"rollout_{uuid.uuid4().hex[:8]}"
        request_queue = self.server.register_rollout(
            rollout_key,
            state=state,
            tool_handler=tool_handler,
            tool_defs=tool_defs,
            user_handler=user_handler,
            stop_handler=stop_handler,
            model_handler=model_handler,
        )
        self._rollout_queues[rollout_key] = cast(asyncio.Queue[str], request_queue)
        endpoint_root_url = f"{await self.url_base()}/rollout/{rollout_key}"
        api_key_var = f"{VF_ENDPOINT_API_KEY_VAR}_{rollout_key.upper()}"
        state["endpoint_rollout_key"] = rollout_key
        state["endpoint_root_url"] = endpoint_root_url
        state["endpoint_base_url"] = f"{endpoint_root_url}/v1"
        state["endpoint_api_key_var"] = api_key_var
        return state["endpoint_base_url"]

    def client(
        self,
        state: State,
        api: EndpointApi | ClientType = "chat_completions",
        *,
        sync: bool = False,
    ) -> EndpointClient:
        api = normalize_endpoint_api(api)
        api_key = self.secret or "intercepted"
        if api == "messages":
            base_url = str(state["endpoint_root_url"])
            if sync:
                return Anthropic(api_key=api_key, base_url=base_url)
            return AsyncAnthropic(api_key=api_key, base_url=base_url)
        base_url = str(state["endpoint_base_url"])
        if sync:
            return OpenAI(api_key=api_key, base_url=base_url)
        return AsyncOpenAI(api_key=api_key, base_url=base_url)

    def config(
        self,
        state: State,
        api: EndpointApi | ClientType = "chat_completions",
    ) -> EndpointConfig:
        api = normalize_endpoint_api(api)
        base_url = (
            str(state["endpoint_root_url"])
            if api == "messages"
            else str(state["endpoint_base_url"])
        )
        return EndpointConfig(
            model=state.get_model(),
            base_url=base_url,
            api_key_var=str(state["endpoint_api_key_var"]),
            api_client_type=endpoint_api_client_type(api),
        )

    def unregister_rollout(self, rollout_key: str) -> None:
        self._rollout_queues.pop(rollout_key, None)
        self.server.unregister_rollout(rollout_key)

    def rollout_queue(self, rollout_key: str) -> asyncio.Queue[str]:
        return self._rollout_queues[rollout_key]

    def get_request(self, request_id: str) -> EndpointInterceptData:
        return cast(EndpointInterceptData, self.server.intercepts[request_id])

    def discard_request(self, request_id: str) -> None:
        """Drop a delivered intercept from the server's per-request store."""
        self.server.intercepts.pop(request_id, None)

    def request_context(
        self, request_id: str, request: EndpointInterceptData
    ) -> ModelRequestContext:
        headers = request.get("headers") or {}
        if not isinstance(headers, dict):
            raise TypeError("Endpoint request headers must be a mapping.")
        header_data: dict[str, str] = {}
        for key, value in headers.items():
            if not isinstance(key, str) or not isinstance(value, str):
                raise TypeError("Endpoint request headers must be strings.")
            header_data[key.lower()] = value
        return ModelRequestContext(
            source="endpoint",
            endpoint_request_id=request_id,
            headers=header_data,
            trajectory_visibility=self.trajectory_visibility(header_data),
        )

    def trajectory_visibility(self, headers: dict[str, str]) -> TrajectoryVisibility:
        value = headers.get(VF_TRAJECTORY_VISIBILITY_HEADER)
        if value is None:
            return "append"
        if not isinstance(value, str):
            raise TypeError(
                f"{VF_TRAJECTORY_VISIBILITY_HEADER} must be 'append' or 'hidden'."
            )
        visibility = value.strip().lower()
        if visibility not in {"append", "hidden"}:
            raise ValueError(
                f"{VF_TRAJECTORY_VISIBILITY_HEADER} must be 'append' or 'hidden'."
            )
        return cast(TrajectoryVisibility, visibility)

    async def url_base(self) -> str:
        if self.use_tunnel:
            return await self.get_tunnel_url()
        return f"http://127.0.0.1:{self.server.port}"

    async def get_tunnel_url(self) -> str:
        from prime_tunnel import Tunnel

        async with self._tunnel_lock:
            tunnel = self._tunnel
            if tunnel is not None and not tunnel.is_running:
                tunnel.sync_stop()
                self._tunnel = None

            tunnel = self._tunnel
            if tunnel is not None:
                now = time.time()
                if now - self._tunnel_last_checked > self.TUNNEL_CHECK_INTERVAL:
                    self._tunnel_last_checked = now
                    if not await tunnel.check_registered():
                        tunnel.sync_stop()
                        self._tunnel = None

            if self._tunnel is None:
                tunnel = cast(
                    TunnelHandle,
                    Tunnel(local_port=self.server.port, labels=self.tunnel_labels),
                )
                url = await tunnel.start()
                self._tunnel = tunnel
                self._tunnel_last_checked = time.time()
                return str(url)

            tunnel = self._tunnel
            if tunnel.url is None:
                raise TunnelError("Tunnel started but URL is unavailable.")
            return str(tunnel.url)

    async def check_tunnel(self) -> None:
        tunnel = self._tunnel
        if tunnel is not None and not tunnel.is_running:
            raise TunnelError("Tunnel process died during rollout.")

    async def teardown(self) -> None:
        async with self._tunnel_lock:
            tunnel = self._tunnel
            if tunnel is not None:
                tunnel.sync_stop()
                self._tunnel = None
        await self.server.stop()


async def run_intercepted_program(
    program: Callable[[Task, State], Awaitable[State | JsonData | None]],
    endpoint: Endpoint,
    runtime: Runtime,
    task: Task,
    state: State,
) -> State | JsonData | None:
    async def call_tool(name: str, arguments: ToolParameters) -> object:
        return await runtime.call_tool(name, task, state, **dict(arguments))

    async def call_user(transcript: list[PromptMessage]) -> list[JsonData]:
        return await runtime.user_messages(task, state, transcript=transcript)

    async def check_stop() -> JsonData:
        done = await runtime.is_completed(task, state)
        stop_condition = state.get("stop_condition")
        return {
            "done": done,
            "stop_condition": stop_condition
            if isinstance(stop_condition, str) or stop_condition is None
            else str(stop_condition),
        }

    model_tasks: set[asyncio.Task[Response]] = set()

    async def call_model(messages: list[PromptMessage], tools: object) -> JsonData:
        # Sandbox sends canonical Messages; host resolves the client, tokenizes,
        # and records the step. Tool defs come from the runtime.
        del tools
        prompt = normalize_messages(
            cast(Messages, messages), field_name="vf.model.messages"
        )
        request = asyncio.ensure_future(
            runtime.submit_model_request(
                prompt,
                task,
                state,
                tool_defs=runtime.tool_defs(state),
                context=ModelRequestContext(source="endpoint"),
            )
        )
        model_tasks.add(request)
        try:
            response = await request
        except Error as exc:
            if isinstance(exc, OverlongPromptError):
                state["prompt_too_long"] = True
                state._set_truncated(True)
                state._set_stop_condition("prompt_too_long", overwrite=True)
            else:
                state._set_error(exc)
                state._set_stop_condition("has_error", overwrite=True)
            raise
        finally:
            if request.done():
                model_tasks.discard(request)
        completion = await parse_response_message(response)
        return cast(JsonData, serializable(completion[0]))

    await endpoint.register_rollout(
        state,
        tool_handler=call_tool,
        tool_defs=runtime.tool_defs(state),
        user_handler=call_user,
        stop_handler=check_stop,
        model_handler=call_model,
    )

    async def execute_program() -> State | JsonData | None:
        return await program(task, state)

    execution = asyncio.create_task(execute_program())
    rollout_key = str(state["endpoint_rollout_key"])
    queue = endpoint.rollout_queue(rollout_key)
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
        await cancel_forwarders(pending)
        await cancel_forwarders(cast("set[asyncio.Task[None]]", model_tasks))
        endpoint.unregister_rollout(rollout_key)


async def raise_finished_forward_errors(pending: set[asyncio.Task[None]]) -> None:
    finished = {task for task in pending if task.done()}
    for task in finished:
        pending.remove(task)
        await task


async def cancel_forwarders(pending: set[asyncio.Task[None]]) -> None:
    for task in pending:
        if not task.done():
            task.cancel()
    if pending:
        await asyncio.gather(*pending, return_exceptions=True)


async def raise_execution_error(
    execution: asyncio.Task[State | JsonData | None],
) -> None:
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
            context=endpoint.request_context(request_id, request),
        )
    except BaseException as e:
        error = e
        if isinstance(e, Error):
            state._set_error(e)
        raise
    finally:
        try:
            if bool(request.get("stream")):
                if request.get("protocol") != "openai_chat_completions":
                    raise NotImplementedError(
                        "Streaming interception is currently supported for OpenAI Chat Completions."
                    )
                await synthesize_stream(request, response, error)
            else:
                deliver_response(request, response, error)
        finally:
            endpoint.discard_request(request_id)


def normalize_endpoint_prompt(request: EndpointInterceptData) -> Messages:
    protocol = request.get("protocol")
    if protocol == "anthropic_messages":
        return normalize_anthropic_messages(request)
    if protocol == "openai_responses":
        return normalize_openai_responses_input(request.get("input"))
    if protocol == "openai_completions":
        return normalize_endpoint_messages(request.get("prompt"))
    return normalize_endpoint_messages(request.get("messages"))


def normalize_endpoint_messages(messages: object) -> Messages:
    if isinstance(messages, str):
        return normalize_messages(messages, field_name="endpoint.messages")
    if isinstance(messages, list):
        return normalize_messages(
            cast(Messages, messages), field_name="endpoint.messages"
        )
    raise TypeError("Endpoint messages must be vf.Messages or str.")


def normalize_anthropic_messages(request: EndpointInterceptData) -> Messages:
    messages: Messages = []
    system = request.get("system")
    if isinstance(system, str) and system:
        messages.append(SystemMessage(content=system))
    raw_messages = request.get("messages")
    if not isinstance(raw_messages, list):
        raise TypeError("Anthropic endpoint messages must be a list.")
    for raw_message in raw_messages:
        if not isinstance(raw_message, dict):
            raise TypeError("Anthropic endpoint message entries must be dicts.")
        raw_message = cast(JsonData, raw_message)
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
        if not isinstance(block, dict):
            continue
        block = cast(JsonData, block)
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
                    content=anthropic_tool_result_content(block.get("content")),
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
        if not isinstance(block, dict):
            continue
        block = cast(JsonData, block)
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
            if not isinstance(block, dict):
                continue
            block = cast(JsonData, block)
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
        if not isinstance(item, dict):
            raise TypeError("OpenAI Responses input entries must be dicts.")
        item = cast(JsonData, item)
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
                        content=responses_tool_output_content(item.get("output")),
                    )
                )
            continue
        role = item.get("role")
        content = responses_content_text(item.get("content"))
        if role in {"system", "developer"}:
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
            if isinstance(part, dict):
                part = cast(JsonData, part)
                text = part.get("text")
                if isinstance(text, str):
                    text_parts.append(text)
        return "\n".join(text_parts)
    return "" if content is None else str(content)


def responses_tool_output_content(output: object) -> MessageContent:
    """Responses function_call_output -> internal tool content, keeping images
    (input_image -> image_url); text-only falls back to a string."""

    if not isinstance(output, list):
        return responses_content_text(output)
    parts: list[ContentPart] = []
    has_image = False
    for item in output:
        if not isinstance(item, dict):
            continue
        item = cast(JsonData, item)
        if item.get("type") == "input_image":
            url = item.get("image_url")
            if isinstance(url, str) and url:
                parts.append({"type": "image_url", "image_url": {"url": url}})
                has_image = True
        else:
            text = item.get("text")
            if isinstance(text, str):
                parts.append({"type": "text", "text": text})
    return parts if has_image else responses_content_text(output)


def anthropic_tool_result_content(content: object) -> MessageContent:
    """Anthropic tool_result content -> internal tool content, keeping images
    (image block -> image_url); text-only falls back to a string."""

    if not isinstance(content, list):
        return anthropic_block_content_text(content)
    parts: list[ContentPart] = []
    has_image = False
    for block in content:
        if not isinstance(block, dict):
            continue
        block = cast(JsonData, block)
        if block.get("type") == "image":
            source = block.get("source")
            url = ""
            if isinstance(source, dict):
                if source.get("type") == "base64":
                    media_type = str(source.get("media_type") or "image/png")
                    data = str(source.get("data") or "")
                    url = f"data:{media_type};base64,{data}"
                elif source.get("type") == "url":
                    url = str(source.get("url") or "")
            if url:
                parts.append({"type": "image_url", "image_url": {"url": url}})
                has_image = True
        else:
            text = block.get("text")
            if isinstance(text, str):
                parts.append({"type": "text", "text": text})
    return parts if has_image else anthropic_block_content_text(content)


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
        raw_tool_data = cast(ToolParameters, raw_tool)
        if protocol == "anthropic_messages":
            normalized.append(
                Tool(
                    name=str(raw_tool_data.get("name", "")),
                    description=str(raw_tool_data.get("description", "")),
                    parameters=endpoint_tool_parameters(
                        raw_tool_data.get("input_schema")
                    ),
                )
            )
            continue
        if protocol == "openai_responses":
            normalized.append(
                Tool(
                    name=str(raw_tool_data.get("name", "")),
                    description=str(raw_tool_data.get("description", "")),
                    parameters=endpoint_tool_parameters(
                        raw_tool_data.get("parameters")
                    ),
                    strict=cast(bool | None, raw_tool_data.get("strict")),
                )
            )
            continue
        function_payload = raw_tool_data.get("function")
        if raw_tool_data.get("type") == "function" and isinstance(
            function_payload, dict
        ):
            function_payload = cast(ToolParameters, function_payload)
            normalized.append(
                Tool(
                    name=str(function_payload.get("name", "")),
                    description=str(function_payload.get("description", "")),
                    parameters=endpoint_tool_parameters(
                        function_payload.get("parameters")
                    ),
                    strict=cast(bool | None, function_payload.get("strict")),
                )
            )
        else:
            normalized.append(Tool.model_validate(raw_tool_data))
    return normalized


def endpoint_tool_parameters(value: object) -> ToolParameters:
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise TypeError("Endpoint tool parameters must be a mapping.")
    return {str(key): item for key, item in value.items()}


def assistant_completion_from_messages(
    prompt: list[JsonData], messages: list[JsonData]
) -> list[JsonData]:
    return messages[len(prompt) :]
