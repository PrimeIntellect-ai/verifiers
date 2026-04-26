from __future__ import annotations

import asyncio
import json
from collections.abc import Iterable
from typing import TYPE_CHECKING, Any, cast

from openai import AsyncOpenAI

from verifiers.decorators import cleanup, stop
from verifiers.envs.experimental.channels import ChannelMap, Endpoint
from verifiers.envs.experimental.configs import EndpointConfig, RunConfig
from verifiers.errors import Error
from verifiers.rubrics.rubric import Rubric
from verifiers.types import (
    AssistantMessage,
    Messages,
    Response,
    State,
    Tool,
)
from verifiers.utils.error_utils import error_info
from verifiers.utils.message_utils import normalize_messages

from verifiers.envs.experimental.harness import Harness, ModelRequest
from verifiers.envs.experimental.task import Task

if TYPE_CHECKING:
    from verifiers.envs.experimental.resources import Resources


class EndpointHarness(Harness):
    """Harness pattern for rollout code that calls a managed LLM endpoint."""

    use_tunnel_for_endpoint = False

    def __init__(
        self,
        endpoint: EndpointConfig | dict[str, object] | None = None,
        run: RunConfig | dict[str, object] | None = None,
        channels: ChannelMap | None = None,
        rubric: Rubric | None = None,
        system_prompt: str | None = None,
        tools: Iterable[object] | None = None,
    ):
        super().__init__(
            channels=channels,
            run=run or RunConfig(max_turns=-1, parallel_model_requests=True),
            rubric=rubric,
            system_prompt=system_prompt,
            tools=tools,
        )
        self.endpoint = EndpointConfig.model_validate(
            endpoint or {"use_tunnel": self.use_tunnel_for_endpoint}
        )
        self.poll_interval = self.endpoint.poll_interval

    def channels(self, task: Task | None = None) -> ChannelMap:
        channels = dict(super().channels(task))
        channels["endpoint"] = {
            "port": self.endpoint.port,
            "url": self.endpoint.url,
            "secret": self.endpoint.secret,
            "api_client_type": self.endpoint.api_client_type,
            "use_tunnel": self.endpoint.use_tunnel,
        }
        return channels

    async def setup_state(self, task: Task, resources: Resources) -> State:
        state = await super().setup_state(task, resources)
        state["execute_completed"] = False
        await self.start_endpoint(state, resources)
        client = self.create_endpoint_client(state, resources)
        resources.runtime["execution"] = asyncio.create_task(
            self.execute(task, state, resources, client)
        )
        return state

    async def start_endpoint(self, state: State, resources: Resources) -> str:
        endpoint = cast(Endpoint, resources.require("endpoint"))
        return await endpoint.register_rollout(state)

    def create_endpoint_client(self, state: State, resources: Resources) -> AsyncOpenAI:
        endpoint = cast(Endpoint, resources.require("endpoint"))
        return AsyncOpenAI(
            api_key=endpoint.secret or "intercepted",
            base_url=state["endpoint_base_url"],
        )

    async def execute(
        self,
        task: Task,
        state: State,
        resources: Resources,
        client: AsyncOpenAI,
    ) -> object:
        """Run endpoint-facing user code.

        Subclasses can override this method to run DSPy, LangChain, direct
        OpenAI-compatible calls, or other Python logic against the supplied
        endpoint client.
        """
        return await client.chat.completions.create(
            model=resources.model,
            messages=self.endpoint_message_payload(state["prompt"]),
            tools=self.endpoint_tool_payload(state["tool_defs"]),
            **resources.sampling_args,
        )

    def endpoint_message_payload(self, messages: Messages) -> list[dict[str, object]]:
        payload: list[dict[str, object]] = []
        for message in messages:
            raw = message.model_dump(exclude_none=True)
            if isinstance(message, AssistantMessage) and message.tool_calls:
                raw["tool_calls"] = [
                    {
                        "id": tool_call.id,
                        "type": "function",
                        "function": {
                            "name": tool_call.name,
                            "arguments": tool_call.arguments,
                        },
                    }
                    for tool_call in message.tool_calls or []
                ]
            payload.append(raw)
        return payload

    def endpoint_tool_payload(
        self, tools: list[Tool]
    ) -> list[dict[str, object]] | None:
        if not tools:
            return None
        payload: list[dict[str, object]] = []
        for tool in tools:
            function: dict[str, object] = {
                "name": tool.name,
                "description": tool.description,
                "parameters": tool.parameters,
            }
            if tool.strict is not None:
                function["strict"] = tool.strict
            payload.append({"type": "function", "function": function})
        return payload

    async def normalize_endpoint_messages(self, messages: object) -> Messages:
        if isinstance(messages, str):
            return normalize_messages(messages, field_name="endpoint.messages")
        if isinstance(messages, list):
            return normalize_messages(
                cast(Messages, messages), field_name="endpoint.messages"
            )
        raise TypeError("Endpoint messages must be vf.Messages or str.")

    def normalize_endpoint_tools(self, tools: object) -> list[Tool] | None:
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
            if raw_tool.get("type") == "function" and isinstance(
                function_payload, dict
            ):
                normalized.append(
                    Tool(
                        name=function_payload.get("name", ""),
                        description=function_payload.get("description", ""),
                        parameters=function_payload.get("parameters") or {},
                        strict=function_payload.get("strict"),
                    )
                )
            else:
                normalized.append(Tool.model_validate(raw_tool))
        return normalized

    def endpoint_tool_cache_key(self, tools: object) -> str | None:
        if not isinstance(tools, list):
            return None
        payload = [
            tool.model_dump(exclude_none=True) if isinstance(tool, Tool) else tool
            for tool in tools
        ]
        return json.dumps(payload, sort_keys=True, default=str)

    async def get_model_request(
        self, task: Task, state: State, resources: Resources
    ) -> ModelRequest | None:
        execution = self.execution(resources)
        request_id = await self.next_endpoint_request(execution, task, state, resources)
        if request_id is None:
            return None
        return await self.prepare_endpoint_request(request_id, state, resources)

    def execution(self, resources: Resources) -> asyncio.Task:
        execution = resources.runtime.get("execution")
        if not isinstance(execution, asyncio.Task):
            raise RuntimeError("EndpointHarness execution was not initialized.")
        return execution

    async def prepare_endpoint_request(
        self, request_id: str, state: State, resources: Resources
    ) -> ModelRequest:
        endpoint = cast(Endpoint, resources.require("endpoint"))
        request = endpoint.get_request(request_id)
        prompt = await self.normalize_endpoint_messages(request["messages"])
        if not state["trajectory"]:
            state["prompt"] = prompt
        raw_tools = request.get("tools")
        if raw_tools is None:
            tools: list[Tool] | None = []
        else:
            cache_key = self.endpoint_tool_cache_key(raw_tools)
            cached = resources.runtime.get("endpoint_cached_tool_defs")
            if isinstance(cached, tuple) and len(cached) == 2:
                cached_key, cached_tools = cached
            else:
                cached_key, cached_tools = (None, None)
            if cache_key is not None and cache_key == cached_key:
                tools = cast(list[Tool] | None, cached_tools)
            else:
                tools = self.normalize_endpoint_tools(raw_tools)
                if cache_key is not None:
                    resources.runtime["endpoint_cached_tool_defs"] = (cache_key, tools)
        return ModelRequest(
            prompt=prompt,
            tool_defs=tools,
            extras={"endpoint": True},
            context={"endpoint_request_id": request_id},
        )

    async def submit_model_request(
        self,
        prompt: Messages,
        task: Task,
        state: State,
        resources: Resources,
        tool_defs: list[Tool] | None = None,
        extras: dict[str, object] | None = None,
        context: dict[str, object] | None = None,
    ) -> Response:
        request_id = context.get("endpoint_request_id") if context else None
        if not isinstance(request_id, str):
            return await super().submit_model_request(
                prompt,
                task,
                state,
                resources,
                tool_defs=tool_defs,
                extras=extras,
                context=context,
            )
        from verifiers.utils.interception_utils import (
            deliver_response,
            synthesize_stream,
        )

        endpoint = cast(Endpoint, resources.require("endpoint"))
        request = endpoint.get_request(request_id)
        response: Response | None = None
        error: BaseException | None = None
        try:
            response = await super().submit_model_request(
                prompt,
                task,
                state,
                resources,
                tool_defs=tool_defs,
                extras=extras,
                context=context,
            )
            return response
        except BaseException as e:
            error = e
            raise
        finally:
            if request.get("stream"):
                await synthesize_stream(request, response, error)
            else:
                deliver_response(request, response, error)

    async def next_endpoint_request(
        self,
        execution: asyncio.Task,
        task: Task,
        state: State,
        resources: Resources,
    ) -> str | None:
        endpoint = cast(Endpoint, resources.require("endpoint"))
        queue = endpoint.request_queue(state["endpoint_request_key"])
        while True:
            if execution.done():
                await self.finish_execution(execution, state)
                if queue.empty():
                    return None
                return queue.get_nowait()
            queue_task = asyncio.create_task(queue.get())
            try:
                done, _ = await asyncio.wait(
                    {queue_task, execution},
                    timeout=self.poll_interval,
                    return_when=asyncio.FIRST_COMPLETED,
                )
                if queue_task in done:
                    return queue_task.result()
                if execution in done:
                    await self.finish_execution(execution, state)
                    if queue.empty():
                        return None
                    return queue.get_nowait()
                await self.on_endpoint_poll_timeout(state, resources)
                if await self.is_completed(task, state, resources):
                    return None
            finally:
                if not queue_task.done():
                    queue_task.cancel()
                    await asyncio.gather(queue_task, return_exceptions=True)

    async def finish_execution(self, execution: asyncio.Task, state: State) -> None:
        if state.get("execute_completed") or state.get("error") is not None:
            return
        try:
            await execution
            state["execute_completed"] = True
        except Error as e:
            state["error"] = error_info(e)
        except Exception as e:
            state["error"] = error_info(e)

    async def on_endpoint_poll_timeout(
        self, state: State, resources: Resources
    ) -> None:
        await cast(Endpoint, resources.require("endpoint")).check_tunnel()

    @cleanup
    async def cleanup_endpoint(
        self, task: Task, state: State, resources: Resources
    ) -> None:
        execution = resources.runtime.get("execution")
        if isinstance(execution, asyncio.Task):
            await self.cancel_execution(execution)
        key = state.get("endpoint_request_key")
        if key:
            cast(Endpoint, resources.require("endpoint")).unregister_rollout(key)

    @stop
    async def max_turns_reached(
        self, task: Task, state: State, resources: Resources
    ) -> bool:
        requests = state.get("num_model_requests", len(state.get("trajectory", [])))
        return requests >= self.max_turns and self.max_turns > 0

    @stop
    async def execute_completed(
        self, task: Task, state: State, resources: Resources
    ) -> bool:
        return bool(state.get("execute_completed"))

    async def cancel_execution(self, execution: asyncio.Task) -> None:
        if execution.done():
            return
        execution.cancel()
        try:
            await execution
        except asyncio.CancelledError:
            pass
