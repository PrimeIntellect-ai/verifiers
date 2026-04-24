from __future__ import annotations

import asyncio
import time
from typing import TYPE_CHECKING

from verifiers.decorators import cleanup, stop
from verifiers.envs.experimental.channels import ChannelMap, Endpoint
from verifiers.errors import Error
from verifiers.types import ClientType, Messages, Response, State, Tool
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
        endpoint_port: int | None = None,
        endpoint_url: str | None = None,
        endpoint_secret: str | None = None,
        api_client_type: ClientType = "openai_chat_completions",
        max_turns: int = -1,
        poll_interval: float = 1.0,
        **kwargs: object,
    ):
        kwargs.setdefault("parallel_model_requests", True)
        super().__init__(**kwargs)
        self.endpoint_port = endpoint_port
        self.endpoint_url = endpoint_url
        self.endpoint_secret = endpoint_secret
        self.api_client_type = api_client_type
        self.max_turns = max_turns
        self.poll_interval = poll_interval

    def channels(self, task: Task | None = None) -> ChannelMap:
        channels = dict(super().channels(task))
        channels["endpoint"] = {
            "port": self.endpoint_port,
            "url": self.endpoint_url,
            "secret": self.endpoint_secret,
            "api_client_type": self.api_client_type,
            "use_tunnel": self.use_tunnel_for_endpoint,
        }
        return channels

    def require_endpoint(self, resources: Resources) -> Endpoint:
        endpoint = resources.endpoint
        if isinstance(endpoint, Endpoint):
            return endpoint
        raise RuntimeError("EndpointHarness requires the endpoint channel.")

    async def setup_state(self, task: Task, resources: Resources) -> State:
        state = await super().setup_state(task, resources)
        state["endpoint_driver_completed"] = False
        await self.start_endpoint(state, resources)
        resources.runtime["endpoint_driver_task"] = asyncio.create_task(
            self.run_endpoint_driver(task, state, resources)
        )
        return state

    async def start_endpoint(self, state: State, resources: Resources) -> str:
        endpoint = self.require_endpoint(resources)
        return await endpoint.register_rollout(state)

    async def run_endpoint_driver(
        self, task: Task, state: State, resources: Resources
    ) -> object:
        request = ModelRequest(prompt=state["prompt"])
        model_task = self.schedule_model_request(task, request, state, resources)
        await self.wait_for_model_request(model_task, state, resources)
        return None

    async def normalize_endpoint_messages(self, messages: object) -> Messages:
        return normalize_messages(messages, field_name="endpoint.messages")  # type: ignore[arg-type]

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

    def endpoint_tool_cache_key(self, tools: object) -> tuple[str, ...] | None:
        if not isinstance(tools, list):
            return None
        names: list[str] = []
        for tool in tools:
            if isinstance(tool, Tool):
                names.append(tool.name)
            elif isinstance(tool, dict):
                function = tool.get("function")
                if isinstance(function, dict):
                    names.append(str(function.get("name", "")))
                else:
                    names.append(str(tool.get("name", "")))
            else:
                names.append("")
        return tuple(sorted(names))

    async def get_model_request(
        self, task: Task, state: State, resources: Resources
    ) -> ModelRequest | None:
        driver_task = self.endpoint_driver_task(resources)
        request_id = await self.next_endpoint_request(
            driver_task, task, state, resources
        )
        if request_id is None:
            return None
        return await self.prepare_endpoint_request(request_id, state, resources)

    def endpoint_driver_task(self, resources: Resources) -> asyncio.Task:
        driver_task = resources.runtime.get("endpoint_driver_task")
        if not isinstance(driver_task, asyncio.Task):
            raise RuntimeError("EndpointHarness driver task was not initialized.")
        return driver_task

    async def prepare_endpoint_request(
        self, request_id: str, state: State, resources: Resources
    ) -> ModelRequest:
        endpoint = self.require_endpoint(resources)
        request = endpoint.get_request(request_id)
        prompt = await self.normalize_endpoint_messages(request["messages"])
        if not state["trajectory"]:
            state["prompt"] = prompt
        raw_tools = request.get("tools")
        if raw_tools is None:
            tools: list[Tool] | None = []
        else:
            cache_key = self.endpoint_tool_cache_key(raw_tools)
            cached_key, cached_tools = resources.runtime.get(
                "endpoint_cached_tool_defs", (None, None)
            )
            if cache_key is not None and cache_key == cached_key:
                tools = cached_tools
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

    async def get_model_response(
        self,
        prompt: Messages,
        task: Task,
        state: State,
        resources: Resources,
        tool_defs: object = None,
        context: dict[str, object] | None = None,
    ) -> Response:
        request_id = context.get("endpoint_request_id") if context else None
        if not isinstance(request_id, str):
            return await super().get_model_response(
                prompt,
                task,
                state,
                resources,
                tool_defs=tool_defs,
                context=context,
            )
        from verifiers.utils.interception_utils import (
            deliver_response,
            synthesize_stream,
        )

        endpoint = self.require_endpoint(resources)
        request = endpoint.get_request(request_id)
        response: Response | None = None
        error: BaseException | None = None
        try:
            response = await super().get_model_response(
                prompt,
                task,
                state,
                resources,
                tool_defs=tool_defs,
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
        driver_task: asyncio.Task,
        task: Task,
        state: State,
        resources: Resources,
    ) -> str | None:
        endpoint = self.require_endpoint(resources)
        queue = endpoint.request_queue(state["endpoint_request_key"])
        while True:
            if driver_task.done():
                await self.finish_endpoint_driver_task(driver_task, state)
                if queue.empty():
                    return None
                return queue.get_nowait()
            queue_task = asyncio.create_task(queue.get())
            try:
                done, _ = await asyncio.wait(
                    {queue_task, driver_task},
                    timeout=self.poll_interval,
                    return_when=asyncio.FIRST_COMPLETED,
                )
                if queue_task in done:
                    return queue_task.result()
                if driver_task in done:
                    await self.finish_endpoint_driver_task(driver_task, state)
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

    async def finish_endpoint_driver_task(
        self, driver_task: asyncio.Task, state: State
    ) -> None:
        if state.get("endpoint_driver_completed") or state.get("error") is not None:
            return
        try:
            await driver_task
            state["endpoint_driver_completed"] = True
        except Error as e:
            state["error"] = e
        except Exception as e:
            state["error"] = e

    async def on_endpoint_poll_timeout(
        self, state: State, resources: Resources
    ) -> None:
        await self.require_endpoint(resources).check_tunnel()

    async def finalize_state(
        self, task: Task, state: State, resources: Resources
    ) -> State:
        state = await super().finalize_state(task, state, resources)
        start = state["timing"]["start_time"]
        state["timing"]["generation_ms"] = (time.time() - start) * 1000
        state["timing"]["total_ms"] = state["timing"]["generation_ms"]
        state["is_completed"] = True
        state["stop_condition"] = (
            state.get("stop_condition") or "endpoint_driver_completed"
        )
        return state

    @cleanup
    async def cleanup_endpoint(self, state: State, resources: Resources) -> None:
        driver_task = resources.runtime.get("endpoint_driver_task")
        if isinstance(driver_task, asyncio.Task):
            await self.cancel_endpoint_driver_task(driver_task)
        key = state.get("endpoint_request_key")
        if key:
            self.require_endpoint(resources).unregister_rollout(key)

    @stop
    async def max_turns_reached(self, state: State, resources: Resources) -> bool:
        requests = state.get("num_model_requests", len(state.get("trajectory", [])))
        return requests >= self.max_turns and self.max_turns > 0

    @stop
    async def endpoint_driver_completed(
        self, state: State, resources: Resources
    ) -> bool:
        return bool(state.get("endpoint_driver_completed"))

    async def cancel_endpoint_driver_task(self, driver_task: asyncio.Task) -> None:
        if driver_task.done():
            return
        driver_task.cancel()
        try:
            await driver_task
        except asyncio.CancelledError:
            pass
