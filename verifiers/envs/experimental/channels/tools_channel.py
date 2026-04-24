from __future__ import annotations

import asyncio
import inspect
import json
import logging
import threading
from collections.abc import Callable, Iterable, Iterator, Mapping
from dataclasses import dataclass, field
from typing import Any

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from mcp.types import TextContent, Tool as MCPTool

from verifiers.rubrics.rubric import Rubric
from verifiers.types import AssistantMessage, Messages, Tool
from verifiers.utils.async_utils import maybe_await
from verifiers.utils.tool_utils import convert_func_to_tool_def

from verifiers.envs.experimental.channels.channel import (
    Channel,
    ChannelConfig,
    ChannelContext,
    as_list,
)


@dataclass(frozen=True)
class CallableTool:
    func: Callable[..., Any]
    name: str | None = None
    description: str | None = None
    injected_args: tuple[str, ...] = ()


@dataclass(frozen=True)
class MCPServerSpec:
    name: str
    command: str
    args: list[str] = field(default_factory=list)
    env: dict[str, str] | None = None
    description: str = ""


INJECTABLE_TOOL_ARGS = frozenset(
    {
        "resources",
        "state",
        "task",
        "client",
        "model",
        "sampling_args",
        "tools",
    }
)


class ToolArgumentError(ValueError):
    pass


class ToolMonitorRubric(Rubric):
    def __init__(self, tool_names: list[str] | None = None):
        super().__init__()
        self.tool_names = list(tool_names or [])
        self.add_metric(self.total_tool_calls)
        for tool_name in self.tool_names:
            self.add_metric(self.tool_call_count_func(tool_name))

    async def total_tool_calls(self, completion: Messages) -> float:
        total = 0
        for message in completion:
            if isinstance(message, AssistantMessage):
                total += len(message.tool_calls or [])
        return float(total)

    def tool_call_count_func(self, tool_name: str) -> Callable[..., Any]:
        async def tool_call_count(completion: Messages) -> float:
            count = 0
            for message in completion:
                if not isinstance(message, AssistantMessage):
                    continue
                for tool_call in message.tool_calls or []:
                    if tool_call.name == tool_name:
                        count += 1
            return float(count)

        tool_call_count.__name__ = f"{tool_name}_calls"
        return tool_call_count


class MCPServerConnection:
    def __init__(self, spec: MCPServerSpec, logger: logging.Logger):
        self.spec = spec
        self.logger = logger
        self.session: ClientSession | None = None
        self.tools: dict[str, MCPTool] = {}
        self._connection_task: asyncio.Task | None = None
        self._ready = asyncio.Event()
        self._error: Exception | None = None
        self.loop: asyncio.AbstractEventLoop | None = None

    async def connect(self) -> dict[str, MCPTool]:
        self.loop = asyncio.get_running_loop()
        self._connection_task = asyncio.create_task(self._run_connection())
        await self._ready.wait()
        if self._error is not None:
            raise self._error
        return self.tools

    async def _run_connection(self) -> None:
        try:
            server_params = StdioServerParameters(
                command=self.spec.command,
                args=self.spec.args,
                env=self.spec.env,
            )
            async with stdio_client(server_params) as (read, write):
                async with ClientSession(read, write) as session:
                    self.session = session
                    await session.initialize()
                    tools_response = await session.list_tools()
                    self.tools = {tool.name: tool for tool in tools_response.tools}
                    self._ready.set()
                    while True:
                        await asyncio.sleep(1)
        except asyncio.CancelledError:
            raise
        except Exception as e:
            self._error = e
            self._ready.set()
        finally:
            self.session = None
            self.tools = {}

    async def call_tool(self, tool_name: str, arguments: dict[str, Any]) -> str:
        if self.session is None:
            raise RuntimeError(f"MCP server {self.spec.name!r} is not connected.")
        if self.loop is None:
            raise RuntimeError(f"MCP server {self.spec.name!r} has no event loop.")
        future = asyncio.run_coroutine_threadsafe(
            self.session.call_tool(tool_name, arguments=arguments), self.loop
        )
        result = await asyncio.wrap_future(future)
        if not result.content:
            return "No result returned from tool"
        text_parts: list[str] = []
        for item in result.content:
            if isinstance(item, TextContent):
                text_parts.append(item.text)
            elif hasattr(item, "text"):
                text_parts.append(str(getattr(item, "text")))
            else:
                text_parts.append(str(item))
        return "\n".join(text_parts)

    async def disconnect(self) -> None:
        if self._connection_task is None:
            return
        self._connection_task.cancel()
        try:
            await self._connection_task
        except asyncio.CancelledError:
            pass
        self.logger.info(f"MCP server {self.spec.name!r} terminated")


class MCPToolWrapper:
    def __init__(
        self,
        *,
        server_name: str,
        tool: MCPTool,
        connection: MCPServerConnection,
    ):
        self.server_name = server_name
        self.tool = tool
        self.connection = connection
        self.name = tool.name

    def defn(self) -> Tool:
        return Tool(
            name=self.tool.name,
            description=self.tool.description or "",
            parameters=dict(self.tool.inputSchema or {"type": "object"}),
        )

    async def __call__(self, resources, **kwargs):
        return await self.connection.call_tool(self.tool.name, kwargs)


class MCPToolRuntime:
    def __init__(self, specs: list[MCPServerSpec], logger: logging.Logger):
        self.specs = specs
        self.logger = logger
        self.connections: dict[str, MCPServerConnection] = {}
        self.tools: dict[str, MCPToolWrapper] = {}
        self._loop = asyncio.new_event_loop()
        self._thread = threading.Thread(target=self._run_loop, daemon=True)
        self._started = False

    def _run_loop(self) -> None:
        asyncio.set_event_loop(self._loop)
        self._loop.run_forever()

    def ensure_started(self) -> None:
        if self._started:
            return
        self._thread.start()
        future = asyncio.run_coroutine_threadsafe(self._connect(), self._loop)
        future.result()
        self._started = True

    async def _connect(self) -> None:
        for spec in self.specs:
            connection = MCPServerConnection(spec, self.logger)
            tools = await connection.connect()
            self.connections[spec.name] = connection
            for tool in tools.values():
                wrapper = MCPToolWrapper(
                    server_name=spec.name,
                    tool=tool,
                    connection=connection,
                )
                if wrapper.name in self.tools:
                    raise ValueError(f"Duplicate MCP tool name: {wrapper.name}")
                self.tools[wrapper.name] = wrapper
                self.logger.info(
                    f"Registered MCP tool {wrapper.name!r} from server {spec.name!r}"
                )

    async def teardown(self) -> None:
        if not self._started:
            return
        futures = [
            asyncio.run_coroutine_threadsafe(connection.disconnect(), self._loop)
            for connection in self.connections.values()
        ]
        for future in futures:
            await asyncio.wrap_future(future)
        self.connections.clear()
        self.tools.clear()
        self._loop.call_soon_threadsafe(self._loop.stop)
        self._thread.join(timeout=5)


class ToolRegistry(Mapping[str, Any]):
    """Name-indexed collection of callable, stateful, and MCP tools."""

    def __init__(self, tools: Iterable[Any] | None = None):
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        self._tools: dict[str, Any] = {}
        self._mcp_specs: list[MCPServerSpec] = []
        self._mcp_runtime: MCPToolRuntime | None = None
        for tool in tools or []:
            self.add(tool)

    def add(self, tool: Any) -> None:
        if isinstance(tool, MCPServerSpec):
            self._mcp_specs.append(tool)
            return
        name = self._tool_name(tool)
        if name in self._tools:
            raise ValueError(f"Duplicate tool name: {name}")
        self._tools[name] = self._normalize_tool(tool)

    def extend(self, tools: Iterable[Any]) -> None:
        if isinstance(tools, ToolRegistry):
            for tool in tools._tools.values():
                self.add(tool)
            self._mcp_specs.extend(tools._mcp_specs)
            return
        for tool in tools:
            self.add(tool)

    def merged(self, tools: Iterable[Any]) -> ToolRegistry:
        registry = ToolRegistry()
        registry._tools = dict(self._tools)
        registry.extend(tools)
        return registry

    def ensure_ready(self) -> None:
        if not self._mcp_specs:
            return
        if self._mcp_runtime is None:
            self._mcp_runtime = MCPToolRuntime(self._mcp_specs, self.logger)
        self._mcp_runtime.ensure_started()
        for name, tool in self._mcp_runtime.tools.items():
            if name in self._tools:
                raise ValueError(f"Duplicate tool name: {name}")
            self._tools[name] = tool
        self._mcp_specs = []

    def defs(self) -> list[Tool]:
        self.ensure_ready()
        defs: list[Tool] = []
        for tool in self._tools.values():
            defs.append(self._tool_def(tool))
        return defs

    def names(self) -> list[str]:
        self.ensure_ready()
        return list(self._tools)

    async def call(
        self,
        name: str,
        resources: Any,
        arguments: str | Mapping[str, Any] | None = None,
        *,
        task: Any = None,
        state: Any = None,
    ) -> Any:
        self.ensure_ready()
        tool = self._tools[name]
        args = self._parse_arguments(arguments)
        if isinstance(tool, Tool):
            raise TypeError(f"Tool {name!r} has a schema but no callable.")
        if isinstance(tool, Mapping):
            raise TypeError(f"Tool {name!r} has a schema but no callable.")
        if isinstance(tool, CallableTool):
            return await self._call_callable_tool(tool, resources, args, task, state)
        return await maybe_await(
            tool, **self._inject_args(tool, resources, args, task, state)
        )

    async def teardown(self) -> None:
        if self._mcp_runtime is not None:
            await self._mcp_runtime.teardown()

    def _normalize_tool(self, tool: Any) -> Any:
        if isinstance(tool, CallableTool | Tool | MCPToolWrapper):
            return tool
        return tool

    def _tool_def(self, tool: Any) -> Tool:
        if isinstance(tool, Tool):
            return tool
        if isinstance(tool, Mapping):
            payload = dict(tool)
            function_payload = payload.get("function")
            if payload.get("type") == "function" and isinstance(
                function_payload, Mapping
            ):
                return Tool(
                    name=str(function_payload.get("name", "")),
                    description=str(function_payload.get("description", "")),
                    parameters=dict(function_payload.get("parameters") or {}),
                    strict=function_payload.get("strict"),  # type: ignore[arg-type]
                )
            return Tool.model_validate(payload)
        if isinstance(tool, CallableTool):
            return self._callable_tool_def(tool)
        if isinstance(tool, MCPToolWrapper):
            return tool.defn()
        schema = getattr(tool, "schema", None)
        if schema is not None:
            if callable(schema):
                schema = schema()
            return Tool.model_validate(schema)
        if callable(tool):
            return convert_func_to_tool_def(self._filtered_callable(tool))
        raise TypeError(f"Cannot build tool definition for {tool!r}")

    def _callable_tool_def(self, tool: CallableTool) -> Tool:
        defn = convert_func_to_tool_def(
            self._filtered_callable(tool.func, extra_hidden=set(tool.injected_args))
        )
        if tool.name is not None:
            defn.name = tool.name
        if tool.description is not None:
            defn.description = tool.description
        return defn

    def _filtered_callable(
        self,
        func: Callable[..., Any],
        *,
        extra_hidden: set[str] | None = None,
    ) -> Callable[..., Any]:
        sig = inspect.signature(func)
        hidden_args = set(INJECTABLE_TOOL_ARGS)
        if extra_hidden:
            hidden_args.update(extra_hidden)
        hidden_args = hidden_args & set(sig.parameters)
        if not hidden_args:
            return func
        filtered_sig = sig.replace(
            parameters=[
                parameter
                for name, parameter in sig.parameters.items()
                if name not in hidden_args and name != "self"
            ]
        )
        filtered_annotations = {
            key: value
            for key, value in getattr(func, "__annotations__", {}).items()
            if key not in hidden_args
        }

        def wrapper(*args, **kwargs):
            return func(*args, **kwargs)

        wrapper.__name__ = getattr(func, "__name__", func.__class__.__name__)
        wrapper.__doc__ = getattr(func, "__doc__", None)
        wrapper.__signature__ = filtered_sig  # type: ignore[attr-defined]
        wrapper.__annotations__ = filtered_annotations
        return wrapper

    async def _call_callable_tool(
        self,
        tool: CallableTool,
        resources: Any,
        args: dict[str, Any],
        task: Any,
        state: Any,
    ) -> Any:
        call_args = self._inject_args(
            tool.func,
            resources,
            args,
            task,
            state,
            extra_injected=set(tool.injected_args),
        )
        return await maybe_await(tool.func, **call_args)

    def _inject_args(
        self,
        func: Callable[..., Any],
        resources: Any,
        args: dict[str, Any],
        task: Any,
        state: Any,
        *,
        extra_injected: set[str] | None = None,
    ) -> dict[str, Any]:
        call_args = dict(args)
        injectable = set(INJECTABLE_TOOL_ARGS)
        if extra_injected:
            injectable.update(extra_injected)
        params = set(inspect.signature(func).parameters)
        for name in params & injectable:
            if name in call_args:
                continue
            value = self._injected_value(name, resources, task, state)
            if value is not _NO_VALUE:
                call_args[name] = value
        return call_args

    def _injected_value(self, name: str, resources: Any, task: Any, state: Any) -> Any:
        if name == "resources":
            return resources
        if name == "state":
            return state
        if name == "task":
            return task
        if name == "client":
            return resources.client
        if name == "model":
            return resources.model
        if name == "sampling_args":
            return resources.sampling_args
        if name == "tools":
            return resources.tools
        return _NO_VALUE

    @staticmethod
    def _parse_arguments(arguments: str | Mapping[str, Any] | None) -> dict[str, Any]:
        if arguments is None:
            return {}
        if isinstance(arguments, str):
            try:
                parsed = json.loads(arguments or "{}")
            except json.JSONDecodeError as e:
                raise ToolArgumentError(str(e)) from e
        else:
            parsed = dict(arguments)
        if not isinstance(parsed, dict):
            raise ToolArgumentError(
                f"Expected tool arguments to be a dict, got {type(parsed).__name__}."
            )
        return parsed

    @staticmethod
    def _tool_name(tool: Any) -> str:
        if isinstance(tool, CallableTool):
            if tool.name:
                return tool.name
            return getattr(tool.func, "__name__", tool.func.__class__.__name__)
        if isinstance(tool, Tool):
            return tool.name
        if isinstance(tool, Mapping):
            if "name" in tool:
                return str(tool["name"])
            function_payload = tool.get("function")
            if isinstance(function_payload, Mapping) and "name" in function_payload:
                return str(function_payload["name"])
        name = getattr(tool, "name", None)
        if isinstance(name, str) and name:
            return name
        function_name = getattr(tool, "__name__", None)
        if isinstance(function_name, str) and function_name:
            return function_name
        raise ValueError(f"Tool has no name: {tool!r}")

    def __getitem__(self, key: str) -> Any:
        self.ensure_ready()
        return self._tools[key]

    def __iter__(self) -> Iterator[str]:
        self.ensure_ready()
        return iter(self._tools)

    def __len__(self) -> int:
        self.ensure_ready()
        return len(self._tools)


class _NoValue:
    pass


_NO_VALUE = _NoValue()


def resolve_tools(
    configs: list[ChannelConfig], context: ChannelContext
) -> dict[str, object]:
    registry = ToolRegistry()
    for config in configs:
        if isinstance(config, ToolRegistry):
            registry.extend(config)
        else:
            registry.extend(as_list(config))
    return {"tools": registry}


tools_channel = Channel(name="tools", outputs=("tools",), resolve_fn=resolve_tools)
