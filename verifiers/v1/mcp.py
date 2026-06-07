from __future__ import annotations

import asyncio
import contextlib
from collections.abc import Mapping
from contextlib import AsyncExitStack
from copy import deepcopy
from dataclasses import dataclass
import inspect
import json
import os
import random
import socket
import sys
from collections.abc import Callable
from typing import TYPE_CHECKING, cast
import urllib.error
import urllib.request

from pydantic import Field
from pydantic import ValidationError
from verifiers.errors import ToolError
from verifiers.types import MessageContent, Messages, Tool

from .config import Config
from .runtime import Runtime, SubprocessRuntime, make_runtime_provider
from .toolset import (
    ServerConfig,
    ToolBinding,
    ToolSpec,
    Toolset,
)
from .types import JsonData, JsonValue
from .utils.config_utils import import_config_ref, registered_config_type
from .utils.json_utils import json_data, json_value

if TYPE_CHECKING:
    from mcp.client.session import ClientSession
    from .task import TaskVisibility

_BINDINGS_TOOL = "__vf_bindings"


@dataclass(frozen=True)
class BoundUpdate:
    target: str
    value: JsonValue
    mode: str = "set"


class ServerResponse(Config):
    content: MessageContent | None = None
    messages: Messages = Field(default_factory=list)


@dataclass(frozen=True)
class ServerResult:
    response: ServerResponse
    updates: tuple[BoundUpdate, ...] = ()
    value: JsonValue = ""


@dataclass(frozen=True)
class ToolDispatch:
    session: "ClientSession"
    toolset_name: str
    raw_name: str
    binding: ToolBinding
    dynamic_name: str | None = None
    name_arg: str = "name"
    input_arg: str = "input"


class MCPToolRegistry:
    def __init__(
        self,
        servers: Mapping[str, ServerConfig],
        *,
        runtime: Runtime | None = None,
        parents: list["MCPToolRegistry"] | None = None,
        expose_tools: bool = True,
    ) -> None:
        self.servers = dict(servers)
        self.runtime = runtime
        self.parents = parents or []
        self.expose_tools = expose_tools
        self._stack = AsyncExitStack()
        self._dispatch: dict[str, ToolDispatch] = {}
        self._dynamic_dispatch: dict[str, ToolDispatch] = {}
        self._tools: list[Tool] = []
        self._dynamic_tools: list[Tool] = []
        self._context: JsonData = {}
        self._toolsets_visibility: TaskVisibility | None = None
        self._tools_visibility: TaskVisibility | None = None
        self._resolution_key: str | None = None

    def set_context(self, context: JsonData) -> None:
        self._context = context
        for parent in self.parents:
            parent.set_context(context)

    def set_visibility(
        self,
        *,
        toolsets: "TaskVisibility | None",
        tools: "TaskVisibility | None",
    ) -> None:
        self._toolsets_visibility = toolsets
        self._tools_visibility = tools
        for parent in self.parents:
            parent.set_visibility(toolsets=toolsets, tools=tools)

    async def __aenter__(self) -> "MCPToolRegistry":
        if not self.servers:
            return self
        from mcp.client.session import ClientSession

        for toolset_name, server in self.servers.items():
            if not isinstance(toolset_name, str) or not toolset_name:
                raise TypeError("MCP server names must be non-empty strings.")
            seen_tools: set[str] = set()
            read, write = await self._stack.enter_async_context(
                self.open_server(toolset_name, server)
            )
            session = await self._stack.enter_async_context(ClientSession(read, write))
            await session.initialize()
            bindings = await server_bindings(session)
            for raw_tool in (await session.list_tools()).tools:
                tool_name = getattr(raw_tool, "name", "")
                if not isinstance(tool_name, str) or not tool_name:
                    raise TypeError("MCP tools require a non-empty name.")
                if tool_name == _BINDINGS_TOOL:
                    continue
                seen_tools.add(tool_name)
                exposed_name = f"{toolset_name}_{tool_name}"
                if exposed_name in self._dispatch:
                    raise ValueError(f"MCP tool {exposed_name!r} is defined twice.")
                raw_schema = getattr(raw_tool, "inputSchema", None)
                schema: dict[str, object] = (
                    dict(raw_schema)
                    if isinstance(raw_schema, dict)
                    else {"type": "object", "properties": {}}
                )
                binding = bindings.get(
                    tool_name, ToolBinding(args={}, sets={}, extends={}, hidden=False)
                )
                visible_schema = model_visible_schema(tool_name, schema, binding)
                if tool_visible(server, tool_name) and not binding.hidden:
                    self._tools.append(
                        Tool(
                            name=exposed_name,
                            description=str(getattr(raw_tool, "description", "") or ""),
                            parameters=visible_schema,
                        )
                    )
                self._dispatch[exposed_name] = ToolDispatch(
                    session=session,
                    toolset_name=toolset_name,
                    raw_name=tool_name,
                    binding=binding,
                )
            missing = sorted(set(bindings) - seen_tools)
            if missing:
                raise ValueError(
                    f"Toolset {toolset_name!r} binds unknown tools: "
                    f"{', '.join(missing)}."
                )
        return self

    @contextlib.asynccontextmanager
    async def open_server(self, name: str, server: ServerConfig):
        if server.placement == "remote":
            if server.url is None:
                raise ValueError("Remote server requires url.")
            async with connect_streamable_http(server.url, server.headers) as (
                read,
                write,
            ):
                yield read, write
            return

        owns_runtime = False
        runtime = self.runtime
        if server.placement == "runtime":
            runtime_config = server.runtime
            if runtime_config is None:
                raise ValueError("Runtime server placement requires runtime config.")
            runtime = make_runtime_provider(runtime_config).create_runtime()
            owns_runtime = True
            await runtime.start()
        elif runtime is None:
            raise ValueError("Harness server placement requires a running runtime.")

        assert runtime is not None
        try:
            port = free_port()
            env = _server_env(server)
            env["MCP_PORT"] = str(port)
            log = f"vf_server_{name}.log"
            await runtime.run_background(
                _server_command(
                    server,
                    name,
                    in_runtime=not isinstance(runtime, SubprocessRuntime),
                ),
                env=env,
                log=log,
            )
            base_url = await runtime.public_url(port)
            if base_url is None:
                base_url = f"http://127.0.0.1:{port}"
            url = f"{base_url.rstrip('/')}/mcp"
            await wait_for_http(url)
            async with connect_streamable_http(url, server.headers) as (read, write):
                yield read, write
        finally:
            if owns_runtime:
                await runtime.stop()

    async def __aexit__(self, *exc: object) -> None:
        await self._stack.aclose()

    def tools(self, *, include_hidden: bool = False) -> list[Tool] | None:
        tools: list[Tool] = []
        for parent in self.parents:
            tools.extend(parent.tools(include_hidden=include_hidden) or [])
        if self.expose_tools or include_hidden:
            tools.extend(
                tool
                for tool in self._tools
                if include_hidden or self.tool_allowed(tool.name)
            )
            tools.extend(
                tool
                for tool in self._dynamic_tools
                if include_hidden or self.tool_allowed(tool.name)
            )
        return tools or None

    def has_tool(self, name: str) -> bool:
        return (
            name in self._dispatch
            or name in self._dynamic_dispatch
            or any(parent.has_tool(name) for parent in self.parents)
        )

    def hidden_matches(self, raw_name: str) -> list[str]:
        matches = [
            exposed_name
            for exposed_name, dispatch in self._dispatch.items()
            if dispatch.raw_name == raw_name
        ]
        for parent in self.parents:
            matches.extend(parent.hidden_matches(raw_name))
        return matches

    async def call(self, name: str, arguments: JsonData) -> ServerResult:
        if name not in self._dispatch and name not in self._dynamic_dispatch:
            for parent in self.parents:
                if parent.has_tool(name):
                    return await parent.call(name, arguments)
            raise ToolError(f"Unknown MCP tool {name!r}.")
        if not self.tool_allowed(name):
            raise ToolError(f"MCP tool {name!r} is disabled for this task.")
        return await self._call(name, arguments)

    async def _call(self, name: str, arguments: JsonData) -> ServerResult:
        dispatch = self._dynamic_dispatch.get(name) or self._dispatch[name]
        payload: JsonData
        if dispatch.dynamic_name is None:
            payload = dict(arguments)
        else:
            payload = {
                dispatch.name_arg: dispatch.dynamic_name,
                dispatch.input_arg: dict(arguments),
            }
        for arg_name, source in dispatch.binding.args.items():
            if source.startswith("resources."):
                continue
            if arg_name in payload:
                raise ToolError(
                    f"MCP tool {name!r} argument {arg_name!r} is bound and cannot "
                    "be provided by the model."
                )
            payload[arg_name] = resolve_binding(self._context, source)
        result = await dispatch.session.call_tool(dispatch.raw_name, payload)
        if bool(getattr(result, "isError", False)):
            raise ToolError(str(mcp_content_value(getattr(result, "content", []))))
        content = mcp_content_value(getattr(result, "content", []))
        return split_result(content, dispatch.binding)

    async def call_hidden(self, raw_name: str, arguments: JsonData) -> ServerResult:
        local_matches = [
            exposed_name
            for exposed_name, dispatch in self._dispatch.items()
            if dispatch.raw_name == raw_name
        ]
        parent_matches = [
            parent for parent in self.parents if parent.hidden_matches(raw_name)
        ]
        matches = [*local_matches, *parent_matches]
        if not matches:
            raise ToolError(f"Unknown hidden MCP tool {raw_name!r}.")
        if len(matches) > 1:
            raise ToolError(f"Hidden MCP tool {raw_name!r} is ambiguous.")
        match = matches[0]
        if isinstance(match, MCPToolRegistry):
            return await match.call_hidden(raw_name, arguments)
        return await self._call(match, arguments)

    def has_hidden(self, raw_name: str) -> bool:
        if any(dispatch.raw_name == raw_name for dispatch in self._dispatch.values()):
            return True
        return any(parent.has_hidden(raw_name) for parent in self.parents)

    def tool_allowed(self, name: str) -> bool:
        dispatch = self._dynamic_dispatch.get(name)
        if dispatch is not None:
            return visibility_allows(
                dispatch.toolset_name, self._toolsets_visibility
            ) and visibility_allows(name, self._tools_visibility)
        dispatch = self._dispatch.get(name)
        if dispatch is not None:
            return (
                not dispatch.binding.hidden
                and visibility_allows(dispatch.toolset_name, self._toolsets_visibility)
                and visibility_allows(name, self._tools_visibility)
            )
        for parent in self.parents:
            if parent.has_tool(name):
                return parent.tool_allowed(name)
        return False

    async def resolve(
        self,
        *,
        context: JsonData,
        resolution_key: str,
        apply_updates: Callable[[list[BoundUpdate]], None] | None = None,
    ) -> None:
        if self._resolution_key == resolution_key:
            return
        self._dynamic_dispatch.clear()
        self._dynamic_tools.clear()
        self.set_context(context)
        updates: list[BoundUpdate] = []
        for setup in self.setup_dispatches():
            result = await self.call_dispatch(setup, {})
            if result.updates:
                updates.extend(result.updates)
            self.register_setup_tools(setup, result.value)
        if updates:
            if apply_updates is None:
                raise RuntimeError("Toolset setup returned state updates.")
            apply_updates(updates)
        self._resolution_key = resolution_key

    def setup_dispatches(self) -> list[ToolDispatch]:
        dispatches: list[ToolDispatch] = []
        for parent in self.parents:
            dispatches.extend(parent.setup_dispatches())
        for dispatch in self._dispatch.values():
            if dispatch.raw_name != "setup":
                continue
            if not dispatch.binding.hidden:
                raise ValueError(
                    f"Toolset {dispatch.toolset_name!r} setup tool must be hidden."
                )
            if visibility_allows(dispatch.toolset_name, self._toolsets_visibility):
                dispatches.append(dispatch)
        return dispatches

    async def call_dispatch(
        self, dispatch: ToolDispatch, arguments: JsonData
    ) -> ServerResult:
        payload: JsonData = dict(arguments)
        for arg_name, source in dispatch.binding.args.items():
            if source.startswith("resources."):
                continue
            if arg_name in payload:
                raise ToolError(
                    f"MCP tool {dispatch.raw_name!r} argument {arg_name!r} is bound "
                    "and cannot be provided by the model."
                )
            payload[arg_name] = resolve_binding(self._context, source)
        result = await dispatch.session.call_tool(dispatch.raw_name, payload)
        if bool(getattr(result, "isError", False)):
            raise ToolError(str(mcp_content_value(getattr(result, "content", []))))
        content = mcp_content_value(getattr(result, "content", []))
        return split_result(content, dispatch.binding)

    def register_setup_tools(self, setup: ToolDispatch, value: JsonValue) -> None:
        if not isinstance(value, dict):
            return
        if "messages" in value:
            raise ValueError("Toolset setup cannot return messages.")
        raw_tools = value.get("tools")
        if raw_tools is None:
            return
        if not isinstance(raw_tools, list):
            raise TypeError("Toolset setup tools must be a list.")
        through = string_field(value, "through", default="call_tool")
        name_arg = string_field(value, "name_arg", default="name")
        input_arg = string_field(value, "input_arg", default="input")
        route = self.dispatch_for(setup.toolset_name, through)
        if not route.binding.hidden:
            raise ValueError(
                f"Dynamic tool route {setup.toolset_name}.{through} must be hidden."
            )
        server = self.server_config(route.toolset_name)
        for raw_tool in raw_tools:
            tool = Tool.model_validate(raw_tool)
            if server is not None and not tool_visible(server, tool.name):
                continue
            if self.has_tool(tool.name):
                raise ValueError(f"Dynamic tool {tool.name!r} is defined twice.")
            self._dynamic_tools.append(tool)
            self._dynamic_dispatch[tool.name] = ToolDispatch(
                session=route.session,
                toolset_name=route.toolset_name,
                raw_name=route.raw_name,
                binding=route.binding,
                dynamic_name=tool.name,
                name_arg=name_arg,
                input_arg=input_arg,
            )

    def server_config(self, toolset_name: str) -> ServerConfig | None:
        server = self.servers.get(toolset_name)
        if server is not None:
            return server
        for parent in self.parents:
            server = parent.server_config(toolset_name)
            if server is not None:
                return server
        return None

    def dispatch_for(self, toolset_name: str, raw_name: str) -> ToolDispatch:
        for dispatch in self._dispatch.values():
            if dispatch.toolset_name == toolset_name and dispatch.raw_name == raw_name:
                return dispatch
        for parent in self.parents:
            try:
                return parent.dispatch_for(toolset_name, raw_name)
            except KeyError:
                pass
        raise KeyError(f"Toolset {toolset_name!r} has no hidden tool {raw_name!r}.")


async def server_bindings(session: "ClientSession") -> dict[str, ToolBinding]:
    result = await session.call_tool(_BINDINGS_TOOL, {})
    if bool(getattr(result, "isError", False)):
        raise ToolError(str(mcp_content_value(getattr(result, "content", []))))
    content = mcp_content_value(getattr(result, "content", []))
    data = mapping_result(content)
    tools = data.get("tools", {})
    if not isinstance(tools, dict):
        raise TypeError("Server bindings metadata must contain a tools object.")
    bindings: dict[str, ToolBinding] = {}
    for tool_name, value in tools.items():
        if not isinstance(tool_name, str):
            raise TypeError("Server binding tool names must be strings.")
        if not isinstance(value, dict):
            raise TypeError(f"Server binding {tool_name!r} must be an object.")
        bindings[tool_name] = ToolBinding(
            args=dict_mapping(value.get("args", {}), field=f"{tool_name}.args"),
            sets=dict_mapping(value.get("sets", {}), field=f"{tool_name}.sets"),
            extends=dict_mapping(
                value.get("extends", {}), field=f"{tool_name}.extends"
            ),
            hidden=bool(value.get("hidden", False)),
        )
    return bindings


def dict_mapping(value: object, *, field: str) -> dict[str, str]:
    if not isinstance(value, Mapping):
        raise TypeError(f"Server binding {field} must be an object.")
    result: dict[str, str] = {}
    for key, item in value.items():
        if not isinstance(key, str) or not isinstance(item, str):
            raise TypeError(f"Server binding {field} must map strings to strings.")
        result[key] = item
    return result


def string_field(data: Mapping[str, JsonValue], field: str, *, default: str) -> str:
    value = data.get(field, default)
    if not isinstance(value, str) or not value:
        raise TypeError(f"Toolset setup {field} must be a non-empty string.")
    return value


def free_port() -> int:
    for _ in range(50):
        port = random.randint(3000, 8999)
        probe = socket.socket()
        try:
            probe.bind(("127.0.0.1", port))
            return port
        except OSError:
            continue
        finally:
            probe.close()
    raise RuntimeError("Could not find a free port in [3000, 9000).")


async def wait_for_http(url: str) -> None:
    for _ in range(180):
        if await asyncio.to_thread(http_serves, url):
            return
        await asyncio.sleep(0.1)
    raise RuntimeError(f"MCP server did not start at {url}.")


@contextlib.asynccontextmanager
async def connect_streamable_http(url: str, headers: dict[str, str]):
    from mcp.client.streamable_http import streamablehttp_client

    async with streamablehttp_client(url, headers=headers or None) as (
        read,
        write,
        *_,
    ):
        yield read, write


def http_serves(url: str) -> bool:
    try:
        urllib.request.urlopen(url, timeout=2)
        return True
    except urllib.error.HTTPError:
        return True
    except Exception:
        return False


def mcp_content_value(content: object) -> JsonValue:
    if not isinstance(content, list):
        return serializable_content(content)
    values = [serializable_content(item) for item in content]
    if len(values) == 1:
        return values[0]
    return values


def serializable_content(item: object) -> JsonValue:
    item_type = getattr(item, "type", None)
    text = getattr(item, "text", None)
    if item_type == "text" and isinstance(text, str):
        return text
    model_dump = getattr(item, "model_dump", None)
    if callable(model_dump):
        return json_value(model_dump(mode="json", exclude_none=True))
    return json_value(item)


def tool_visible(server: ServerConfig, tool_name: str) -> bool:
    if server.show is not None:
        return tool_name in server.show
    if server.hide is not None:
        return tool_name not in server.hide
    return True


def visibility_allows(name: str, visibility: "TaskVisibility | None") -> bool:
    if visibility is None:
        return True
    if visibility.show is not None:
        return name in visibility.show
    if visibility.hide is not None:
        return name not in visibility.hide
    return True


def _server_command(
    server: ServerConfig, name: str, *, in_runtime: bool = False
) -> list[str]:
    python = "python" if in_runtime else sys.executable
    payload = {
        "name": name,
        "server": server.implementation_ref(),
        "config": server.model_dump(mode="json"),
    }
    return [
        python,
        "-m",
        "verifiers.v1.mcp",
        json.dumps(payload),
        "streamable-http",
    ]


def _server_env(server: ServerConfig) -> dict[str, str]:
    resolved = dict(server.env)
    pythonpath = os.pathsep.join(path for path in sys.path if path)
    if pythonpath:
        resolved.setdefault("PYTHONPATH", pythonpath)
    return resolved


def build_fastmcp(toolset: Toolset):
    from mcp.server.fastmcp import FastMCP

    mcp = FastMCP(toolset.name)
    bindings = {
        name: {
            "args": dict(spec.args),
            "sets": dict(spec.sets),
            "extends": dict(spec.extends),
            "hidden": spec.hidden,
        }
        for name, spec in type(toolset).tool_specs().items()
    }

    @mcp.tool(name=_BINDINGS_TOOL)
    def vf_bindings() -> dict[str, object]:
        return {"tools": bindings}

    for method_name, method in inspect.getmembers(toolset, predicate=callable):
        spec = getattr(getattr(type(toolset), method_name, None), "__vf_tool__", None)
        if not isinstance(spec, ToolSpec):
            continue
        tool_name = spec.name or method_name
        mcp.tool(name=tool_name)(server_tool_wrapper(toolset, method, spec))
    return mcp


def server_tool_wrapper(
    toolset: Toolset, method: Callable[..., object], spec: ToolSpec
):
    resource_args = {
        arg_name: source
        for arg_name, source in spec.args.items()
        if source.startswith("resources.")
    }
    if not resource_args:
        return method
    signature = inspect.signature(method)
    parameters = [
        parameter
        for parameter in signature.parameters.values()
        if parameter.name not in resource_args
    ]

    async def invoke(**kwargs: object) -> object:
        for arg_name, source in resource_args.items():
            kwargs[arg_name] = resolve_resource(toolset, source)
        result = method(**kwargs)
        if inspect.isawaitable(result):
            return await result
        return result

    invoke.__name__ = getattr(method, "__name__", "tool")
    invoke.__doc__ = getattr(method, "__doc__", None)
    setattr(invoke, "__signature__", signature.replace(parameters=parameters))
    annotations = dict(getattr(method, "__annotations__", {}))
    for arg_name in resource_args:
        annotations.pop(arg_name, None)
    invoke.__annotations__ = annotations
    return invoke


def resolve_resource(toolset: Toolset, source: str) -> object:
    parts = source.split(".")
    if len(parts) < 2 or parts[0] != "resources":
        raise ValueError(f"Resource binding {source!r} must start with resources.")
    value: object = toolset.resources
    for part in parts[1:]:
        if isinstance(value, dict):
            value = cast(dict[str, object], value)[part]
        else:
            value = getattr(value, part)
    return value


def _run_toolset_server(config_json: str, transport: str = "stdio") -> None:
    payload = json.loads(config_json)
    if not isinstance(payload, dict):
        raise TypeError("Toolset runner payload must decode to an object.")
    server = payload.get("server")
    if not isinstance(server, str) or not server:
        raise TypeError("Toolset runner payload requires a server.")
    name = payload.get("name")
    if not isinstance(name, str) or not name:
        raise TypeError("Toolset runner payload requires a name.")
    data = payload.get("config")
    if not isinstance(data, dict):
        raise TypeError("Toolset runner payload requires a config object.")
    config_data = json_data(data, context="Toolset runner config")
    config_cls = config_type_for_server(server)
    config = config_cls.model_validate(config_data)
    toolset = Toolset.load_ref(server, config)
    toolset.name = name
    toolset.start()
    toolset.load_resources()
    try:
        mcp = build_fastmcp(toolset)
        if transport == "streamable-http":
            port = int(os.environ["MCP_PORT"])
            mcp.settings.host = "127.0.0.1"
            mcp.settings.port = port
            mcp.run(transport="streamable-http")
        else:
            mcp.run(transport="stdio")
    finally:
        toolset.stop()


def config_type_for_server(server: str) -> type[ServerConfig]:
    obj = import_config_ref(server)
    if isinstance(obj, type) and issubclass(obj, Toolset):
        return registered_config_type(obj, ServerConfig)
    return ServerConfig


def model_visible_schema(
    tool_name: str,
    schema: dict[str, object],
    binding: ToolBinding,
) -> dict[str, object]:
    if schema.get("type") != "object":
        if binding.args:
            raise ValueError(f"MCP tool {tool_name!r} has bindings but no args schema.")
        return dict(schema)
    properties = schema.get("properties")
    if not isinstance(properties, dict):
        if binding.args:
            raise ValueError(f"MCP tool {tool_name!r} has bindings but no properties.")
        return dict(schema)
    json_bound_args = {
        arg_name
        for arg_name, source in binding.args.items()
        if not source.startswith("resources.")
    }
    missing = sorted(json_bound_args - {str(key) for key in properties})
    if missing:
        raise ValueError(
            f"MCP tool {tool_name!r} binds unknown args: {', '.join(missing)}."
        )
    visible = dict(schema)
    visible["properties"] = {
        str(key): value
        for key, value in properties.items()
        if str(key) not in binding.args
    }
    required = schema.get("required")
    if isinstance(required, list):
        visible["required"] = [item for item in required if item not in binding.args]
    return visible


def resolve_binding(context: JsonData, source: str) -> JsonValue:
    if not source:
        raise ValueError("Binding source must be non-empty.")
    value: JsonValue = context
    for part in source.split("."):
        if not part:
            raise ValueError(f"Binding source {source!r} has an empty path segment.")
        if isinstance(value, dict):
            if part not in value:
                raise KeyError(f"Binding source {source!r} is missing {part!r}.")
            value = value[part]
        elif isinstance(value, list):
            if not part.isdigit():
                raise TypeError(
                    f"Binding source {source!r} indexes a list with {part!r}."
                )
            index = int(part)
            try:
                value = value[index]
            except IndexError as exc:
                raise IndexError(
                    f"Binding source {source!r} list index {index} is out of range."
                ) from exc
        else:
            raise TypeError(
                f"Binding source {source!r} cannot traverse {type(value).__name__}."
            )
    return deepcopy(value)


def split_result(content: JsonValue, binding: ToolBinding) -> ServerResult:
    if not binding.sets and not binding.extends:
        return ServerResult(response=server_response(content), value=deepcopy(content))
    result = mapping_result(content)
    visible = dict(result)
    updates: list[BoundUpdate] = []
    for result_key, target in binding.sets.items():
        if result_key not in result:
            continue
        updates.append(
            BoundUpdate(
                target=target,
                value=deepcopy(result[result_key]),
                mode="set",
            )
        )
        visible.pop(result_key, None)
    for result_key, target in binding.extends.items():
        if result_key not in result:
            continue
        updates.append(
            BoundUpdate(
                target=target,
                value=deepcopy(result[result_key]),
                mode="extend",
            )
        )
        visible.pop(result_key, None)
    if not visible:
        model_content: JsonValue = ""
    elif set(visible) == {"content"}:
        model_content = visible["content"]
    else:
        model_content = dict(visible)
    return ServerResult(
        response=server_response(model_content),
        updates=tuple(updates),
        value=deepcopy(dict(result)),
    )


def mapping_result(content: JsonValue) -> JsonData:
    if isinstance(content, dict):
        return content
    if isinstance(content, str):
        parsed = json.loads(content)
        if isinstance(parsed, dict):
            return json_data(parsed, context="Bound MCP tool return")
    raise TypeError("Bound MCP tool returns must be JSON objects.")


def server_response(content: JsonValue) -> ServerResponse:
    if content == "":
        return ServerResponse()
    if isinstance(content, str):
        try:
            parsed = json.loads(content)
        except json.JSONDecodeError:
            return ServerResponse(content=content)
        if isinstance(parsed, Mapping) and (
            "content" in parsed or "messages" in parsed
        ):
            return ServerResponse.model_validate(parsed)
        return ServerResponse(content=content)
    if isinstance(content, Mapping):
        if "content" in content or "messages" in content:
            return ServerResponse.model_validate(content)
        return ServerResponse(content=json.dumps(content))
    if isinstance(content, list):
        try:
            return ServerResponse(content=cast(MessageContent, content))
        except ValidationError:
            return ServerResponse(content=json.dumps(content))
    return ServerResponse(content=json.dumps(content))


def main() -> None:
    if len(sys.argv) not in (2, 3):
        raise SystemExit("usage: python -m verifiers.v1.mcp CONFIG_JSON [TRANSPORT]")
    transport = sys.argv[2] if len(sys.argv) == 3 else "stdio"
    _run_toolset_server(sys.argv[1], transport)


if __name__ == "__main__":
    main()
