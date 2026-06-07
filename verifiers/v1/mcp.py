from __future__ import annotations

from contextlib import AsyncExitStack
from collections.abc import Mapping
from copy import deepcopy
from dataclasses import dataclass
import json
from typing import TYPE_CHECKING, cast

from pydantic import Field
from verifiers.errors import ToolError
from verifiers.types import Messages, TextMessage, Tool
from verifiers.utils.message_utils import normalize_messages

from .config import Config
from .toolset import ServerConfig, ToolBinding
from .types import JsonData, JsonValue
from .utils.json_utils import jsonable

if TYPE_CHECKING:
    from mcp import ClientSession


@dataclass(frozen=True)
class BoundUpdate:
    target: str
    value: JsonValue
    mode: str = "set"


class ServerResponse(Config):
    messages: Messages = Field(default_factory=list)


@dataclass(frozen=True)
class ServerResult:
    response: ServerResponse
    updates: tuple[BoundUpdate, ...] = ()


@dataclass(frozen=True)
class ToolDispatch:
    session: "ClientSession"
    raw_name: str
    binding: ToolBinding


class MCPToolRegistry:
    def __init__(
        self,
        servers: list[ServerConfig],
        *,
        parents: list["MCPToolRegistry"] | None = None,
        expose_tools: bool = True,
    ) -> None:
        self.servers = servers
        self.parents = parents or []
        self.expose_tools = expose_tools
        self._stack = AsyncExitStack()
        self._dispatch: dict[str, ToolDispatch] = {}
        self._tools: list[Tool] = []
        self._context: JsonData = {}

    def set_context(self, context: JsonData) -> None:
        self._context = context
        for parent in self.parents:
            parent.set_context(context)

    async def __aenter__(self) -> "MCPToolRegistry":
        if not self.servers:
            return self
        from mcp import ClientSession
        from mcp.client.stdio import StdioServerParameters, stdio_client

        for server in self.servers:
            bindings = server.tool_bindings()
            seen_tools: set[str] = set()
            toolset_name = server.resolved_name()
            command = server.server_command()
            read, write = await self._stack.enter_async_context(
                stdio_client(
                    StdioServerParameters(
                        command=command[0],
                        args=command[1:],
                        env=server.server_env() or None,
                    )
                )
            )
            session = await self._stack.enter_async_context(ClientSession(read, write))
            await session.initialize()
            for raw_tool in (await session.list_tools()).tools:
                tool_name = getattr(raw_tool, "name", "")
                if not isinstance(tool_name, str) or not tool_name:
                    raise TypeError("MCP tools require a non-empty name.")
                seen_tools.add(tool_name)
                exposed_name = f"{toolset_name}_{tool_name}"
                if exposed_name in self._dispatch:
                    raise ValueError(f"MCP tool {exposed_name!r} is defined twice.")
                schema = getattr(raw_tool, "inputSchema", None)
                if not isinstance(schema, dict):
                    schema = {"type": "object", "properties": {}}
                binding = bindings.get(
                    tool_name, ToolBinding(args={}, sets={}, extends={})
                )
                visible_schema = model_visible_schema(tool_name, schema, binding)
                if tool_visible(server, tool_name):
                    self._tools.append(
                        Tool(
                            name=exposed_name,
                            description=str(getattr(raw_tool, "description", "") or ""),
                            parameters=visible_schema,
                        )
                    )
                self._dispatch[exposed_name] = ToolDispatch(
                    session=session, raw_name=tool_name, binding=binding
                )
            missing = sorted(set(bindings) - seen_tools)
            if missing:
                raise ValueError(
                    f"Toolset {toolset_name!r} binds unknown tools: "
                    f"{', '.join(missing)}."
                )
        return self

    async def __aexit__(self, *exc: object) -> None:
        await self._stack.aclose()

    def tool_defs(self, *, include_hidden: bool = False) -> list[Tool] | None:
        tools: list[Tool] = []
        for parent in self.parents:
            tools.extend(parent.tool_defs(include_hidden=include_hidden) or [])
        if self.expose_tools or include_hidden:
            tools.extend(self._tools)
        return tools or None

    def has_tool(self, name: str) -> bool:
        return name in self._dispatch or any(
            parent.has_tool(name) for parent in self.parents
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
        if name not in self._dispatch:
            for parent in self.parents:
                if parent.has_tool(name):
                    return await parent.call(name, arguments)
            raise ToolError(f"Unknown MCP tool {name!r}.")
        dispatch = self._dispatch[name]
        payload: JsonData = dict(arguments)
        for arg_name, source in dispatch.binding.args.items():
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
        matches = self.hidden_matches(raw_name)
        if not matches:
            raise ToolError(f"Unknown hidden MCP tool {raw_name!r}.")
        if len(matches) > 1:
            raise ToolError(f"Hidden MCP tool {raw_name!r} is ambiguous.")
        return await self.call(matches[0], arguments)

    def has_hidden(self, raw_name: str) -> bool:
        if any(dispatch.raw_name == raw_name for dispatch in self._dispatch.values()):
            return True
        return any(parent.has_hidden(raw_name) for parent in self.parents)


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
        return cast(JsonValue, jsonable(model_dump(mode="json", exclude_none=True)))
    return cast(JsonValue, jsonable(item))


def tool_visible(server: ServerConfig, tool_name: str) -> bool:
    if server.show is not None:
        return tool_name in server.show
    if server.hide is not None:
        return tool_name not in server.hide
    return True


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
    missing = sorted(set(binding.args) - {str(key) for key in properties})
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
        if isinstance(value, Mapping):
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
    return cast(JsonValue, deepcopy(jsonable(value)))


def split_result(content: JsonValue, binding: ToolBinding) -> ServerResult:
    if not binding.sets and not binding.extends:
        return ServerResult(response=server_response(content))
    result = mapping_result(content)
    visible = dict(result)
    updates: list[BoundUpdate] = []
    for result_key, target in binding.sets.items():
        if result_key not in result:
            continue
        updates.append(
            BoundUpdate(
                target=target,
                value=cast(JsonValue, deepcopy(result[result_key])),
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
                value=cast(JsonValue, deepcopy(result[result_key])),
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
    )


def mapping_result(content: JsonValue) -> Mapping[str, JsonValue]:
    if isinstance(content, Mapping):
        return cast(Mapping[str, JsonValue], content)
    if isinstance(content, str):
        parsed = json.loads(content)
        if isinstance(parsed, Mapping):
            return cast(Mapping[str, JsonValue], parsed)
    raise TypeError("Bound MCP tool returns must be JSON objects.")


def server_response(content: JsonValue) -> ServerResponse:
    if content == "":
        return ServerResponse()
    if isinstance(content, str):
        try:
            parsed = json.loads(content)
        except json.JSONDecodeError:
            return ServerResponse(messages=[TextMessage(content=content)])
        if isinstance(parsed, Mapping) and "messages" in parsed:
            return ServerResponse.model_validate(parsed)
        return ServerResponse(messages=[TextMessage(content=content)])
    if isinstance(content, Mapping):
        if "messages" in content:
            return ServerResponse.model_validate(content)
        return ServerResponse(messages=[TextMessage(content=json.dumps(content))])
    if isinstance(content, list):
        return ServerResponse(
            messages=normalize_messages(content, field_name="server.messages")
        )
    return ServerResponse(messages=[TextMessage(content=json.dumps(content))])
