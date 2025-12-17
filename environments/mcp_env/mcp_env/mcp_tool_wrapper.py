from typing import Any

from mcp.types import Tool

from .mcp_server_connection import MCPServerConnection


class MCPToolWrapper:
    def __init__(
        self, server_name: str, tool: Tool, server_connection: MCPServerConnection
    ):
        self.server_name = server_name
        self.tool = tool
        self.server_connection = server_connection

        self.__name__ = tool.name
        self.__doc__ = tool.description or ""

        self.__annotations__ = self._build_annotations()

    def _build_annotations(self) -> dict:
        annotations = {}

        if self.tool.inputSchema:
            properties = self.tool.inputSchema.get("properties", {})

            for param_name, param_spec in properties.items():
                param_type = param_spec.get("type", "string")
                if param_type == "string":
                    annotations[param_name] = str
                elif param_type == "integer":
                    annotations[param_name] = int
                elif param_type == "number":
                    annotations[param_name] = float
                elif param_type == "boolean":
                    annotations[param_name] = bool
                elif param_type == "array":
                    annotations[param_name] = list
                elif param_type == "object":
                    annotations[param_name] = dict
                else:
                    annotations[param_name] = Any

        annotations["return"] = str
        return annotations

    async def __call__(self, **kwargs):
        return await self.server_connection.call_tool(self.tool.name, kwargs)

    def to_oai_tool(self) -> dict:
        return {
            "type": "function",
            "function": {
                "name": self.__name__,
                "description": self.__doc__ or "",
                "parameters": self.tool.inputSchema
                or {"type": "object", "properties": {}},
            },
        }
