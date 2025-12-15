from typing import Any
import copy
from mcp.types import Tool

from verifiers.utils.mcp_utils.transports.base import MCPTransport


class MCPToolWrapper:
    def __init__(self, server_name: str, tool: Tool, server_connection: MCPTransport):
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

    def _remove_additional_properties(self, schema: dict) -> dict:
        """
        Recursively remove additionalProperties from schema to comply with OpenAI strict mode.
        """
        if not isinstance(schema, dict):
            return schema
        
        # Create a copy to avoid modifying the original
        schema = dict(schema)
        
        # Remove additionalProperties at this level
        schema.pop("additionalProperties", None)
        
        # Recursively process nested objects
        if "properties" in schema and isinstance(schema["properties"], dict):
            schema["properties"] = {
                key: self._remove_additional_properties(value)
                for key, value in schema["properties"].items()
            }
        
        # Handle arrays
        if "items" in schema:
            schema["items"] = self._remove_additional_properties(schema["items"])
        
        # Handle anyOf, oneOf, allOf
        for key in ["anyOf", "oneOf", "allOf"]:
            if key in schema and isinstance(schema[key], list):
                schema[key] = [
                    self._remove_additional_properties(sub_schema)
                    for sub_schema in schema[key]
                ]
        
        return schema

    def to_oai_tool(self) -> dict:
        # Get the input schema and ensure it's OpenAI-compatible
        parameters = self.tool.inputSchema or {"type": "object", "properties": {}}
        
        # Deep copy to avoid modifying the original
        parameters = copy.deepcopy(parameters)
        
        # Remove additionalProperties to comply with OpenAI strict schema
        parameters = self._remove_additional_properties(parameters)
        
        return {
            "type": "function",
            "function": {
                "name": self.__name__,
                "description": self.__doc__ or "",
                "parameters": parameters,
            },
        }
