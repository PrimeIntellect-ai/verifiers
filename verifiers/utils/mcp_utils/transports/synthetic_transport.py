"""
Synthetic MCP Transport for testing with fake backends.

Instead of connecting to real MCP servers (like Zapier/Airtable),
this transport intercepts tool calls and routes them to in-memory handlers.
"""

import json
from typing import Dict, Callable, Any, Optional
from dataclasses import dataclass, field
from mcp.types import Tool

from verifiers.utils.mcp_utils.transports.base import MCPTransport


class SyntheticTransport(MCPTransport):
    """
    A mock transport that intercepts tool calls and routes them to synthetic handlers.
    
    Usage:
        # Define your synthetic data
        data = {
            "candidates": [
                {"id": "1", "name": "Alice", "status": "active"},
                {"id": "2", "name": "Bob", "status": "interviewing"},
            ]
        }
        
        # Define handlers for each tool
        handlers = {
            "search_candidates": lambda data, args: json.dumps(data["candidates"]),
            "get_candidate": lambda data, args: json.dumps(
                next((c for c in data["candidates"] if c["id"] == args["id"]), None)
            ),
        }
        
        # Create transport
        transport = SyntheticTransport(
            tools=create_tool_definitions(),
            handlers=handlers,
            data=data
        )
    """
    
    def __init__(
        self,
        tools: Dict[str, Tool],
        handlers: Dict[str, Callable[[dict, dict], str]],
        data: Optional[dict] = None,
        name: str = "synthetic"
    ):
        """
        Args:
            tools: Dict mapping tool names to MCP Tool definitions
            handlers: Dict mapping tool names to handler functions.
                      Each handler receives (data, arguments) and returns a string.
            data: The synthetic data store that handlers can read/write
            name: Name for this transport (for logging)
        """
        self._tools = tools
        self.handlers = handlers
        self.data = data if data is not None else {}
        self.name = name
        self._connected = False
    
    @property
    def tools(self) -> Dict[str, Tool]:
        return self._tools
    
    async def connect(self) -> Dict[str, Tool]:
        """No-op connect - we're already 'connected' to our in-memory data."""
        self._connected = True
        return self._tools
    
    async def call_tool(self, tool_name: str, arguments: dict) -> str:
        """Route the tool call to the appropriate handler."""
        if not self._connected:
            raise RuntimeError(f"Transport '{self.name}' not connected")
        
        if tool_name not in self.handlers:
            raise ValueError(
                f"No handler registered for tool '{tool_name}'. "
                f"Available handlers: {list(self.handlers.keys())}"
            )
        
        handler = self.handlers[tool_name]
        
        # Call handler - it can be sync or async
        result = handler(self.data, arguments)
        
        # If handler is async, await it
        if hasattr(result, '__await__'):
            result = await result
        
        return str(result)
    
    async def disconnect(self) -> None:
        """No-op disconnect."""
        self._connected = False
    
    async def is_connected(self) -> bool:
        return self._connected


# =============================================================================
# Helper to create Tool definitions programmatically
# =============================================================================

def create_tool(
    name: str,
    description: str,
    parameters: Dict[str, Any],
    required: Optional[list] = None
) -> Tool:
    """
    Helper to create an MCP Tool definition.
    
    Args:
        name: Tool name
        description: Tool description
        parameters: Dict of parameter_name -> {"type": ..., "description": ...}
        required: List of required parameter names
    """
    properties = {}
    for param_name, param_spec in parameters.items():
        if isinstance(param_spec, str):
            # Simple type string
            properties[param_name] = {"type": param_spec}
        else:
            # Full spec dict
            properties[param_name] = param_spec
    
    input_schema = {
        "type": "object",
        "properties": properties,
    }
    if required:
        input_schema["required"] = required
    
    return Tool(
        name=name,
        description=description,
        inputSchema=input_schema
    )


# =============================================================================
# Example: Synthetic Airtable-like backend
# =============================================================================

class SyntheticAirtable:
    """
    A synthetic Airtable-like data store with common operations.
    
    Usage:
        airtable = SyntheticAirtable()
        
        # Add some tables with data
        airtable.add_table("Candidates", [
            {"id": "rec1", "Name": "Alice", "Status": "Active", "Role": "Engineer"},
            {"id": "rec2", "Name": "Bob", "Status": "Interviewing", "Role": "Designer"},
        ])
        
        # Get tools and handlers for use with SyntheticTransport
        tools = airtable.get_tools()
        handlers = airtable.get_handlers()
        
        transport = SyntheticTransport(tools, handlers, airtable.data)
    """
    
    def __init__(self):
        self.data: Dict[str, list] = {}  # table_name -> list of records
        self._record_counter = 0
    
    def add_table(self, table_name: str, records: list):
        """Add a table with initial records."""
        # Ensure each record has an id
        for record in records:
            if "id" not in record:
                self._record_counter += 1
                record["id"] = f"rec{self._record_counter}"
        self.data[table_name] = records
    
    def get_tools(self) -> Dict[str, Tool]:
        """Return MCP Tool definitions for Airtable-like operations."""
        return {
            "list_records": create_tool(
                name="list_records",
                description="List all records in a table",
                parameters={
                    "table_name": {
                        "type": "string",
                        "description": "Name of the table to list records from"
                    },
                    "max_records": {
                        "type": "integer",
                        "description": "Maximum number of records to return"
                    }
                },
                required=["table_name"]
            ),
            "search_records": create_tool(
                name="search_records",
                description="Search for records matching a query string",
                parameters={
                    "table_name": {
                        "type": "string",
                        "description": "Name of the table to search"
                    },
                    "query": {
                        "type": "string",
                        "description": "Search query - matches against all fields"
                    },
                    "field": {
                        "type": "string",
                        "description": "Optional: specific field to search in"
                    }
                },
                required=["table_name", "query"]
            ),
            "get_record": create_tool(
                name="get_record",
                description="Get a specific record by ID",
                parameters={
                    "table_name": {
                        "type": "string",
                        "description": "Name of the table"
                    },
                    "record_id": {
                        "type": "string",
                        "description": "ID of the record to retrieve"
                    }
                },
                required=["table_name", "record_id"]
            ),
            "create_record": create_tool(
                name="create_record",
                description="Create a new record in a table",
                parameters={
                    "table_name": {
                        "type": "string",
                        "description": "Name of the table"
                    },
                    "fields": {
                        "type": "object",
                        "description": "Record fields as key-value pairs"
                    }
                },
                required=["table_name", "fields"]
            ),
            "update_record": create_tool(
                name="update_record",
                description="Update an existing record",
                parameters={
                    "table_name": {
                        "type": "string",
                        "description": "Name of the table"
                    },
                    "record_id": {
                        "type": "string",
                        "description": "ID of the record to update"
                    },
                    "fields": {
                        "type": "object",
                        "description": "Fields to update as key-value pairs"
                    }
                },
                required=["table_name", "record_id", "fields"]
            ),
            "count_records": create_tool(
                name="count_records",
                description="Count records in a table, optionally matching a query",
                parameters={
                    "table_name": {
                        "type": "string",
                        "description": "Name of the table"
                    },
                    "query": {
                        "type": "string",
                        "description": "Optional: only count records matching this query"
                    }
                },
                required=["table_name"]
            ),
        }
    
    def get_handlers(self) -> Dict[str, Callable]:
        """Return handler functions for each tool."""
        return {
            "list_records": self._handle_list_records,
            "search_records": self._handle_search_records,
            "get_record": self._handle_get_record,
            "create_record": self._handle_create_record,
            "update_record": self._handle_update_record,
            "count_records": self._handle_count_records,
        }
    
    def _handle_list_records(self, data: dict, args: dict) -> str:
        table_name = args.get("table_name")
        max_records = args.get("max_records")
        
        if table_name not in data:
            return json.dumps({"error": f"Table '{table_name}' not found"})
        
        records = data[table_name]
        if max_records:
            records = records[:max_records]
        
        return json.dumps({"records": records, "total": len(data[table_name])})
    
    def _handle_search_records(self, data: dict, args: dict) -> str:
        table_name = args.get("table_name")
        query = args.get("query", "").lower()
        field = args.get("field")
        
        if table_name not in data:
            return json.dumps({"error": f"Table '{table_name}' not found"})
        
        results = []
        for record in data[table_name]:
            if field:
                # Search specific field
                value = str(record.get(field, "")).lower()
                if query in value:
                    results.append(record)
            else:
                # Search all fields
                for value in record.values():
                    if query in str(value).lower():
                        results.append(record)
                        break
        
        return json.dumps({"records": results, "count": len(results)})
    
    def _handle_get_record(self, data: dict, args: dict) -> str:
        table_name = args.get("table_name")
        record_id = args.get("record_id")
        
        if table_name not in data:
            return json.dumps({"error": f"Table '{table_name}' not found"})
        
        for record in data[table_name]:
            if record.get("id") == record_id:
                return json.dumps({"record": record})
        
        return json.dumps({"error": f"Record '{record_id}' not found"})
    
    def _handle_create_record(self, data: dict, args: dict) -> str:
        table_name = args.get("table_name")
        fields = args.get("fields", {})
        
        if table_name not in data:
            return json.dumps({"error": f"Table '{table_name}' not found"})
        
        self._record_counter += 1
        new_record = {"id": f"rec{self._record_counter}", **fields}
        data[table_name].append(new_record)
        
        return json.dumps({"record": new_record, "created": True})
    
    def _handle_update_record(self, data: dict, args: dict) -> str:
        table_name = args.get("table_name")
        record_id = args.get("record_id")
        fields = args.get("fields", {})
        
        if table_name not in data:
            return json.dumps({"error": f"Table '{table_name}' not found"})
        
        for record in data[table_name]:
            if record.get("id") == record_id:
                record.update(fields)
                return json.dumps({"record": record, "updated": True})
        
        return json.dumps({"error": f"Record '{record_id}' not found"})
    
    def _handle_count_records(self, data: dict, args: dict) -> str:
        table_name = args.get("table_name")
        query = args.get("query", "").lower() if args.get("query") else None
        
        if table_name not in data:
            return json.dumps({"error": f"Table '{table_name}' not found"})
        
        if not query:
            count = len(data[table_name])
        else:
            count = 0
            for record in data[table_name]:
                for value in record.values():
                    if query in str(value).lower():
                        count += 1
                        break
        
        return json.dumps({"count": count})
