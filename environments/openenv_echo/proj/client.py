# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Echo Environment Client.

This module provides the client for connecting to an Echo Environment server.
EchoEnv extends MCPToolClient to provide tool-calling style interactions.

Example:
    >>> with EchoEnv(base_url="http://localhost:8000") as env:
    ...     env.reset()
    ...
    ...     # Discover tools
    ...     tools = env.list_tools()
    ...     print([t.name for t in tools])  # ['echo_message', 'echo_with_length']
    ...
    ...     # Call tools
    ...     result = env.call_tool("echo_message", message="Hello!")
    ...     print(result)  # "Hello!"
    ...
    ...     result = env.call_tool("echo_with_length", message="Test")
    ...     print(result)  # {"message": "Test", "length": 4}
"""

from openenv.core.mcp_client import MCPToolClient


class EchoEnv(MCPToolClient):
    """
    Client for the Echo Environment.

    This client provides a simple interface for interacting with the Echo
    Environment via MCP tools. It inherits all functionality from MCPToolClient:
    - `list_tools()`: Discover available tools
    - `call_tool(name, **kwargs)`: Call a tool by name
    - `reset(**kwargs)`: Reset the environment
    - `step(action)`: Execute an action (for advanced use)

    Example:
        >>> # Connect to a running server
        >>> with EchoEnv(base_url="http://localhost:8000") as env:
        ...     env.reset()
        ...
        ...     # List available tools
        ...     tools = env.list_tools()
        ...     for tool in tools:
        ...         print(f"{tool.name}: {tool.description}")
        ...
        ...     # Echo a message
        ...     result = env.call_tool("echo_message", message="Hello!")
        ...     print(result)  # "Hello!"
        ...
        ...     # Echo with length
        ...     result = env.call_tool("echo_with_length", message="Test")
        ...     print(result)  # {"message": "Test", "length": 4}

    Example with Docker:
        >>> # Automatically start container and connect
        >>> env = EchoEnv.from_docker_image("echo-env:latest")
        >>> try:
        ...     env.reset()
        ...     tools = env.list_tools()
        ...     result = env.call_tool("echo_message", message="Hello!")
        ... finally:
        ...     env.close()

    Example with HuggingFace Space:
        >>> # Run from HuggingFace Space
        >>> env = EchoEnv.from_env("openenv/echo-env")
        >>> try:
        ...     env.reset()
        ...     result = env.call_tool("echo_message", message="Hello!")
        ... finally:
        ...     env.close()
    """

    pass  # MCPToolClient provides all needed functionality
