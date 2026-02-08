# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Echo Environment - A pure MCP environment for testing and demonstration.

This environment exposes all functionality through MCP tools:
- `echo_message(message)`: Echo back the provided message
- `echo_with_length(message)`: Echo back the message with its length

Example:
    >>> from echo_env import EchoEnv
    >>>
    >>> with EchoEnv(base_url="http://localhost:8000") as env:
    ...     env.reset()
    ...     tools = env.list_tools()
    ...     result = env.call_tool("echo_message", message="Hello!")
    ...     print(result)  # "Hello!"
"""

from .client import EchoEnv

# Re-export MCP types for convenience
from openenv.core.env_server.mcp_types import CallToolAction, ListToolsAction

__all__ = ["EchoEnv", "CallToolAction", "ListToolsAction"]
