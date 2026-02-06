# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Echo Environment Implementation.

A pure MCP environment that echoes back messages sent to it.
This demonstrates how to build an MCPEnvironment with inline FastMCP tools.

All interactions happen through MCP tools:
- `echo_message(message)`: Echo back the provided message
- `echo_with_length(message)`: Echo back the message with its length

Example:
    >>> from openenv.core.env_server.mcp_types import ListToolsAction, CallToolAction
    >>> env = EchoEnvironment()
    >>> env.reset()
    >>>
    >>> # List available tools
    >>> obs = env.step(ListToolsAction())
    >>> print([t.name for t in obs.tools])  # ["echo_message", "echo_with_length"]
    >>>
    >>> # Call a tool
    >>> obs = env.step(CallToolAction(tool_name="echo_message", arguments={"message": "Hi!"}))
    >>> print(obs.result)  # "Hi!"
"""

from typing import Any, Optional
from uuid import uuid4

# Support both in-repo and standalone imports
try:
    # In-repo imports (when running from OpenEnv repository)
    from openenv.core.env_server.mcp_environment import MCPEnvironment
    from openenv.core.env_server.types import Action, Observation, State
except ImportError:
    # Standalone imports (when environment is standalone with openenv from pip)
    from openenv.core.env_server.mcp_environment import MCPEnvironment
    from openenv.core.env_server.types import Action, Observation, State

from fastmcp import FastMCP


class EchoEnvironment(MCPEnvironment):
    """
    A pure MCP echo environment that echoes back messages.

    This environment exposes all functionality through MCP tools:
    - `echo_message`: Echo back the provided message
    - `echo_with_length`: Echo back the message with its length

    The environment inherits MCP support (ListToolsAction, CallToolAction)
    from the MCPEnvironment base class. No legacy action types are supported.

    Example using MCPToolClient:
        >>> from openenv.core.mcp_client import MCPToolClient
        >>>
        >>> with MCPToolClient(base_url="http://localhost:8000") as env:
        ...     env.reset()
        ...     tools = env.list_tools()
        ...     print([t.name for t in tools])
        ...     result = env.call_tool("echo_message", message="Hello!")
        ...     print(result)
    """

    def __init__(self):
        """Initialize the echo environment with MCP server and tools."""
        # Create MCP server and define tools inline
        mcp = FastMCP("echo_env")

        @mcp.tool
        def echo_message(message: str) -> str:
            """
            Echo back the provided message.

            Args:
                message: The message to echo back

            Returns:
                The same message that was provided
            """
            return message

        @mcp.tool
        def echo_with_length(message: str) -> dict:
            """
            Echo back the message with its length.

            Args:
                message: The message to echo back

            Returns:
                Dictionary with the message and its length
            """
            return {"message": message, "length": len(message)}

        # Pass the MCP server to the base class
        super().__init__(mcp)
        self._state = State(episode_id=str(uuid4()), step_count=0)
        self._reset_count = 0

    def reset(
        self,
        seed: Optional[int] = None,
        episode_id: Optional[str] = None,
        **kwargs: Any,
    ) -> Observation:
        """
        Reset the environment.

        Args:
            seed: Optional random seed (unused in echo env)
            episode_id: Optional episode ID to use
            **kwargs: Additional reset options

        Returns:
            Observation indicating the environment is ready
        """
        self._state = State(
            episode_id=episode_id or str(uuid4()),
            step_count=0,
        )
        self._reset_count += 1

        return Observation(
            done=False,
            reward=0.0,
            metadata={"status": "ready", "message": "Echo environment ready!"},
        )

    def _step_impl(
        self,
        action: Action,
        timeout_s: Optional[float] = None,
        **kwargs: Any,
    ) -> Observation:
        """
        Handle non-MCP actions.

        This environment only supports MCP actions (ListToolsAction, CallToolAction).
        Any other action type returns an error observation.

        Args:
            action: The action to execute
            timeout_s: Optional timeout (unused)
            **kwargs: Additional arguments

        Returns:
            Observation with error for unknown action types
        """
        return Observation(
            done=False,
            reward=0.0,
            metadata={
                "error": f"Unknown action type: {type(action).__name__}. "
                "Use ListToolsAction or CallToolAction for MCP interactions."
            },
        )

    def step(
        self,
        action: Action,
        timeout_s: Optional[float] = None,
        **kwargs: Any,
    ) -> Observation:
        """
        Execute a step in the environment.

        Delegates to base class for MCP actions. Increments step count for all actions.

        Args:
            action: The MCP action to execute (ListToolsAction or CallToolAction)
            timeout_s: Optional timeout for the action
            **kwargs: Additional arguments

        Returns:
            Observation from the action execution
        """
        # Increment step count for all actions
        self._state.step_count += 1

        # Let the base class handle MCP actions and non-MCP routing
        return super().step(action, timeout_s=timeout_s, **kwargs)

    @property
    def state(self) -> State:
        """
        Get the current environment state.

        Returns:
            Current State with episode_id and step_count
        """
        return self._state
