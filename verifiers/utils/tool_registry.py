"""
Tool Registry for Environment-Specific Tool Management

This module provides a centralized registry system for managing tools across different
environments. Tools are registered by environment ID and can be retrieved by name.

Core Functions:
    register_tool: Decorator to register tools in the registry
    get_tool: Retrieve a single tool by environment ID and tool name
    get_tools: Retrieve multiple tools by environment ID and tool names
    validate_tools: Validate that all specified tools are registered
    list_tools: List all available tools for an environment

Example:
    @register_tool("tool-test", "tool_A")
    async def tool_A(x: int) -> int:
        return x + 1

    tools = get_tools("tool-test", ["tool_A", "tool_B"])
"""

import logging
import threading
from collections import defaultdict
from typing import Callable

# Global tool registry: {env_id: {tool_name: tool_function}}
_tool_registry: dict[str, dict[str, Callable]] = defaultdict(dict)

# Thread-safe lock for registry operations
_registry_lock = threading.RLock()

logger = logging.getLogger("verifiers.utils.tool_registry")


def register_tool(env_id: str, tool_name: str):
    """
    Decorator to register a tool function in the global registry.

    Args:
        env_id: Environment identifier (e.g., "tool-test", "wiki-search")
        tool_name: Name to register the tool under (typically function.__name__)

    Returns:
        Decorator function that registers the tool and returns it unchanged

    Example:
        @register_tool("tool-test", "my_tool")
        async def my_tool(x: int) -> int:
            return x + 1
    """
    def decorator(tool_func: Callable) -> Callable:
        with _registry_lock:
            _tool_registry[env_id][tool_name] = tool_func
            logger.debug(
                f"Registered tool '{tool_name}' for environment '{env_id}': "
                f"{tool_func.__module__}.{tool_func.__name__}"
            )
        return tool_func

    return decorator


def get_tool(env_id: str, tool_name: str) -> Callable:
    """
    Retrieve a single tool function from the registry.

    Args:
        env_id: Environment identifier
        tool_name: Name of the tool to retrieve

    Returns:
        The tool function

    Raises:
        KeyError: If environment or tool not found

    Example:
        tool = get_tool("tool-test", "tool_A")
    """
    with _registry_lock:
        if env_id not in _tool_registry:
            available_envs = sorted(_tool_registry.keys())
            raise KeyError(
                f"Environment '{env_id}' not found in tool registry. "
                f"Available environments: {available_envs}"
            )

        if tool_name not in _tool_registry[env_id]:
            available_tools = sorted(_tool_registry[env_id].keys())
            raise KeyError(
                f"Tool '{tool_name}' not found for environment '{env_id}'. "
                f"Available tools: {available_tools}"
            )

        return _tool_registry[env_id][tool_name]


def get_tools(env_id: str, tool_names: list[str]) -> list[Callable]:
    """
    Retrieve multiple tool functions from the registry.

    Args:
        env_id: Environment identifier
        tool_names: List of tool names to retrieve

    Returns:
        List of tool functions in the same order as tool_names

    Raises:
        KeyError: If any tool not found (includes list of available tools)

    Example:
        tools = get_tools("tool-test", ["tool_A", "tool_B", "tool_C"])
    """
    with _registry_lock:
        if env_id not in _tool_registry:
            available_envs = sorted(_tool_registry.keys())
            raise KeyError(
                f"Environment '{env_id}' not found in tool registry. "
                f"Available environments: {available_envs}"
            )

        tools = []
        missing_tools = []

        for tool_name in tool_names:
            if tool_name not in _tool_registry[env_id]:
                missing_tools.append(tool_name)
            else:
                tools.append(_tool_registry[env_id][tool_name])

        if missing_tools:
            available_tools = sorted(_tool_registry[env_id].keys())
            raise KeyError(
                f"Tools {missing_tools} not found for environment '{env_id}'. "
                f"Available tools: {available_tools}"
            )

        return tools


def validate_tools(env_id: str, tool_names: list[str]) -> None:
    """
    Validate that all specified tools are registered for the environment.

    Args:
        env_id: Environment identifier
        tool_names: List of tool names to validate

    Raises:
        ValueError: If any tool is not registered (includes helpful context)

    Example:
        try:
            validate_tools("tool-test", ["tool_A", "tool_B"])
        except ValueError as e:
            print(f"Invalid tools: {e}")
    """
    with _registry_lock:
        if env_id not in _tool_registry:
            available_envs = sorted(_tool_registry.keys())
            raise ValueError(
                f"Environment '{env_id}' not found in tool registry. "
                f"Available environments: {available_envs}"
            )

        missing_tools = [
            tool_name for tool_name in tool_names if tool_name not in _tool_registry[env_id]
        ]

        if missing_tools:
            available_tools = sorted(_tool_registry[env_id].keys())
            raise ValueError(
                f"Unregistered tools found: {missing_tools}. "
                f"Available tools for '{env_id}': {available_tools}"
            )


def list_tools(env_id: str) -> list[str]:
    """
    List all available tool names for a specific environment.

    Args:
        env_id: Environment identifier

    Returns:
        Sorted list of tool names registered for this environment

    Raises:
        KeyError: If environment not found in registry

    Example:
        tools = list_tools("tool-test")
        # Returns: ["tool_A", "tool_B", "tool_C", "tool_D"]
    """
    with _registry_lock:
        if env_id not in _tool_registry:
            available_envs = sorted(_tool_registry.keys())
            raise KeyError(
                f"Environment '{env_id}' not found in tool registry. "
                f"Available environments: {available_envs}"
            )

        return sorted(_tool_registry[env_id].keys())


def list_environments() -> list[str]:
    """
    List all environment IDs that have registered tools.

    Returns:
        Sorted list of environment IDs

    Example:
        envs = list_environments()
        # Returns: ["tool-test", "wiki-search", ...]
    """
    with _registry_lock:
        return sorted(_tool_registry.keys())


def clear_registry() -> None:
    """
    Clear all tools from the registry.

    This is primarily useful for testing to ensure a clean state between tests.

    Example:
        clear_registry()
    """
    with _registry_lock:
        _tool_registry.clear()
        logger.debug("Tool registry cleared")
