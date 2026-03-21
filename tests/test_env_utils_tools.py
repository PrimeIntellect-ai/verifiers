"""
Integration tests for env_utils tool resolution functionality

Tests cover:
- Loading environment with string tools (resolved via registry)
- Loading environment with callable tools (passed through)
- Loading environment with no tools (backward compatibility)
- Error handling for mixed tool types
"""

import pytest

from verifiers.utils.env_utils import load_environment
from verifiers.utils.tool_registry import register_tool


@pytest.fixture(autouse=True)
def clear_registry():
    """Clear the registry before and after each test."""
    from verifiers.utils import tool_registry

    # Clear before test
    tool_registry._tool_registry.clear()
    yield
    # Clear after test
    tool_registry._tool_registry.clear()


def test_load_environment_with_string_tools(clear_registry):
    """Test loading environment with string tool names (registry resolution)."""

    # Register test tools
    @register_tool("tool-test", "test_tool_a")
    async def test_tool_a(x: int) -> int:
        return x + 1

    @register_tool("tool-test", "test_tool_b")
    async def test_tool_b(x: str) -> str:
        return x + "suffix"

    # Load environment with string tools
    env = load_environment("tool-test", tools=["test_tool_a", "test_tool_b"])

    # Verify tools were resolved and attached
    assert hasattr(env, "tools")
    assert len(env.tools) == 2
    assert test_tool_a in env.tools
    assert test_tool_b in env.tools


def test_load_environment_with_callable_tools(clear_registry):
    """Test loading environment with callable tools (direct pass-through)."""

    # Define test tools
    async def direct_tool_a(x: int) -> int:
        return x + 1

    async def direct_tool_b(x: str) -> str:
        return x + "suffix"

    # Load environment with callable tools
    env = load_environment(
        "tool-test", tools=[direct_tool_a, direct_tool_b]
    )

    # Verify tools were passed through
    assert hasattr(env, "tools")
    assert len(env.tools) == 2
    assert direct_tool_a in env.tools
    assert direct_tool_b in env.tools


def test_load_environment_no_tools(clear_registry):
    """Test loading environment without tools parameter (backward compatibility)."""

    # Load environment without tools parameter
    env = load_environment("tool-test")

    # Verify environment loaded with default tools
    assert hasattr(env, "tools")
    # tool-test environment has 4 default tools
    assert len(env.tools) == 4


def test_load_environment_empty_tool_list(clear_registry):
    """Test loading environment with empty tool list."""

    # Load environment with empty tools list
    env = load_environment("tool-test", tools=[])

    # Verify environment has no tools
    assert hasattr(env, "tools")
    assert len(env.tools) == 0


def test_mixed_tool_types_error(clear_registry):
    """Test that mixing Callable and str tools raises TypeError."""

    # Define a callable tool
    async def my_tool(x: int) -> int:
        return x + 1

    # Register a string tool
    @register_tool("tool-test", "registered_tool")
    async def registered_tool() -> str:
        return "registered"

    # Attempt to load with mixed types - should raise TypeError
    with pytest.raises(TypeError, match="tools must be all Callable or all str"):
        load_environment("tool-test", tools=[my_tool, "registered_tool"])

    with pytest.raises(TypeError, match="tools must be all Callable or all str"):
        load_environment("tool-test", tools=["registered_tool", my_tool])


def test_invalid_tool_name_in_registry(clear_registry):
    """Test that unregistered tool name raises KeyError from registry."""

    # Register one tool so environment exists in registry
    @register_tool("tool-test", "valid_tool")
    async def valid_tool(x: int) -> int:
        return x + 1

    # Try to load with a different, unregistered tool name
    with pytest.raises(KeyError, match=r"Tools \['nonexistent_tool'\] not found"):
        load_environment("tool-test", tools=["valid_tool", "nonexistent_tool"])


def test_invalid_tool_type_error(clear_registry):
    """Test that invalid tool type raises TypeError."""

    # Load with invalid tool type (int, not Callable or str)
    with pytest.raises(TypeError, match="tools must be list of Callable or list of str"):
        load_environment("tool-test", tools=[123, 456])


def test_environment_with_other_args(clear_registry):
    """Test that tools parameter works alongside other environment arguments."""

    # Register a tool
    @register_tool("tool-test", "custom_tool")
    async def custom_tool() -> str:
        return "custom"

    # Load environment with tools and other args
    env = load_environment(
        "tool-test",
        tools=["custom_tool"],
        num_train_examples=50,
        num_eval_examples=10,
    )

    # Verify both tools and other args were applied
    assert hasattr(env, "tools")
    assert len(env.tools) >= 1
    assert custom_tool in env.tools
    # num_train_examples should affect dataset size
    # (actual value depends on tool-test env implementation)


def test_single_string_tool(clear_registry):
    """Test loading environment with single string tool."""

    @register_tool("tool-test", "single_tool")
    async def single_tool(x: int) -> int:
        return x * 2

    env = load_environment("tool-test", tools=["single_tool"])

    assert hasattr(env, "tools")
    assert single_tool in env.tools


def test_single_callable_tool(clear_registry):
    """Test loading environment with single callable tool."""

    async def my_tool() -> str:
        return "result"

    env = load_environment("tool-test", tools=[my_tool])

    assert hasattr(env, "tools")
    assert my_tool in env.tools
