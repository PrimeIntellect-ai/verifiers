"""
Unit tests for tool_registry module

Tests cover:
- Tool registration and retrieval
- Batch tool retrieval
- Tool validation
- Listing tools and environments
- Error cases and edge conditions
"""

import pytest

from verifiers.utils.tool_registry import (
    clear_registry,
    get_tool,
    get_tools,
    list_tools,
    list_environments,
    register_tool,
    validate_tools,
)


@pytest.fixture(autouse=True)
def clear_registry_before_and_after_test():
    """Clear the registry before and after each test."""
    clear_registry()
    yield
    clear_registry()


def test_registration_and_retrieval():
    """Test registering a tool and retrieving it."""
    # Register a test tool
    @register_tool("test-env", "test_tool")
    async def test_tool(x: int) -> int:
        return x + 1

    # Retrieve the tool
    retrieved = get_tool("test-env", "test_tool")

    # Verify it's the same function
    assert retrieved == test_tool
    assert retrieved.__name__ == "test_tool"


def test_batch_retrieval():
    """Test retrieving multiple tools at once."""
    # Register multiple tools
    @register_tool("test-env", "tool_a")
    async def tool_a(x: int) -> int:
        return x + 1

    @register_tool("test-env", "tool_b")
    async def tool_b(x: str) -> str:
        return x + "suffix"

    # Retrieve both tools
    tools = get_tools("test-env", ["tool_a", "tool_b"])

    # Verify both were retrieved
    assert len(tools) == 2
    assert tool_a in tools
    assert tool_b in tools


def test_validation():
    """Test tool validation."""
    # Register tools
    @register_tool("test-env", "tool_a")
    async def tool_a(x: int) -> int:
        return x + 1

    @register_tool("test-env", "tool_b")
    async def tool_b(x: str) -> str:
        return x + "suffix"

    # Valid tools should pass validation
    validate_tools("test-env", ["tool_a", "tool_b"])  # Should not raise

    # Invalid tool should raise ValueError
    with pytest.raises(ValueError, match="Unregistered tools"):
        validate_tools("test-env", ["tool_a", "nonexistent_tool"])


def test_clear_registry():
    """Test clearing the registry."""
    # Register a tool
    @register_tool("test-env", "test_tool")
    async def test_tool(x: int) -> int:
        return x + 1

    # Verify it's registered
    assert get_tool("test-env", "test_tool") == test_tool

    # Clear registry
    clear_registry()

    # Verify it's gone
    with pytest.raises(KeyError):
        get_tool("test-env", "test_tool")


def test_multiple_environments():
    """Test that tools from different environments don't interfere."""
    # Register tools in different environments
    @register_tool("env-a", "shared_name")
    async def env_a_tool(x: int) -> int:
        return x + 1

    @register_tool("env-b", "shared_name")
    async def env_b_tool(x: int) -> int:
        return x + 2

    # Retrieve from each environment
    tool_from_a = get_tool("env-a", "shared_name")
    tool_from_b = get_tool("env-b", "shared_name")

    # Verify they're different functions
    assert tool_from_a == env_a_tool
    assert tool_from_b == env_b_tool
    assert tool_from_a != tool_from_b
    assert tool_from_a.__name__ == "shared_name"
    assert tool_from_b.__name__ == "shared_name"


def test_list_tools():
    """Test listing all tools in an environment."""
    # Register tools
    @register_tool("test-env", "tool_a")
    async def tool_a(x: int) -> int:
        return x + 1

    @register_tool("test-env", "tool_b")
    async def tool_b(x: str) -> str:
        return x + "suffix"

    # List tools
    tools = list_tools("test-env")

    # Verify both tools are listed
    assert len(tools) == 2
    assert "tool_a" in tools
    assert "tool_b" in tools


def test_list_environments():
    """Test listing all environments with registered tools."""
    # Register tools in different environments
    @register_tool("env-a", "tool_a")
    async def tool_a(x: int) -> int:
        return x + 1

    @register_tool("env-b", "tool_b")
    async def tool_b(x: str) -> str:
        return x + "suffix"

    # List environments
    envs = list_environments()

    # Verify both environments are listed
    assert len(envs) == 2
    assert "env-a" in envs
    assert "env-b" in envs


def test_error_get_nonexistent_tool():
    """Test error when retrieving a nonexistent tool."""
    with pytest.raises(KeyError, match="not found"):
        get_tool("test-env", "nonexistent_tool")


def test_error_get_tools_partial_match():
    """Test error when some tools don't exist."""
    # Register only one tool
    @register_tool("test-env", "tool_a")
    async def tool_a(x: int) -> int:
        return x + 1

    # Try to retrieve multiple tools where one doesn't exist
    with pytest.raises(KeyError, match="not found"):
        get_tools("test-env", ["tool_a", "nonexistent_tool"])
