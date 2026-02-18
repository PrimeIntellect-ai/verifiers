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
    get_tool,
    get_tools,
    list_tools,
    list_environments,
    register_tool,
    validate_tools,
)