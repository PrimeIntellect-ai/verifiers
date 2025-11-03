"""Tests for the tool_utils module."""

from typing import Optional

from verifiers.utils.tool_utils import convert_func_to_oai_tool


class TestToolUtils:
    """Test cases for the tool_utils module."""

    def test_convert_func_to_oai_tool(self):
        """Test the convert_func_to_oai_tool function with a description."""

        def test_func(param1: int, param2: str, param3: bool):
            # google style docstring
            """This is a test function.

            Args:
                param1: This is test integer parameter.
                param2: This is test string parameter.
                param3: This is test boolean parameter.

            Returns:
                This is test return value.
            """
            return 1.0

        result = convert_func_to_oai_tool(test_func)
        assert result == {
            "type": "function",
            "function": {
                "name": "test_func",
                "description": "This is a test function.",
                "parameters": {
                    "type": "object",
                    "additionalProperties": False,
                    "properties": {
                        "param1": {
                            "type": "integer",
                            "description": "This is test integer parameter.",
                            "title": "Param1",
                        },
                        "param2": {
                            "type": "string",
                            "description": "This is test string parameter.",
                            "title": "Param2",
                        },
                        "param3": {
                            "type": "boolean",
                            "description": "This is test boolean parameter.",
                            "title": "Param3",
                        },
                    },
                    "required": ["param1", "param2", "param3"],
                    "title": "test_func_args",
                },
            },
        }

    def test_convert_func_to_oai_tool_with_default_values(self):
        """Test the convert_func_to_oai_tool function with default values."""

        def test_func(param1: int, param2: str = "test", param3: bool = True):
            return 1.0

        result = convert_func_to_oai_tool(test_func)
        assert result == {
            "type": "function",
            "function": {
                "name": "test_func",
                "description": "",
                "parameters": {
                    "type": "object",
                    "additionalProperties": False,
                    "properties": {
                        "param1": {"type": "integer", "title": "Param1"},
                        "param2": {
                            "type": "string",
                            "title": "Param2",
                            "default": "test",
                        },
                        "param3": {
                            "type": "boolean",
                            "title": "Param3",
                            "default": True,
                        },
                    },
                    "required": ["param1", "param2", "param3"],
                    "title": "test_func_args",
                },
            },
        }

    def test_convert_func_to_oai_tool_with_optional_values(self):
        """Test the convert_func_to_oai_tool function with optional values."""

        def test_func(param1: int, param2: str, param3: Optional[bool] = True):
            return None

        result = convert_func_to_oai_tool(test_func)
        assert result == {
            "type": "function",
            "function": {
                "name": "test_func",
                "description": "",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "param1": {"type": "integer", "title": "Param1"},
                        "param2": {"type": "string", "title": "Param2"},
                        "param3": {
                            "default": True,
                            "title": "Param3",
                            "anyOf": [{"type": "boolean"}, {"type": "null"}],
                        },
                    },
                    "required": ["param1", "param2", "param3"],
                    "title": "test_func_args",
                    "additionalProperties": False,
                },
            },
        }

    def test_convert_func_to_oai_tool_with_list_type_hint(self):
        """Test the convert_func_to_oai_tool function with list type hint."""

        def test_func(param1: list[int]):
            return None

        result = convert_func_to_oai_tool(test_func)
        assert result == {
            "type": "function",
            "function": {
                "name": "test_func",
                "description": "",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "param1": {
                            "type": "array",
                            "items": {"type": "integer"},
                            "title": "Param1",
                        },
                    },
                    "required": ["param1"],
                    "title": "test_func_args",
                    "additionalProperties": False,
                },
            },
        }

    def test_convert_func_to_oai_tool_without_type_hint(self):
        """Test the convert_func_to_oai_tool function without type hint."""

        def test_func(param1):
            """This is a test function."""
            return None

        result = convert_func_to_oai_tool(test_func)
        assert result == {
            "type": "function",
            "function": {
                "name": "test_func",
                "description": "This is a test function.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "param1": {"title": "Param1"},
                    },
                    "required": ["param1"],
                    "title": "test_func_args",
                    "additionalProperties": False,
                },
            },
        }
