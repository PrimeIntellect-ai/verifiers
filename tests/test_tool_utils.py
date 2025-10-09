"""Tests for the tool_utils module."""

from typing import Optional

from verifiers.types import Message
from verifiers.envs.stateful_tool_env import StatefulToolEnv
from verifiers.utils.tool_utils import build_schema_only_tool, convert_func_to_oai_tool


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
                "strict": True,
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
                "strict": True,
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
                "strict": True,
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
                "strict": True,
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
                "strict": True,
            },
        }

    def test_build_schema_only_tool_skips_complex_args(self):
        """Schema-only tools should drop skipped non-pydantic params from the generated schema."""

        def complex_tool(
            user_id: str,
            msg: Message,
            payload: list[dict[str, str]],
            env: StatefulToolEnv,
        ) -> str:
            return "ok"

        schema_stub = build_schema_only_tool(
            complex_tool,
            args_to_skip=["msg", "env"],
        )
        stub_schema = convert_func_to_oai_tool(schema_stub)
        parameters = stub_schema["function"]["parameters"]
        properties = parameters["properties"]
        required = parameters.get("required", [])

        assert set(properties) == {"user_id", "payload"}
        assert "msg" not in properties
        assert "env" not in properties
        assert "msg" not in required
        assert "env" not in required

    def test_add_tool_to_stateful_tool_env(
        self, mock_stateful_tool_env: StatefulToolEnv
    ):
        def complex_tool(
            user_id: str,
            msg: Message,
            payload: list[dict[str, str]],
            env: StatefulToolEnv,
        ) -> str:
            return "ok"

        env = mock_stateful_tool_env
        original_tool_count = len(env.tools)
        env.add_tool(complex_tool, args_to_skip=["msg", "env"])

        assert len(env.tools) == original_tool_count + 1
        assert len(env.oai_tools) == original_tool_count + 1

        oai_tool = env.oai_tools[-1]
        props = oai_tool["function"]["parameters"]["properties"]
        required = oai_tool["function"]["parameters"].get("required", [])
        assert set(props) == {"user_id", "payload"}
        assert "msg" not in props
        assert "env" not in props
        assert "msg" not in required
        assert "env" not in required
        assert env.skipped_args["complex_tool"] == ["msg", "env"]
