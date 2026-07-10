"""Dialect wire-parsing contracts (unit-level, no model)."""

from verifiers.v1.dialects import parse_tools


def test_parse_tools_empty_and_non_function_are_none():
    # The tools contract shared by all three dialects: None when nothing parses — never [].
    # An empty parse must not be truthy-distinguishable from "no tools", or a request whose
    # tools array holds only non-function entries would clear `Trace.tool_defs`.
    assert parse_tools(None) is None
    assert parse_tools([]) is None
    assert parse_tools([{"type": "custom", "name": "grep", "format": {}}]) is None


def test_parse_tools_keeps_function_tools_and_skips_others():
    tools = parse_tools(
        [
            {"type": "custom", "name": "grep", "format": {}},
            {
                "type": "function",
                "function": {
                    "name": "search",
                    "description": "Search the corpus.",
                    "parameters": {"type": "object", "properties": {}},
                },
            },
        ]
    )
    assert tools is not None
    (tool,) = tools
    assert tool.name == "search"
    assert tool.strict is None
