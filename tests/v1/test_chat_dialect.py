import json

import pytest

from verifiers.v1.dialects.chat import (
    alias_chat_tool_names,
    restore_chat_sse_tool_names,
)


def test_alias_chat_tool_names_uses_explicit_mapping():
    internal = "mcp__verifiers-bare-tools__submit_now"
    body = {
        "tools": [{"type": "function", "function": {"name": internal}}],
        "messages": [],
    }

    aliased, reverse = alias_chat_tool_names(body, {internal: "submit__now"})

    assert body["tools"][0]["function"]["name"] == internal
    assert aliased["tools"][0]["function"]["name"] == "submit__now"
    assert reverse == {"submit__now": internal}


@pytest.mark.parametrize("terminator", [b"\n\n", b"\r\n\r\n", b"\r\r"])
def test_restore_chat_sse_tool_names_preserves_framing(terminator):
    event = {
        "choices": [
            {
                "delta": {
                    "tool_calls": [
                        {
                            "function": {
                                "name": "submit__now",
                                "arguments": '{"name":"submit__now"}',
                            }
                        }
                    ]
                }
            }
        ]
    }
    raw = b"data: " + json.dumps(event).encode() + terminator

    restored = restore_chat_sse_tool_names(
        raw, {"submit__now": "mcp__verifiers-bare-tools__submit_now"}
    )

    assert restored.endswith(terminator)
    payload = restored.removesuffix(terminator).removeprefix(b"data: ")
    function = json.loads(payload)["choices"][0]["delta"]["tool_calls"][0]["function"]
    assert function == {
        "name": "mcp__verifiers-bare-tools__submit_now",
        "arguments": '{"name":"submit__now"}',
    }
