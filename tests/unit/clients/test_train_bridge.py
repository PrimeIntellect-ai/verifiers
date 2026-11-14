"""The train client hands the renderer bridge the tool call ids of the assistant turn it
extends, in call order."""

from verifiers.v1.clients.train import _tool_call_ids


def test_tool_call_ids_follow_call_order():
    message = {
        "role": "assistant",
        "tool_calls": [
            {
                "id": "b",
                "type": "function",
                "function": {"name": "f", "arguments": "{}"},
            },
            {
                "id": "a",
                "type": "function",
                "function": {"name": "g", "arguments": "{}"},
            },
        ],
    }
    assert _tool_call_ids(message) == ["b", "a"]
    assert _tool_call_ids({"role": "assistant", "content": "done"}) == []
