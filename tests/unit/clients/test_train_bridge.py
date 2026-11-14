"""The train client hands the issuing assistant's tool call ids only to bridges that take
them, so parallel tool results can extend a turn without a full re-render."""

from verifiers.v1.clients.train import _bridge_takes_tool_call_ids, _tool_call_ids


class OrderedBridgeRenderer:
    def bridge_to_next_turn(
        self,
        previous_prompt_ids,
        previous_completion_ids,
        new_messages,
        *,
        tools=None,
        previous_tool_call_ids=None,
    ):
        return None


class PlainBridgeRenderer:
    def bridge_to_next_turn(
        self, previous_prompt_ids, previous_completion_ids, new_messages, *, tools=None
    ):
        return None


def test_only_bridges_that_take_call_ids_get_them():
    assert _bridge_takes_tool_call_ids(OrderedBridgeRenderer)
    assert not _bridge_takes_tool_call_ids(PlainBridgeRenderer)


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
