from renderers.base import ParsedToolCall

from verifiers.v1.clients.train import response_from_generate


def test_tool_calls_without_ids_get_ids_unique_across_turns():
    def tool_call_ids(*calls: ParsedToolCall) -> list[str]:
        usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
        result = {"tool_calls": list(calls), "usage": usage}
        response = response_from_generate(result, model="m")
        return [call.id for call in response.message.tool_calls or []]

    first_turn = tool_call_ids(
        ParsedToolCall(raw="", name="click"), ParsedToolCall(raw="", name="type")
    )
    second_turn = tool_call_ids(ParsedToolCall(raw="", name="click"))

    assert len(set(first_turn + second_turn)) == 3
