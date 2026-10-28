from renderers.base import ParsedToolCall

from verifiers.v1.clients.train import response_from_generate


def test_tool_call_arguments_keep_the_text_the_model_wrote():
    # gpt-oss mostly writes compact JSON; json.dumps would add spaces it never sampled.
    sampled = '{"path":"notes.md","limit":20}'
    result = {
        "finish_reason": "tool_calls",
        "tool_calls": [
            ParsedToolCall(
                raw=sampled,
                name="read",
                arguments={"path": "notes.md", "limit": 20},
                arguments_text=sampled,
            ),
            # Formats without one argument span (raw is the whole block) are re-dumped.
            ParsedToolCall(
                raw='<tool_call>{"name":"ls","arguments":{"path":"."}}</tool_call>',
                name="ls",
                arguments={"path": "."},
            ),
        ],
    }
    calls = response_from_generate(result, "openai/gpt-oss-20b").message.tool_calls
    assert [c.arguments for c in calls] == [sampled, '{"path": "."}']
