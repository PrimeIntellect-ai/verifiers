from types import SimpleNamespace

from renderers.base import ParsedToolCall

from verifiers.v1.dialects.chat import message_to_wire, parse_message

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


def test_tool_calls_keep_the_header_content_type_the_model_sampled():
    # gpt-oss-20b sometimes samples ` code` for JSON arguments; a re-render must write that header back.
    sampled = '{"command":"ls -R"}'
    call = SimpleNamespace(raw=sampled, name="bash", arguments={"command": "ls -R"}, arguments_text=sampled,
                           content_type="code", id=None, status=None)
    (tc,) = response_from_generate({"finish_reason": "tool_calls", "tool_calls": [call]},
                                   "openai/gpt-oss-20b").message.tool_calls
    assert tc.content_type == "code"
    message = response_from_generate({"finish_reason": "tool_calls", "tool_calls": [call]},
                                      "openai/gpt-oss-20b").message
    wire = message_to_wire(message)
    assert wire["tool_calls"][0]["function"]["content_type"] == "code"
    assert parse_message(wire).tool_calls[0].content_type == "code"
    # A call without one (other formats, or older renderers) writes none.
    plain = SimpleNamespace(raw=sampled, name="bash", arguments={}, arguments_text=sampled, id=None, status=None)
    message = response_from_generate({"finish_reason": "tool_calls", "tool_calls": [plain]}, "m").message
    assert "content_type" not in message_to_wire(message)["tool_calls"][0]["function"]
