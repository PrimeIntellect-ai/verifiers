"""An `@intercept` AssistantMessage rewrite is committed as the re-parse of its serialized
wire body (see `interception.server._intercept`): the committed message must hash-match the
assistant message the dialect parses back out of the next request's replayed history, or the
graph prefix walk breaks and the turn is duplicated (see `graph.message_hash`)."""

import pytest

from verifiers.v1.dialects import DIALECTS
from verifiers.v1.dialects.anthropic import AnthropicDialect
from verifiers.v1.dialects.chat import ChatDialect
from verifiers.v1.dialects.responses import ResponsesDialect
from verifiers.v1.graph import message_hash
from verifiers.v1.types import AssistantMessage, Response, ToolCall, Usage

# A dialect-native `provider_state` an interceptor may carry over from the original turn.
# Responses has none: there `provider_state` is the whole replayed `output` item list.
PROVIDER_STATE = {
    ChatDialect: [{"type": "reasoning.text", "text": "hmm", "format": "razor"}],
    AnthropicDialect: [{"type": "thinking", "thinking": "hmm", "signature": "sig"}],
    ResponsesDialect: None,
}


def replay_request(dialect, raw: dict) -> dict:
    """The next harness request, restating the rewritten turn from its wire body."""
    if isinstance(dialect, ChatDialect):
        return {"messages": [raw["choices"][0]["message"]]}
    if isinstance(dialect, ResponsesDialect):
        return {"input": list(raw["output"])}
    return {"messages": [{"role": "assistant", "content": raw["content"]}]}


def rewrite_cases():
    for dialect in DIALECTS:
        rewrites = {
            "content": AssistantMessage(content="blocked by policy"),
            "tool-call": AssistantMessage(
                tool_calls=[
                    ToolCall(id="call_1", name="shell", arguments='{"cmd": "ls"}')
                ],
            ),
        }
        if (state := PROVIDER_STATE[type(dialect)]) is not None:
            rewrites["provider-state"] = AssistantMessage(
                content="kept native reasoning", provider_state=state
            )
        for name, rewrite in rewrites.items():
            for usage in (Usage(prompt_tokens=10, completion_tokens=5), None):
                label = f"{type(dialect).__name__}-{name}{'' if usage else '-no-usage'}"
                yield pytest.param(dialect, rewrite, usage, id=label)


@pytest.mark.parametrize("dialect,rewrite,usage", rewrite_cases())
def test_rewrite_roundtrips_to_replayed_prompt(dialect, rewrite, usage):
    response = Response(
        id="resp_orig",
        created=123,
        model="test-model",
        message=rewrite,
        finish_reason="tool_calls" if rewrite.tool_calls else "stop",
        usage=usage,
    )
    # what `_intercept` does with an AssistantMessage rewrite:
    raw = dialect.serialize_response(response)
    committed = dialect.parse_response(dialect.validate_response(raw)).message

    prompt, _ = dialect.parse_request(replay_request(dialect, raw))
    replayed = [m for m in prompt if isinstance(m, AssistantMessage)]
    assert len(replayed) == 1
    assert message_hash(committed) == message_hash(replayed[0])
