"""Each dialect imposes the eval's sampling in its native request shape, replacing the
harness's own values, and parses the same shape back for the trace. For a Claude model whose
program set no prompt-cache breakpoint, it adds one on the last block of the copy it forwards."""

import copy

import pytest

from verifiers.v1.configs.sampling import SamplingConfig
from verifiers.v1.dialects import AnthropicDialect, ChatDialect, ResponsesDialect

SAMPLING = SamplingConfig(temperature=0.3, top_p=0.9, reasoning_effort="low")


@pytest.mark.parametrize(
    "dialect,body,effort",
    [
        (
            ChatDialect(),
            {"temperature": 1.0, "top_p": 1.0, "reasoning_effort": "high"},
            lambda body: body["reasoning_effort"],
        ),
        (
            ResponsesDialect(),
            {
                "temperature": 1.0,
                "top_p": 1.0,
                "reasoning": {"effort": "high", "summary": "detailed"},
            },
            lambda body: body["reasoning"]["effort"],
        ),
        (
            AnthropicDialect(),
            {"temperature": 1.0, "top_p": 1.0, "output_config": {"effort": "high"}},
            lambda body: body["output_config"]["effort"],
        ),
    ],
    ids=["chat", "responses", "anthropic"],
)
def test_apply_overrides_sampling(dialect, body, effort):
    sent = dialect.apply_overrides(body, "m", SAMPLING)
    assert (sent["temperature"], sent["top_p"], effort(sent)) == (0.3, 0.9, "low")
    parsed = dialect.parse_sampling(sent)
    assert (parsed.temperature, parsed.top_p, parsed.reasoning_effort) == (
        0.3,
        0.9,
        "low",
    )


def test_apply_overrides_keeps_other_reasoning_fields():
    body = {"reasoning": {"effort": "high", "summary": "detailed"}}
    sent = ResponsesDialect().apply_overrides(body, "m", SAMPLING)
    assert sent["reasoning"] == {"effort": "low", "summary": "detailed"}


CLAUDE = "anthropic/claude-sonnet-5.5"
MARKER = {"type": "ephemeral"}
CHAT = {
    "model": CLAUDE,
    "messages": [
        {"role": "system", "content": "You are terse."},
        {"role": "user", "content": [{"type": "text", "text": "List the files."}]},
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": "c1",
                    "type": "function",
                    "function": {"name": "ls", "arguments": "{}"},
                }
            ],
        },
        {
            "role": "tool",
            "tool_call_id": "c1",
            "content": [{"type": "text", "text": "a.txt"}],
        },
    ],
}
ANTHROPIC = {
    "model": CLAUDE,
    "system": [{"type": "text", "text": "You are terse."}],
    "messages": [
        {"role": "user", "content": "List the files."},
        {
            "role": "assistant",
            "content": [
                {"type": "thinking", "thinking": "", "signature": "s"},
                {"type": "tool_use", "id": "c1", "name": "ls", "input": {}},
            ],
        },
        {
            "role": "user",
            "content": [
                {"type": "tool_result", "tool_use_id": "c1", "content": "a.txt"}
            ],
        },
    ],
}
DIALECTS = pytest.mark.parametrize(
    "dialect,body",
    [(ChatDialect(), CHAT), (AnthropicDialect(), ANTHROPIC)],
    ids=["chat", "anthropic"],
)


def markers(value) -> list[dict]:
    """Every object in a native request that carries a `cache_control` key."""
    if isinstance(value, dict):
        own = [value] if "cache_control" in value else []
        return own + [found for item in value.values() for found in markers(item)]
    if isinstance(value, list):
        return [found for item in value for found in markers(item)]
    return []


def last_content(body: dict):
    return body["messages"][-1]["content"]


@DIALECTS
def test_prompt_cache_marks_the_last_block(dialect, body):
    program = copy.deepcopy(body)
    sent = dialect.apply_prompt_cache(body, CLAUDE, "auto")
    assert markers(sent) == [{**last_content(body)[-1], "cache_control": MARKER}]
    assert last_content(sent)[-1]["cache_control"] == MARKER
    assert body == program  # the program's request is a copy, not rewritten in place


@pytest.mark.parametrize(
    "dialect,message",
    [
        (ChatDialect(), {"role": "tool", "tool_call_id": "c1", "content": "a.txt"}),
        (AnthropicDialect(), {"role": "user", "content": "a.txt"}),
    ],
    ids=["chat", "anthropic"],
)
def test_prompt_cache_turns_string_content_into_a_text_block(dialect, message):
    body = {"model": CLAUDE, "messages": [message]}
    sent = dialect.apply_prompt_cache(body, CLAUDE, "auto")
    assert last_content(sent) == [
        {"type": "text", "text": "a.txt", "cache_control": MARKER}
    ]
    assert body["messages"][0]["content"] == "a.txt"


@DIALECTS
def test_prompt_cache_marks_the_last_block_that_can_carry_one(dialect, body):
    """Empty text cannot carry a marker, so the block before it gets it; a message with
    nothing markable, such as empty content, is left alone."""
    padded = {
        **body,
        "messages": [
            *body["messages"][:-1],
            {
                **body["messages"][-1],
                "content": [*last_content(body), {"type": "text", "text": ""}],
            },
        ],
    }
    sent = dialect.apply_prompt_cache(padded, CLAUDE, "auto")
    assert last_content(sent)[-2]["cache_control"] == MARKER
    assert len(markers(sent)) == 1
    empty = {"model": CLAUDE, "messages": [{"role": "user", "content": ""}]}
    assert dialect.apply_prompt_cache(empty, CLAUDE, "auto") is empty


MARKED_SYSTEM = {"type": "text", "text": "You are terse.", "cache_control": MARKER}


@pytest.mark.parametrize(
    "dialect,body",
    [
        (
            ChatDialect(),
            {
                **CHAT,
                "messages": [
                    {"role": "system", "content": [MARKED_SYSTEM]},
                    *CHAT["messages"][1:],
                ],
            },
        ),
        (AnthropicDialect(), {**ANTHROPIC, "system": [MARKED_SYSTEM]}),
        (AnthropicDialect(), {**ANTHROPIC, "cache_control": MARKER}),
    ],
    ids=["chat-marker", "anthropic-marker", "anthropic-automatic"],
)
def test_prompt_cache_keeps_the_programs_own_breakpoints(dialect, body):
    assert dialect.apply_prompt_cache(body, CLAUDE, "auto") is body


@DIALECTS
def test_prompt_cache_off_sends_the_request_unchanged(dialect, body):
    assert dialect.apply_prompt_cache(body, CLAUDE, "off") is body


def test_prompt_cache_is_claude_only():
    """Other providers cache prefixes without markers, so their requests stay unchanged;
    the Anthropic format carries `cache_control` for any model behind it."""
    for model in ("internal/glm-5.3-high-throughput", "openai/gpt-6.1-sol"):
        assert ChatDialect().apply_prompt_cache(CHAT, model, "auto") is CHAT
    responses = {"model": CLAUDE, "input": [{"role": "user", "content": "hi"}]}
    assert ResponsesDialect().apply_prompt_cache(responses, CLAUDE, "auto") is responses
    sent = AnthropicDialect().apply_prompt_cache(ANTHROPIC, "deepseek-v4.1", "auto")
    assert len(markers(sent)) == 1
