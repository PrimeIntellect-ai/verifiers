"""Each dialect imposes the eval's sampling in its native request shape, replacing the
harness's own values, and parses the same shape back for the trace."""

import pytest

from verifiers.v1.configs.sampling import SamplingConfig
from verifiers.v1.dialects import AnthropicDialect, ChatDialect, ResponsesDialect
from verifiers.v1.dialects.base import add_prompt_cache

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


def test_add_prompt_cache_marks_the_last_block_that_can_carry_a_breakpoint():
    chat = {
        "messages": [
            {"role": "user", "content": "ls"},
            {"role": "tool", "content": "a.txt"},
        ]
    }
    sent = add_prompt_cache(chat, "anthropic/claude-sonnet-5.5")
    assert sent["messages"][1]["content"] == [
        {"type": "text", "text": "a.txt", "cache_control": {"type": "ephemeral"}}
    ]
    assert (
        chat["messages"][1]["content"] == "a.txt"
    )  # the program's request is not mutated
    result = {"type": "tool_result", "tool_use_id": "c1", "content": "a.txt"}
    thinking = {"type": "thinking", "thinking": "", "signature": "s"}
    sent = add_prompt_cache(
        {"messages": [{"role": "user", "content": [result, thinking]}]},
        "claude-sonnet-5-5",
    )
    assert sent["messages"][0]["content"] == [
        {**result, "cache_control": {"type": "ephemeral"}},
        thinking,
    ]


def test_add_prompt_cache_leaves_other_models_and_marked_requests_alone():
    body = {"messages": [{"role": "user", "content": "ls"}]}
    assert add_prompt_cache(body, "internal/glm-5.3-high-throughput") is body
    marked = {
        "messages": [
            {
                "role": "user",
                "content": [{"type": "text", "text": "ls", "cache_control": {}}],
            }
        ]
    }
    assert add_prompt_cache(marked, "anthropic/claude-sonnet-5.5") is marked
