"""Each dialect imposes the eval's sampling in its native request shape, replacing the
harness's own values, and parses the same shape back for the trace."""

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
