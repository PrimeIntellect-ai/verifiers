"""Each dialect imposes the eval's sampling in its native request shape, replacing the
harness's own values, and parses the same shape back for the trace."""

import pytest

from verifiers.v1.dialects import AnthropicDialect, ChatDialect, ResponsesDialect
from verifiers.v1.types import SamplingConfig

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


@pytest.mark.parametrize("key", ["max_tokens", "max_completion_tokens"])
@pytest.mark.parametrize("layer", ["direct", "run", "seat"])
def test_explicit_null_clears_chat_output_limits(key, layer):
    from verifiers.v1.configs.agent import AgentConfig, resolve_agent

    sampling = SamplingConfig.model_validate({key: None})
    if layer == "run":
        sampling = resolve_agent(AgentConfig(), sampling=sampling).sampling
    elif layer == "seat":
        sampling = resolve_agent(
            AgentConfig(sampling=sampling),
            sampling=SamplingConfig(max_tokens=32768),
        ).sampling
    original = {"max_tokens": 32768, "max_completion_tokens": 32768, "messages": []}
    sent = ChatDialect().apply_overrides(original, "m", sampling)
    assert "max_tokens" not in sent and "max_completion_tokens" not in sent
    assert original["max_tokens"] == original["max_completion_tokens"] == 32768


def test_unset_chat_sampling_preserves_the_harness_limit():
    from verifiers.v1.configs.agent import AgentConfig, resolve_agent

    sampling = resolve_agent(AgentConfig(), sampling=SamplingConfig()).sampling
    assert "max_tokens" not in sampling.model_fields_set
    assert (
        ChatDialect().apply_overrides({"max_tokens": 512}, "m", sampling)["max_tokens"]
        == 512
    )
