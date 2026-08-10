from verifiers.v1.dialects.chat import ChatDialect
from verifiers.v1.types import SamplingConfig


def test_chat_sampling_overrides_are_authoritative():
    body = {
        "model": "harness-model",
        "messages": [{"role": "user", "content": "hello"}],
        "stream": True,
        "temperature": 0.2,
        "top_p": 0.8,
        "top_k": 20,
        "max_tokens": 16_384,
        "max_completion_tokens": 8_192,
        "reasoning_effort": "low",
        "stop": ["done"],
        "tool_choice": "auto",
    }

    result = ChatDialect().apply_overrides(
        body,
        "eval-model",
        SamplingConfig(temperature=1.0, top_p=1.0, reasoning_effort="max"),
    )

    assert result == {
        "model": "eval-model",
        "messages": [{"role": "user", "content": "hello"}],
        "stream": True,
        "temperature": 1.0,
        "top_p": 1.0,
        "reasoning_effort": "max",
    }
    assert body["max_tokens"] == 16_384

    limited = ChatDialect().apply_overrides(
        body, "eval-model", SamplingConfig(max_tokens=4_096)
    )
    assert limited["max_tokens"] == 4_096
    assert "max_completion_tokens" not in limited
