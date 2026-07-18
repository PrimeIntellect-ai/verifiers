"""Dialect response validation: preserving provider-specific `service_tier` values.

OpenAI pins `service_tier` to a closed Literal, but OpenAI-compatible gateways (e.g. OpenRouter)
report their own tiers like `provisioned` or `openai/flex`. vf must stay agnostic: preserve the
provider's value rather than rejecting the response or dropping the field.
"""

import pytest

from verifiers.v1.dialects.anthropic import AnthropicDialect
from verifiers.v1.dialects.chat import ChatDialect


def _chat_raw(service_tier):
    return {
        "id": "x",
        "object": "chat.completion",
        "created": 0,
        "model": "google/gemini-3-flash-preview",
        "service_tier": service_tier,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": "hi"},
                "finish_reason": "stop",
            }
        ],
        "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
    }


@pytest.mark.parametrize(
    "tier", ["provisioned", "openai/flex", "default", "priority", None]
)
def test_chat_preserves_service_tier(tier):
    completion = ChatDialect().validate_response(_chat_raw(tier))
    assert completion.service_tier == tier
    # the response still parses end-to-end
    assert ChatDialect().parse_response(completion).message.content == "hi"


def _anthropic_raw(service_tier):
    return {
        "id": "x",
        "type": "message",
        "role": "assistant",
        "model": "claude-x",
        "content": [{"type": "text", "text": "hi"}],
        "stop_reason": "end_turn",
        "usage": {
            "input_tokens": 1,
            "output_tokens": 1,
            "service_tier": service_tier,
        },
    }


@pytest.mark.parametrize("tier", ["provisioned", "standard", "priority", None])
def test_anthropic_preserves_service_tier(tier):
    message = AnthropicDialect().validate_response(_anthropic_raw(tier))
    assert message.usage.service_tier == tier
    assert AnthropicDialect().parse_response(message).message.content == "hi"
