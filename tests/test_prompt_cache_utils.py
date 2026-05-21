from verifiers.types import ClientConfig
from verifiers.utils.prompt_cache_utils import apply_prompt_cache_to_kwargs


def test_anthropic_cache_control_hint_is_default_only():
    extra_kwargs = apply_prompt_cache_to_kwargs(
        config=ClientConfig(
            client_type="anthropic_messages",
            api_base_url="https://api.anthropic.com/v1",
        ),
        sampling_args={"max_tokens": 16},
        extra_kwargs={},
    )

    assert extra_kwargs == {"cache_control": {"type": "ephemeral"}}

    extra_kwargs = apply_prompt_cache_to_kwargs(
        config=ClientConfig(
            client_type="anthropic_messages",
            api_base_url="https://api.anthropic.com/v1",
        ),
        sampling_args={"cache_control": {"type": "custom"}},
        extra_kwargs={},
    )

    assert extra_kwargs == {}
