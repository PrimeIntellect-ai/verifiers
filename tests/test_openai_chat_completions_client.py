from types import SimpleNamespace

import pytest

from verifiers.clients.openai_chat_completions_client import (
    OpenAIChatCompletionsClient,
    normalize_openai_chat_sampling_args,
)


def test_normalize_openai_chat_sampling_args_moves_unknown_fields_into_extra_body():
    normalized = normalize_openai_chat_sampling_args(
        {
            "temperature": 0.2,
            "reasoning": {"effort": "high"},
        }
    )

    assert normalized == {
        "temperature": 0.2,
        "extra_body": {"reasoning": {"effort": "high"}},
    }


def test_normalize_openai_chat_sampling_args_merges_extra_body_with_unknown_fields():
    normalized = normalize_openai_chat_sampling_args(
        {
            "extra_body": {"provider": {"order": ["azure"]}},
            "reasoning": {"enabled": True},
            "temperature": 0.1,
        }
    )

    assert normalized == {
        "temperature": 0.1,
        "extra_body": {
            "provider": {"order": ["azure"]},
            "reasoning": {"enabled": True},
        },
    }


def test_normalize_openai_chat_sampling_args_renames_max_tokens():
    normalized = normalize_openai_chat_sampling_args({"max_tokens": 123})

    assert normalized == {"max_completion_tokens": 123}


def test_normalize_openai_chat_sampling_args_preserves_unknown_top_level_keys():
    normalized = normalize_openai_chat_sampling_args({"temperatur": 0.2})

    assert normalized == {"temperatur": 0.2}


@pytest.mark.asyncio
async def test_get_native_response_passes_reasoning_via_extra_body():
    recorded: dict[str, object] = {}

    class DummyCompletions:
        async def create(self, **kwargs):  # noqa: ANN003
            recorded.update(kwargs)
            return SimpleNamespace()

    raw_client = SimpleNamespace(chat=SimpleNamespace(completions=DummyCompletions()))
    client = OpenAIChatCompletionsClient(raw_client)

    response = await client.get_native_response(
        prompt=[{"role": "user", "content": "hello"}],
        model="test-model",
        sampling_args={
            "temperature": 0.3,
            "reasoning": {"effort": "medium"},
        },
    )

    assert isinstance(response, SimpleNamespace)
    assert recorded == {
        "model": "test-model",
        "messages": [{"role": "user", "content": "hello"}],
        "extra_headers": None,
        "temperature": 0.3,
        "extra_body": {"reasoning": {"effort": "medium"}},
    }
