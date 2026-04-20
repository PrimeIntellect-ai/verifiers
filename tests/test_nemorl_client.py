from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest

from verifiers.clients.nemorl_chat_completions_client import (
    NeMoRLChatCompletionsClient,
)
from verifiers.types import ClientConfig


def _make_nemo_gym_response_dict(
    *,
    content: str = "Hello",
    prompt_token_ids: list[int] | None = None,
    generation_token_ids: list[int | str] | None = None,
    generation_log_probs: list[float] | None = None,
) -> dict:
    """Build a JSON dict mimicking the NeMo Gym vllm_model server response.

    The vllm_model server embeds token data as extra fields in the message dict
    and strips logprobs from the choice (they're redundant once extracted).
    """
    message: dict = {"role": "assistant", "content": content}
    if prompt_token_ids is not None:
        message["prompt_token_ids"] = prompt_token_ids
    if generation_token_ids is not None:
        message["generation_token_ids"] = generation_token_ids
    if generation_log_probs is not None:
        message["generation_log_probs"] = generation_log_probs

    return {
        "id": "chatcmpl-test",
        "object": "chat.completion",
        "created": 1700000000,
        "model": "test-model",
        "choices": [
            {
                "index": 0,
                "message": message,
                "finish_reason": "stop",
                "logprobs": None,  # vllm_model pops raw logprobs after extraction
            }
        ],
        "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
    }


def _make_client() -> NeMoRLChatCompletionsClient:
    """Create a client with a mocked AsyncOpenAI underneath."""
    config = ClientConfig(
        client_type="nemorl_chat_completions",
        api_base_url="http://localhost:8080/v1",
        api_key_var="EMPTY",
    )
    client = NeMoRLChatCompletionsClient(config)
    return client


@pytest.mark.asyncio
async def test_token_ids_relocated_from_message_to_response():
    """Token IDs in the message dict are relocated to response/choice level."""
    prompt_ids = [1, 2, 3, 4, 5]
    gen_ids = [10, 11, 12]
    gen_logprobs = [-0.1, -0.2, -0.3]

    response_dict = _make_nemo_gym_response_dict(
        prompt_token_ids=prompt_ids,
        generation_token_ids=gen_ids,
        generation_log_probs=gen_logprobs,
    )

    client = _make_client()

    # Mock the parent's get_native_response to return a ChatCompletion
    # parsed from our dict (simulating what the OpenAI SDK does).
    from openai.types.chat import ChatCompletion

    chat_completion = ChatCompletion.model_validate(response_dict)
    # Simulate the message having extra fields (OpenAI SDK extra="allow")
    setattr(chat_completion.choices[0].message, "prompt_token_ids", prompt_ids)
    setattr(chat_completion.choices[0].message, "generation_token_ids", gen_ids)
    setattr(chat_completion.choices[0].message, "generation_log_probs", gen_logprobs)

    with patch.object(
        NeMoRLChatCompletionsClient.__bases__[0],
        "get_native_response",
        new_callable=AsyncMock,
        return_value=chat_completion,
    ):
        result = await client.get_native_response(
            prompt=[{"role": "user", "content": "Hi"}],
            model="test-model",
            sampling_args={"max_tokens": 100},
        )

    # Verify relocation
    assert hasattr(result, "prompt_token_ids")
    assert result.prompt_token_ids == prompt_ids
    assert hasattr(result.choices[0], "token_ids")
    assert result.choices[0].token_ids == gen_ids

    # Verify logprobs reconstruction
    assert result.choices[0].logprobs is not None
    logprobs_content = result.choices[0].logprobs["content"]
    assert len(logprobs_content) == 3
    assert logprobs_content[0]["token"] == "token_id:10"
    assert logprobs_content[0]["logprob"] == -0.1


@pytest.mark.asyncio
async def test_string_token_ids_normalized_to_ints():
    """String token IDs (from vllm_model removeprefix) are converted to ints."""
    response_dict = _make_nemo_gym_response_dict(
        prompt_token_ids=[1, 2],
        generation_token_ids=["100", "200"],  # strings, not ints
        generation_log_probs=[-0.5, -0.6],
    )

    client = _make_client()

    from openai.types.chat import ChatCompletion

    chat_completion = ChatCompletion.model_validate(response_dict)
    setattr(chat_completion.choices[0].message, "prompt_token_ids", [1, 2])
    setattr(chat_completion.choices[0].message, "generation_token_ids", ["100", "200"])
    setattr(chat_completion.choices[0].message, "generation_log_probs", [-0.5, -0.6])

    with patch.object(
        NeMoRLChatCompletionsClient.__bases__[0],
        "get_native_response",
        new_callable=AsyncMock,
        return_value=chat_completion,
    ):
        result = await client.get_native_response(
            prompt=[{"role": "user", "content": "Hi"}],
            model="test-model",
            sampling_args={"max_tokens": 100},
        )

    # Token IDs should be ints, not strings
    assert all(isinstance(tid, int) for tid in result.choices[0].token_ids)
    assert result.choices[0].token_ids == [100, 200]


@pytest.mark.asyncio
async def test_no_token_data_passes_through_unchanged():
    """When model server doesn't return token IDs, response is unchanged."""
    response_dict = _make_nemo_gym_response_dict()  # no token fields

    client = _make_client()

    from openai.types.chat import ChatCompletion

    chat_completion = ChatCompletion.model_validate(response_dict)

    with patch.object(
        NeMoRLChatCompletionsClient.__bases__[0],
        "get_native_response",
        new_callable=AsyncMock,
        return_value=chat_completion,
    ):
        result = await client.get_native_response(
            prompt=[{"role": "user", "content": "Hi"}],
            model="test-model",
            sampling_args={"max_tokens": 100},
        )

    # No token attributes should be set
    assert not hasattr(result, "prompt_token_ids")
    assert not hasattr(result.choices[0], "token_ids")
    assert result.choices[0].logprobs is None


@pytest.mark.asyncio
async def test_from_native_response_produces_response_tokens():
    """End-to-end: token IDs flow through to ResponseTokens via parse_tokens."""
    prompt_ids = [1, 2, 3]
    gen_ids = [10, 11]
    gen_logprobs = [-0.1, -0.2]

    response_dict = _make_nemo_gym_response_dict(
        prompt_token_ids=prompt_ids,
        generation_token_ids=gen_ids,
        generation_log_probs=gen_logprobs,
    )

    client = _make_client()

    from openai.types.chat import ChatCompletion

    chat_completion = ChatCompletion.model_validate(response_dict)
    setattr(chat_completion.choices[0].message, "prompt_token_ids", prompt_ids)
    setattr(chat_completion.choices[0].message, "generation_token_ids", gen_ids)
    setattr(chat_completion.choices[0].message, "generation_log_probs", gen_logprobs)

    with patch.object(
        NeMoRLChatCompletionsClient.__bases__[0],
        "get_native_response",
        new_callable=AsyncMock,
        return_value=chat_completion,
    ):
        native_response = await client.get_native_response(
            prompt=[{"role": "user", "content": "Hi"}],
            model="test-model",
            sampling_args={"max_tokens": 100},
        )

    # Now test from_native_response which calls parse_tokens
    vf_response = await client.from_native_response(native_response)

    assert vf_response.message.tokens is not None
    tokens = vf_response.message.tokens
    assert tokens.prompt_ids == prompt_ids
    assert tokens.completion_ids == gen_ids
    assert tokens.completion_logprobs == gen_logprobs
    assert tokens.prompt_mask == [0] * len(prompt_ids)
    assert tokens.completion_mask == [1] * len(gen_ids)
