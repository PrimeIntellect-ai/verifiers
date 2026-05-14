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


@pytest.mark.asyncio
async def test_to_native_prompt_emits_token_extras():
    """Assistant message extras flow onto the outgoing native dict."""
    from verifiers.types import AssistantMessage, UserMessage

    client = _make_client()
    assistant = AssistantMessage(content="hello")
    assistant.prompt_token_ids = [1, 2, 3]
    assistant.generation_token_ids = [4, 5]
    assistant.generation_log_probs = [-0.1, -0.2]
    messages = [UserMessage(content="hi"), assistant, UserMessage(content="more")]

    native_messages, _ = await client.to_native_prompt(messages)
    assert native_messages[1]["prompt_token_ids"] == [1, 2, 3]
    assert native_messages[1]["generation_token_ids"] == [4, 5]
    assert native_messages[1]["generation_log_probs"] == [-0.1, -0.2]
    assert "prompt_token_ids" not in native_messages[0]
    assert "prompt_token_ids" not in native_messages[2]


@pytest.mark.asyncio
async def test_to_native_prompt_without_extras_is_clean():
    """AssistantMessages with no token extras serialize without token fields."""
    from verifiers.types import AssistantMessage

    client = _make_client()
    native_messages, _ = await client.to_native_prompt(
        [AssistantMessage(content="plain")]
    )
    for key in ("prompt_token_ids", "generation_token_ids", "generation_log_probs"):
        assert key not in native_messages[0]


@pytest.mark.asyncio
async def test_get_response_annotates_from_trajectory():
    """get_response walks state['trajectory'] and annotates matching assistants."""
    from verifiers.clients.nemorl_chat_completions_client import (
        _attach_trajectory_tokens_to_prompt,
    )
    from verifiers.types import AssistantMessage, UserMessage

    a0 = AssistantMessage(content="turn0")
    a1 = AssistantMessage(content="turn1")
    prompt = [
        UserMessage(content="u0"),
        a0,
        UserMessage(content="u1"),
        a1,
        UserMessage(content="u2"),
    ]
    state = {
        "trajectory": [
            {
                "tokens": {
                    "prompt_ids": [1],
                    "completion_ids": [10],
                    "completion_logprobs": [-0.5],
                }
            },
            {
                "tokens": {
                    "prompt_ids": [1, 10, 2],
                    "completion_ids": [20, 21],
                    "completion_logprobs": [-0.1, -0.2],
                }
            },
        ]
    }
    _attach_trajectory_tokens_to_prompt(prompt, state)
    assert a0.prompt_token_ids == [1]
    assert a0.generation_token_ids == [10]
    assert a1.prompt_token_ids == [1, 10, 2]
    assert a1.generation_token_ids == [20, 21]
    assert a1.generation_log_probs == [-0.1, -0.2]


@pytest.mark.asyncio
async def test_get_response_empty_trajectory_is_noop():
    from verifiers.clients.nemorl_chat_completions_client import (
        _attach_trajectory_tokens_to_prompt,
    )
    from verifiers.types import AssistantMessage, UserMessage

    a = AssistantMessage(content="x")
    prompt = [UserMessage(content="u"), a]
    _attach_trajectory_tokens_to_prompt(prompt, {"trajectory": []})
    assert not hasattr(a, "prompt_token_ids") or a.prompt_token_ids is None


@pytest.mark.asyncio
async def test_get_response_none_tokens_step_is_skipped():
    from verifiers.clients.nemorl_chat_completions_client import (
        _attach_trajectory_tokens_to_prompt,
    )
    from verifiers.types import AssistantMessage, UserMessage

    a0 = AssistantMessage(content="x")
    a1 = AssistantMessage(content="y")
    prompt = [UserMessage(content="u"), a0, UserMessage(content="u"), a1]
    state = {
        "trajectory": [
            {"tokens": None},
            {
                "tokens": {
                    "prompt_ids": [1],
                    "completion_ids": [2],
                    "completion_logprobs": [-0.1],
                }
            },
        ]
    }
    _attach_trajectory_tokens_to_prompt(prompt, state)
    assert not hasattr(a0, "prompt_token_ids") or a0.prompt_token_ids is None
    assert a1.prompt_token_ids == [1]


@pytest.mark.asyncio
async def test_get_response_positional_pairing_with_extra_leading_assistant():
    """Extra leading assistant messages (e.g. few-shot) are ignored; last N paired."""
    from verifiers.clients.nemorl_chat_completions_client import (
        _attach_trajectory_tokens_to_prompt,
    )
    from verifiers.types import AssistantMessage, UserMessage

    few_shot = AssistantMessage(content="few-shot example")
    a0 = AssistantMessage(content="turn0")
    prompt = [UserMessage(content="demo"), few_shot, UserMessage(content="u0"), a0]
    state = {
        "trajectory": [
            {
                "tokens": {
                    "prompt_ids": [7],
                    "completion_ids": [8],
                    "completion_logprobs": [-0.3],
                }
            },
        ]
    }
    _attach_trajectory_tokens_to_prompt(prompt, state)
    assert (
        not hasattr(few_shot, "prompt_token_ids") or few_shot.prompt_token_ids is None
    )
    assert a0.prompt_token_ids == [7]
    assert a0.generation_token_ids == [8]
