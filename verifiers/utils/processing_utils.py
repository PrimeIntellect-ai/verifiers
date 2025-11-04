from typing import TYPE_CHECKING

from verifiers.types import ChatCompletion, ChatMessage, Completion, State

if TYPE_CHECKING:
    from transformers.tokenization_utils_base import (  # type: ignore
        PreTrainedTokenizerBase,
    )


def parse_chat_completion_logprobs(chat_completion: ChatCompletion) -> list[float]:
    """Parses the completion logprobs from a vLLM chat completion"""
    assert len(chat_completion.choices) == 1, "Response should always have one choice"
    assert chat_completion.choices[0].logprobs is not None, (
        "Logprobs should not be None. Make sure to set logprobs=True in the extra body when making the request to /v1/chat/completions"
    )
    assert chat_completion.choices[0].logprobs.content is not None, (
        "Logprob content should not be None. Make sure to set logprobs=True in the extra body when making the request to /v1/chat/completions"
    )
    logprobs = [
        logprob.logprob for logprob in chat_completion.choices[0].logprobs.content
    ]
    return logprobs


def parse_completion_logprobs(completion: Completion) -> list[float]:
    """Parses the completion logprobs from a vLLM chat completion"""
    assert len(completion.choices) == 1, "Response should always have one choice"
    assert completion.choices[0].logprobs is not None, (
        "Logprobs should not be None. Make sure to set logprobs=True in the extra body when making the request to /v1/completions"
    )
    assert completion.choices[0].logprobs.token_logprobs is not None, (
        "Logprob token_logprobs should not be None. Make sure to set logprobs=True in the extra body when making the request to /v1/completions"
    )
    return completion.choices[0].logprobs.token_logprobs


def parse_chat_completion_tokens(chat_completion: ChatCompletion) -> list[int]:
    """Parses the output token ids from a list of chat completions returned by vLLM OAI server."""
    assert len(chat_completion.choices) == 1, "Response should always have one choice"
    assert chat_completion.choices[0].logprobs is not None, (
        "Logprobs should not be None. Make sure to set logprobs=True in the extra body when making the request to /v1/chat/completions"
    )
    assert chat_completion.choices[0].logprobs.content is not None, (
        "Logprob content should not be None. Make sure to set logprobs=True in the extra body when making the request to /v1/chat/completions"
    )
    tokens = [
        # tokens are token_id:<int> because we request `return_tokens_as_token_ids` from vllm in GRPOTrainer
        int(token.token.split(":")[-1])
        for token in chat_completion.choices[0].logprobs.content
    ]
    return tokens


def parse_completion_tokens(completion: Completion) -> list[int]:
    """Parses the output token ids from a list of chat completions returned by vLLM OAI server."""
    assert len(completion.choices) == 1, "Response should always have one choice"
    assert completion.choices[0].logprobs is not None, (
        "Logprobs should not be None. Make sure to set logprobs=True in the extra body when making the request to /v1/completions"
    )
    assert completion.choices[0].logprobs.tokens is not None, (
        "Logprob tokens should not be None. Make sure to set logprobs=True in the extra body when making the request to /v1/completions"
    )
    tokens = [
        # tokens are token_id:<int> because we request `return_tokens_as_token_ids` from vllm in GRPOTrainer
        int(token.split(":")[-1])
        for token in completion.choices[0].logprobs.tokens
    ]
    return tokens


def process_chat_format_vllm(
    prompt: list[ChatMessage],
    completion: list[ChatMessage],
    state: State,
    processing_class: "PreTrainedTokenizerBase",
    mask_env_responses: bool = False,
) -> tuple[list[int], list[int], list[int], list[int], list[float]]:
    """
    Process chat format conversations using incremental prefixes.
    """
    responses = state["responses"]
    responses_idx = 0
    zipped = []
    for turn in completion:
        if turn["role"] == "assistant":
            zipped.append((turn, responses[responses_idx]))
            responses_idx += 1
        else:
            zipped.append((turn, None))
    assert len(responses) == responses_idx, "Responses not fully consumed"
    assert len(zipped) == len(completion), "Length mismatch"
    # get tools from state["info"]["oai_tools"]
    oai_tools = state.get("info", {}).get("oai_tools", []) or []
    prompt_ids: list[int] = processing_class.apply_chat_template(
        conversation=prompt,  # type: ignore
        add_generation_prompt=True,
        tools=oai_tools,
    )
    messages_consumed: list[ChatMessage | dict] = [m for m in prompt]
    prompt_mask: list[int] = [0] * len(prompt_ids)
    completion_ids: list[int] = []
    completion_mask: list[int] = []
    completion_logprobs: list[float] = []
    i = 0
    while i < len(zipped):
        message, response = zipped[i]

        # assistant case -- use response
        if message["role"] == "assistant":
            assert response is not None, "Response should not be None"
            completion_turn_ids = parse_chat_completion_tokens(response)
            completion_turn_mask = [1] * len(completion_turn_ids)
            completion_turn_logprobs = parse_chat_completion_logprobs(response)
            completion_ids.extend(completion_turn_ids)
            completion_mask.extend(completion_turn_mask)
            completion_logprobs.extend(completion_turn_logprobs)
            messages_consumed.append(message)
            i += 1
        # user/tool case -- use message
        else:
            assert message["role"] == "user" or message["role"] == "tool"
            # Collect all consecutive non-assistant messages
            consecutive_messages = [message]
            j = i + 1
            while j < len(zipped) and zipped[j][0]["role"] != "assistant":
                consecutive_messages.append(zipped[j][0])
                j += 1
            # Tokenize conversation ending at last completed assistant response
            token_prefix: list[int] = processing_class.apply_chat_template(
                conversation=messages_consumed,  # type: ignore
                add_generation_prompt=False,
                tools=oai_tools,
            )  # type: ignore
            # Tokenize with new user/tool messages and assistant prompt for next generation
            # Must include add_generation_prompt=True to match what vLLM sees
            token_prefix_with_turn: list[int] = processing_class.apply_chat_template(
                conversation=messages_consumed + consecutive_messages,  # type: ignore
                add_generation_prompt=True,
                tools=oai_tools,
            )  # type: ignore
            assert token_prefix_with_turn[: len(token_prefix)] == token_prefix, (
                f"Token prefix mismatch. Token prefix: {token_prefix}, token prefix with turn: {token_prefix_with_turn}"
            )
            completion_turn_ids = token_prefix_with_turn[len(token_prefix) :]
            if mask_env_responses:
                completion_turn_mask = [0] * len(completion_turn_ids)
            else:
                completion_turn_mask = [1] * len(completion_turn_ids)
            completion_turn_logprobs = [0.0] * len(completion_turn_ids)
            completion_ids.extend(completion_turn_ids)
            completion_mask.extend(completion_turn_mask)
            completion_logprobs.extend(completion_turn_logprobs)
            messages_consumed.extend(consecutive_messages)
            i = j
    return (
        prompt_ids,
        prompt_mask,
        completion_ids,
        completion_mask,
        completion_logprobs,
    )


def process_completion_format_vllm(
    prompt: str,
    completion: str,
    state: State,
    processing_class: "PreTrainedTokenizerBase",
    mask_env_responses: bool = False,
) -> tuple[list[int], list[int], list[int], list[int], list[float]]:
    """
    Process completion format conversations using incremental prefixes.
    """
    responses: list[Completion] = state["responses"]
    responses_start_idx: list[int] = state["responses_start_idx"]
    assert len(responses) == len(responses_start_idx), (
        "Should have an index for each completion response"
    )

    idx = 0
    zipped: list[tuple[str, Completion | None]] = []
    for response, response_start_idx in zip(responses, responses_start_idx):
        if response_start_idx > idx:
            # non-model-generated section
            zipped.append((completion[idx:response_start_idx], None))
        response_text = response.choices[0].text or ""
        zipped.append((response_text, response))
        idx = response_start_idx + len(response_text)
    assert idx == len(completion), "Completion not fully consumed"

    prompt_ids: list[int] = processing_class.encode(prompt)
    rollout_consumed = prompt
    prompt_mask: list[int] = [0] * len(prompt_ids)
    completion_ids: list[int] = []
    completion_mask: list[int] = []
    completion_logprobs: list[float] = []
    i = 0
    while i < len(zipped):
        text, response = zipped[i]
        # model-generated case -- use response
        if response is not None:
            completion_turn_ids = parse_completion_tokens(response)
            completion_turn_mask = [1] * len(completion_turn_ids)
            completion_turn_logprobs = parse_completion_logprobs(response)
            completion_ids.extend(completion_turn_ids)
            completion_mask.extend(completion_turn_mask)
            completion_logprobs.extend(completion_turn_logprobs)
            rollout_consumed += text
            i += 1
        # non-model-generated (user/tool case) -- use text
        else:
            token_prefix: list[int] = processing_class.encode(rollout_consumed)
            token_prefix_with_turn: list[int] = processing_class.encode(
                rollout_consumed + text
            )
            assert token_prefix_with_turn[: len(token_prefix)] == token_prefix, (
                f"Token prefix mismatch. Token prefix: {token_prefix}, token prefix with turn: {token_prefix_with_turn}"
            )
            completion_turn_ids = token_prefix_with_turn[len(token_prefix) :]
            if mask_env_responses:
                completion_turn_mask = [0] * len(completion_turn_ids)
            else:
                completion_turn_mask = [1] * len(completion_turn_ids)
            completion_turn_logprobs = [0.0] * len(completion_turn_ids)
            completion_ids.extend(completion_turn_ids)
            completion_mask.extend(completion_turn_mask)
            completion_logprobs.extend(completion_turn_logprobs)
            rollout_consumed += text
            i += 1
    return (
        prompt_ids,
        prompt_mask,
        completion_ids,
        completion_mask,
        completion_logprobs,
    )
