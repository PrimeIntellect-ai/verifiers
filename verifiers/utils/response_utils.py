from verifiers.types import (
    AssistantMessage,
    Messages,
    Response,
    TrajectoryStepTokens,
)


async def parse_response_tokens(
    response: Response, max_seq_len: int | None = None
) -> TrajectoryStepTokens | None:
    if response is None:
        return None
    tokens = response.message.tokens
    if tokens is None:
        return None
    prompt_ids = tokens.prompt_ids
    prompt_mask = tokens.prompt_mask
    completion_ids = tokens.completion_ids
    completion_mask = tokens.completion_mask
    completion_logprobs = tokens.completion_logprobs

    if max_seq_len is not None:
        prompt_len = len(prompt_ids)
        completion_len = len(completion_ids)
        overlong_prompt = prompt_len > max_seq_len
        if overlong_prompt:
            is_truncated = True
            prompt_ids = prompt_ids[:max_seq_len]
            prompt_mask = prompt_mask[:max_seq_len]
            completion_ids = []
            completion_mask = []
            completion_logprobs = []
        elif prompt_len + completion_len > max_seq_len:
            is_truncated = True
            completion_ids = tokens.completion_ids[: max_seq_len - prompt_len]
            completion_mask = tokens.completion_mask[: max_seq_len - prompt_len]
            completion_logprobs = tokens.completion_logprobs[: max_seq_len - prompt_len]
        else:
            is_truncated = False
    else:
        overlong_prompt = False
        is_truncated = False

    return TrajectoryStepTokens(
        prompt_ids=prompt_ids,
        prompt_mask=prompt_mask,
        completion_ids=completion_ids,
        completion_mask=completion_mask,
        completion_logprobs=completion_logprobs,
        overlong_prompt=overlong_prompt,
        is_truncated=is_truncated,
    )


async def parse_response_message(response: Response) -> Messages:
    response_message = response.message
    message = AssistantMessage(
        role="assistant",
        content=response_message.content,
        reasoning_content=response_message.reasoning_content,
        tool_calls=response_message.tool_calls,
    )
    return [message]
