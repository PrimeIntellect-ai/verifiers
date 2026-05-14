import base64
from io import BytesIO
from typing import Any, cast

import numpy as np

from verifiers.types import (
    AssistantMessage,
    Messages,
    Response,
    TrajectoryStepTokens,
)


def parse_routed_experts(raw: Any) -> str | None:
    if raw is None:
        return None
    return cast(str, raw)


def truncate_routed_experts(routed_experts: str | None, seq_len: int) -> str | None:
    if routed_experts is None:
        return None

    array = np.load(BytesIO(base64.b64decode(routed_experts)), allow_pickle=False)
    assert array.ndim == 3
    assert 0 <= seq_len <= array.shape[0]

    buffer = BytesIO()
    np.save(buffer, np.ascontiguousarray(array[:seq_len]), allow_pickle=False)
    return base64.b64encode(buffer.getvalue()).decode("ascii")


async def parse_response_message(response: Response) -> Messages:
    """Parse a vf.Response into a vf.Messages list (single vf.AssistantMessage)."""
    response_message = response.message
    extras = getattr(response_message, "model_extra", None) or {}
    message = AssistantMessage(
        content=response_message.content,
        reasoning_content=response_message.reasoning_content,
        thinking_blocks=response_message.thinking_blocks,
        tool_calls=response_message.tool_calls,
        **extras,
    )
    return [message]


async def parse_response_tokens(
    response: Response, max_seq_len: int | None = None
) -> TrajectoryStepTokens | None:
    """Parse token data from a vf.Response."""
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
    routed_experts = tokens.routed_experts
    multi_modal_data = tokens.multi_modal_data

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
            routed_experts = truncate_routed_experts(routed_experts, len(prompt_ids))
        elif prompt_len + completion_len > max_seq_len:
            is_truncated = True
            completion_ids = tokens.completion_ids[: max_seq_len - prompt_len]
            completion_mask = tokens.completion_mask[: max_seq_len - prompt_len]
            completion_logprobs = tokens.completion_logprobs[: max_seq_len - prompt_len]
            routed_experts = truncate_routed_experts(
                routed_experts, prompt_len + len(completion_ids)
            )
        else:
            is_truncated = False
    else:
        overlong_prompt = False
        is_truncated = False

    out = TrajectoryStepTokens(
        prompt_ids=prompt_ids,
        prompt_mask=prompt_mask,
        completion_ids=completion_ids,
        completion_mask=completion_mask,
        completion_logprobs=completion_logprobs,
        overlong_prompt=overlong_prompt,
        is_truncated=is_truncated,
        routed_experts=routed_experts,
    )
    if multi_modal_data is not None:
        out["multi_modal_data"] = multi_modal_data
        # Move (not copy) the sidecar to its canonical home on the parsed
        # step. Leaving it on ``response.message.tokens`` too means every
        # downstream pass (msgpack, save) has to dedupe the duplicate.
        tokens.multi_modal_data = None
    return out
