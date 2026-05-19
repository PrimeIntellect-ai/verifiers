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


def _truncate_prompt_attribution(attribution: Any, prompt_len: int) -> Any:
    """Slice a ``renderers.RenderedTokens`` prompt-attribution sidecar to
    ``prompt_len`` tokens.

    Only the per-token lists (``token_ids`` / ``message_indices`` /
    ``sampled_mask`` / ``is_content``) need truncation; ``message_roles``
    is indexed by message position (not token position), so it stays
    intact even when some trailing messages contributed only truncated
    tokens — ``message_indices[k]`` still points into the correct slot.
    ``multi_modal_data`` is left as-is for the same reason: callers
    truncate it themselves if they need exact byte-alignment against the
    truncated prompt (matches the existing ``routed_experts`` policy
    where the slicing happens in :func:`truncate_routed_experts`).

    Returns ``None`` for falsy input so callers can chain through
    ``attribution = _truncate_prompt_attribution(attribution, N)``
    without branching.
    """
    if attribution is None:
        return None
    # Lazy import — keeps the hard ``renderers`` dependency out of
    # ``response_utils`` for clients that don't go through RendererClient.
    from renderers.base import RenderedTokens

    if not isinstance(attribution, RenderedTokens):
        return attribution

    return RenderedTokens(
        token_ids=list(attribution.token_ids[:prompt_len]),
        message_indices=list(attribution.message_indices[:prompt_len]),
        sampled_mask=list(attribution.sampled_mask[:prompt_len])
        if attribution.sampled_mask
        else [],
        is_content=list(attribution.is_content[:prompt_len])
        if attribution.is_content
        else [],
        message_roles=list(attribution.message_roles),
        multi_modal_data=attribution.multi_modal_data,
    )


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
    prompt_attribution = tokens.prompt_attribution

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
            prompt_attribution = _truncate_prompt_attribution(
                prompt_attribution, len(prompt_ids)
            )
        elif prompt_len + completion_len > max_seq_len:
            is_truncated = True
            completion_ids = tokens.completion_ids[: max_seq_len - prompt_len]
            completion_mask = tokens.completion_mask[: max_seq_len - prompt_len]
            completion_logprobs = tokens.completion_logprobs[: max_seq_len - prompt_len]
            routed_experts = truncate_routed_experts(
                routed_experts, prompt_len + len(completion_ids)
            )
            # ``prompt_attribution`` covers only the prompt and the
            # prompt itself wasn't truncated here, so no slicing needed.
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
    if prompt_attribution is not None:
        out["prompt_attribution"] = prompt_attribution
        # Same move-not-copy policy as ``multi_modal_data`` — the parsed
        # step is the canonical home; clearing the response-side ref
        # avoids duplicate serialisation on save / msgpack.
        tokens.prompt_attribution = None
    return out
