import asyncio
from typing import Any

from verifiers.types import (
    AssistantMessage,
    Messages,
    Response,
    TrajectoryStepTokens,
)

ROUTED_EXPERTS_DATA_PREFIX = b'"routed_experts":{"data":"'


def strip_routed_experts_data(raw: bytes) -> tuple[bytes, memoryview | None]:
    data_start = raw.find(ROUTED_EXPERTS_DATA_PREFIX)
    if data_start < 0:
        return raw, None

    data_start += len(ROUTED_EXPERTS_DATA_PREFIX)
    data_end = raw.index(b'"', data_start)
    routed_data = memoryview(raw)[data_start:data_end]
    stripped = raw[:data_start] + raw[data_end:]
    return stripped, routed_data


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
    truncated prompt (matches the previous ``routed_experts`` policy,
    now retired on main — routed_experts is no longer sliced here).

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

    def _sync() -> TrajectoryStepTokens | None:
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
                prompt_attribution = _truncate_prompt_attribution(
                    prompt_attribution, len(prompt_ids)
                )
            elif prompt_len + completion_len > max_seq_len:
                is_truncated = True
                completion_ids = tokens.completion_ids[: max_seq_len - prompt_len]
                completion_mask = tokens.completion_mask[: max_seq_len - prompt_len]
                completion_logprobs = tokens.completion_logprobs[
                    : max_seq_len - prompt_len
                ]
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
        if routed_experts is not None:
            tokens.routed_experts = None
        if prompt_attribution is not None:
            out["prompt_attribution"] = prompt_attribution
            # Same move-not-copy policy as ``multi_modal_data`` /
            # ``routed_experts`` — the parsed step is the canonical home;
            # clearing the response-side ref avoids duplicate
            # serialisation on save / msgpack.
            tokens.prompt_attribution = None
        return out

    return await asyncio.to_thread(_sync)
