from typing import Any, cast

from verifiers.types import (
    AssistantMessage,
    MessageType,
    Messages,
    Response,
    TextMessage,
    TrajectoryStepTokens,
)


# --- New Response-based utilities ---


def _content_to_text(content: object) -> str:
    if isinstance(content, str):
        return content
    if content is None:
        return ""
    if isinstance(content, list):
        chunks: list[str] = []
        for part in content:
            if isinstance(part, dict):
                part_dict = cast(dict[str, Any], part)
                if part_dict.get("type") == "text":
                    text = part_dict.get("text")
                    if isinstance(text, str):
                        chunks.append(text)
                continue
            text = getattr(part, "text", None)
            if isinstance(text, str):
                chunks.append(text)
        return "".join(chunks)
    return str(content)


async def parse_response_message(
    response: Response, message_type: MessageType = "chat"
) -> Messages:
    """Parse a vf.Response into a Messages list for chat or raw completion mode."""
    response_message = response.message
    if message_type == "completion":
        return [TextMessage(content=_content_to_text(response_message.content))]

    message = AssistantMessage(
        role="assistant",
        content=response_message.content,
        reasoning_content=response_message.reasoning_content,
        tool_calls=response_message.tool_calls,
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
