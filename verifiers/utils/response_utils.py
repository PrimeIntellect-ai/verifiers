from typing import cast

from verifiers.types import (
    ChatCompletion,
    ChatMessage,
    Completion,
    Messages,
    MessageType,
    ModelResponse,
    TrajectoryStepTokens,
)


async def parse_response_tokens(
    response: ModelResponse, message_type: MessageType, max_seq_len: int | None = None
) -> TrajectoryStepTokens | None:
    if message_type == "chat":
        assert isinstance(response, ChatCompletion)
        assert len(response.choices) == 1, "Response should always have one choice"
        if not hasattr(response.choices[0], "token_ids"):
            return None
        if not hasattr(response, "prompt_token_ids"):
            return None
        if not hasattr(response.choices[0], "logprobs"):
            return None
        if response.choices[0].logprobs is None:
            return None
        has_logprobs_obj = (
            hasattr(response.choices[0].logprobs, "content")
            and response.choices[0].logprobs.content is not None
        )
        has_logprobs_dict = (
            isinstance(response.choices[0].logprobs, dict)
            and "content" in response.choices[0].logprobs.keys()
            and response.choices[0].logprobs["content"] is not None
        )
        if not (has_logprobs_obj or has_logprobs_dict):
            return None
        prompt_ids = getattr(response, "prompt_token_ids")
        prompt_mask = [0] * len(prompt_ids)
        completion_ids = getattr(response.choices[0], "token_ids")
        completion_mask = [1] * len(completion_ids)
        if has_logprobs_obj:
            assert response.choices[0].logprobs.content is not None
            logprobs_content = response.choices[0].logprobs.content
            completion_logprobs = [token.logprob for token in logprobs_content]
        else:
            assert isinstance(response.choices[0].logprobs, dict)
            logprobs_content = response.choices[0].logprobs["content"]
            completion_logprobs = [token["logprob"] for token in logprobs_content]
    elif message_type == "completion":
        assert isinstance(response, Completion)
        if not hasattr(response.choices[0], "prompt_token_ids"):
            return None
        if not hasattr(response.choices[0], "token_ids"):
            return None
        if not hasattr(response.choices[0], "logprobs"):
            return None
        if response.choices[0].logprobs is None:
            return None
        if not hasattr(response.choices[0].logprobs, "token_logprobs"):
            return None
        prompt_ids = getattr(response.choices[0], "prompt_token_ids")
        prompt_mask = [0] * len(prompt_ids)
        completion_ids = getattr(response.choices[0], "token_ids")
        completion_mask = [1] * len(completion_ids)
        completion_logprobs = getattr(response.choices[0].logprobs, "token_logprobs")
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
            completion_ids = completion_ids[: max_seq_len - prompt_len]
            completion_mask = completion_mask[: max_seq_len - prompt_len]
            completion_logprobs = completion_logprobs[: max_seq_len - prompt_len]
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


async def parse_response_messages(
    response: ModelResponse, message_type: MessageType
) -> Messages:
    response_text = ""
    if message_type == "chat":
        assert isinstance(response, ChatCompletion)
        if response.choices and response.choices[0].message:
            response_text = response.choices[0].message.content or ""
        response_message: dict[str, object] = {
            "role": "assistant",
            "content": response_text,
        }
        if (
            response.choices
            and response.choices[0].message
            and response.choices[0].message.tool_calls
        ):
            tool_calls = response.choices[0].message.tool_calls
            response_message["tool_calls"] = [
                tool_call.model_dump() for tool_call in tool_calls
            ]
        completion_messages = list[ChatMessage]([cast(ChatMessage, response_message)])
    else:
        assert isinstance(response, Completion)
        if response.choices and response.choices[0]:
            response_text = response.choices[0].text or ""
        completion_messages = str(response_text)
    return completion_messages
