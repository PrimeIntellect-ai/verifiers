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
        if (
            not hasattr(response.choices[0], "token_ids")
            or response.choices[0].token_ids is None
        ):
            return None
        if (
            not hasattr(response, "prompt_token_ids")
            or response.prompt_token_ids is None
        ):
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


class TopLogprobs:
    """Compact top-k logprobs: two parallel lists coupled by index.

    ``tokens[i][j]`` is the j-th most likely token at completion position i.
    ``logprobs[i][j]`` is the corresponding log-probability.
    """

    __slots__ = ("tokens", "logprobs")

    def __init__(self, tokens: list[list[str]], logprobs: list[list[float]]) -> None:
        self.tokens = tokens
        self.logprobs = logprobs


async def extract_top_logprobs(
    response: ModelResponse, message_type: MessageType
) -> TopLogprobs:
    """Extract top-k logprobs from a standard OpenAI/vLLM response.

    Returns a ``TopLogprobs`` with two parallel ``list[list[...]]``
    (tokens and logprobs), coupled by index.

    Raises ``ValueError`` with a specific message when the response does
    not contain the expected logprobs data.
    """
    if response is None:
        raise ValueError("Response is None; cannot extract top_logprobs.")

    if message_type == "chat":
        if not isinstance(response, ChatCompletion):
            raise ValueError(
                f"Expected ChatCompletion response, got {type(response).__name__}."
            )
        if not response.choices:
            raise ValueError("Response has no choices.")
        choice = response.choices[0]
        if not hasattr(choice, "logprobs") or choice.logprobs is None:
            raise ValueError(
                "Response choice has no logprobs. "
                "Ensure logprobs=true is set in sampling args."
            )
        logprobs_obj = choice.logprobs

        # logprobs_obj may be an object or a dict (depends on provider)
        if hasattr(logprobs_obj, "content") and logprobs_obj.content is not None:
            content = logprobs_obj.content
        elif isinstance(logprobs_obj, dict) and logprobs_obj.get("content") is not None:
            content = logprobs_obj["content"]
        else:
            raise ValueError(
                "Response logprobs has no content. "
                "The endpoint may not support top_logprobs."
            )

        all_tokens: list[list[str]] = []
        all_logprobs: list[list[float]] = []
        for token_info in content:
            if hasattr(token_info, "top_logprobs") and token_info.top_logprobs:
                toks = [t.token for t in token_info.top_logprobs]
                lps = [t.logprob for t in token_info.top_logprobs]
            elif isinstance(token_info, dict) and token_info.get("top_logprobs"):
                toks = [t["token"] for t in token_info["top_logprobs"]]
                lps = [t["logprob"] for t in token_info["top_logprobs"]]
            else:
                # top_logprobs missing on this token but logprobs present —
                # fall back to the chosen token's logprob
                if hasattr(token_info, "token"):
                    toks = [token_info.token]
                    lps = [token_info.logprob]
                elif isinstance(token_info, dict):
                    toks = [token_info["token"]]
                    lps = [token_info["logprob"]]
                else:
                    raise ValueError(
                        f"Unexpected token_info format: {type(token_info).__name__}."
                    )
            all_tokens.append(toks)
            all_logprobs.append(lps)
        if not all_tokens:
            raise ValueError("Response logprobs content is empty.")
        return TopLogprobs(all_tokens, all_logprobs)

    elif message_type == "completion":
        if not isinstance(response, Completion):
            raise ValueError(
                f"Expected Completion response, got {type(response).__name__}."
            )
        if not response.choices:
            raise ValueError("Response has no choices.")
        choice = response.choices[0]
        if not hasattr(choice, "logprobs") or choice.logprobs is None:
            raise ValueError(
                "Response choice has no logprobs. "
                "Ensure logprobs=true is set in sampling args."
            )
        top_logprobs_list = getattr(choice.logprobs, "top_logprobs", None)
        if not top_logprobs_list:
            raise ValueError(
                "Response logprobs has no top_logprobs. "
                "The endpoint may not support the top_logprobs parameter."
            )
        all_tokens: list[list[str]] = []
        all_logprobs: list[list[float]] = []
        for position_dict in top_logprobs_list:
            if isinstance(position_dict, dict):
                toks = list(position_dict.keys())
                lps = list(position_dict.values())
            else:
                raise ValueError(
                    f"Unexpected top_logprobs entry format: {type(position_dict).__name__}."
                )
            all_tokens.append(toks)
            all_logprobs.append(lps)
        if not all_tokens:
            raise ValueError("Response top_logprobs is empty.")
        return TopLogprobs(all_tokens, all_logprobs)

    raise ValueError(f"Unsupported message_type: {message_type!r}.")


async def parse_is_truncated(
    response: ModelResponse, message_type: MessageType
) -> bool:
    if message_type == "chat":
        assert isinstance(response, ChatCompletion)
        assert len(response.choices) == 1, "Response should always have one choice"
        return response.choices[0].finish_reason == "length"
    elif message_type == "completion":
        assert isinstance(response, Completion)
        assert len(response.choices) == 1, "Response should always have one choice"
        return response.choices[0].finish_reason == "length"
    else:
        return False
