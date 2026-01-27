from openai.types.chat.chat_completion_message_function_tool_call import (
    ChatCompletionMessageFunctionToolCall,
)

from verifiers.types import (
    AssistantMessage,
    ChatCompletion,
    ChatMessage,
    ChatResponse,
    Completion,
    Messages,
    MessageType,
    Response,
    TextMessage,
    TextResponse,
    ToolCall,
    TrajectoryStepTokens,
)


async def parse_response_tokens(
    response: Response, message_type: MessageType, max_seq_len: int | None = None
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


def parse_text_response(response: TextResponse) -> TextMessage:
    text_message = response.choices[0].text
    return text_message


def parse_chat_response(response: ChatResponse) -> ChatMessage:
    if response.choices and response.choices[0].message:
        content = response.choices[0].message.content
    else:
        content = None
    # TODO: parse reasoning content
    if (
        response.choices
        and response.choices[0].message
        and response.choices[0].message.tool_calls
    ):
        oai_tool_calls = response.choices[0].message.tool_calls
        tool_calls = []
        for oai_tool_call in oai_tool_calls:
            if isinstance(
                oai_tool_call, ChatCompletionMessageFunctionToolCall
            ):  # only support function tool calls
                tool_calls.append(
                    ToolCall(
                        id=oai_tool_call.id,
                        name=oai_tool_call.function.name,
                        arguments=oai_tool_call.function.arguments,
                    )
                )
    else:
        tool_calls = None
    chat_message = AssistantMessage(
        role="assistant",
        content=content,
        reasoning_content=None,  # TODO
        tool_calls=tool_calls,
    )
    return chat_message


async def parse_response_messages(
    response: Response, message_type: MessageType
) -> Messages:
    if message_type == "chat":
        assert isinstance(response, ChatResponse)
        return [parse_chat_response(response)]
    else:
        assert isinstance(response, Completion)
        return parse_text_response(response)


async def parse_is_truncated(response: Response, message_type: MessageType) -> bool:
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
