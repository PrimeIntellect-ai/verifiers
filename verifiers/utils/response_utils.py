from typing import TYPE_CHECKING

from verifiers.types import (
    ChatCompletion,
    ChatMessage,
    Completion,
    Messages,
    MessageType,
    ModelResponse,
    TrajectoryStepTokens,
)

if TYPE_CHECKING:
    pass


async def parse_response_tokens(
    response: ModelResponse, message_type: MessageType
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
        if not hasattr(response.choices[0].logprobs, "content"):
            return None
        if response.choices[0].logprobs.content is None:
            return None
        prompt_ids = getattr(response, "prompt_token_ids")
        prompt_mask = [0] * len(prompt_ids)
        completion_ids = getattr(response.choices[0], "token_ids")
        completion_mask = [1] * len(completion_ids)
        completion_logprobs = [
            token.logprob for token in response.choices[0].logprobs.content
        ]

        return TrajectoryStepTokens(
            prompt_ids=prompt_ids,
            prompt_mask=prompt_mask,
            completion_ids=completion_ids,
            completion_mask=completion_mask,
            completion_logprobs=completion_logprobs,
        )
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

        return TrajectoryStepTokens(
            prompt_ids=prompt_ids,
            prompt_mask=prompt_mask,
            completion_ids=completion_ids,
            completion_mask=completion_mask,
            completion_logprobs=completion_logprobs,
        )
    return None


async def parse_response_messages(
    response: ModelResponse, message_type: MessageType
) -> Messages:
    response_text = ""
    if message_type == "chat":
        assert isinstance(response, ChatCompletion)
        if response.choices and response.choices[0].message:
            response_text = response.choices[0].message.content or ""
        response_message: ChatMessage = {
            "role": "assistant",
            "content": response_text,
        }
        if (
            response.choices
            and response.choices[0].message
            and response.choices[0].message.tool_calls
        ):
            tool_calls = response.choices[0].message.tool_calls
            response_message["tool_calls"] = [  # type: ignore
                tool_call.model_dump() for tool_call in tool_calls
            ]
        completion_messages = list[ChatMessage]([response_message])
    else:
        assert isinstance(response, Completion)
        if response.choices and response.choices[0]:
            response_text = response.choices[0].text or ""
        completion_messages = str(response_text)
    return completion_messages
