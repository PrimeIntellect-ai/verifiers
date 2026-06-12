"""OpenAI-compatible chat-completions client."""

from typing import Any, cast

from openai import AsyncOpenAI, OpenAIError
from openai.types.chat import (
    ChatCompletion,
    ChatCompletionAssistantMessageParam,
    ChatCompletionContentPartImageParam,
    ChatCompletionContentPartParam,
    ChatCompletionContentPartTextParam,
    ChatCompletionFunctionToolParam,
    ChatCompletionMessageFunctionToolCall,
    ChatCompletionMessageFunctionToolCallParam,
    ChatCompletionMessageParam,
    ChatCompletionSystemMessageParam,
    ChatCompletionToolMessageParam,
    ChatCompletionUserMessageParam,
)
from openai.types.chat.chat_completion import Choice
from openai.types.shared_params import FunctionDefinition

from verifiers.v1.clients.client import Client
from verifiers.v1.errors import ModelError, OverlongPromptError
from verifiers.v1.types import (
    AssistantMessage,
    FinishReason,
    Message,
    Messages,
    Response,
    SamplingConfig,
    SystemMessage,
    TextContentPart,
    Tool,
    ToolCall,
    ToolMessage,
    TurnTokens,
    Usage,
    UserMessage,
)

_CONTEXT_LENGTH_PHRASES = (
    "this model's maximum context length is",
    "is longer than the model's context length",
    "is longer than the maximum model length",
    "exceeds the model's context length",
    "exceed the configured limit",
    "exceeds the configured limit",
    "exceeded model",
    "prompt_too_long",
    "context length",
    "maximum model length",
)


def model_error(e: OpenAIError) -> ModelError:
    """Map a provider client error to our error type, distinguishing an overlong prompt
    from any other model-call failure (auth, rate limit, a genuine bad request, ...)."""
    text = str(e).casefold()
    if any(phrase in text for phrase in _CONTEXT_LENGTH_PHRASES):
        return OverlongPromptError(str(e))
    return ModelError(str(e))


def content_to_wire(content) -> str | list[ChatCompletionContentPartParam]:
    if isinstance(content, str):
        return content
    parts: list[ChatCompletionContentPartParam] = []
    for part in content:
        if isinstance(part, TextContentPart):
            parts.append(
                ChatCompletionContentPartTextParam(type="text", text=part.text)
            )
        elif part.image_url.detail == "original":
            raise ValueError("OpenAI Chat does not support image detail='original'")
        else:
            parts.append(
                ChatCompletionContentPartImageParam(
                    type="image_url",
                    image_url=cast(Any, part.image_url.model_dump(exclude_none=True)),
                )
            )
    return parts


def message_to_wire(message: Message) -> ChatCompletionMessageParam:
    if isinstance(message, AssistantMessage):
        wire = ChatCompletionAssistantMessageParam(
            role="assistant", content=message.content
        )
        if message.tool_calls:
            wire["tool_calls"] = [
                ChatCompletionMessageFunctionToolCallParam(
                    id=call.id,
                    type="function",
                    function={"name": call.name, "arguments": call.arguments},
                )
                for call in message.tool_calls
            ]
        if message.provider_state:
            cast(Any, wire)["reasoning_details"] = message.provider_state
        return wire
    if isinstance(message, ToolMessage):
        return ChatCompletionToolMessageParam(
            role="tool",
            tool_call_id=message.tool_call_id,
            content=message.content,
        )
    if isinstance(message, SystemMessage):
        content = message.content
        if not isinstance(content, str):
            if any(not isinstance(part, TextContentPart) for part in content):
                raise ValueError("OpenAI Chat system messages do not support images")
            content = [
                ChatCompletionContentPartTextParam(type="text", text=part.text)
                for part in content
                if isinstance(part, TextContentPart)
            ]
        return ChatCompletionSystemMessageParam(role="system", content=content)
    assert isinstance(message, UserMessage)
    return ChatCompletionUserMessageParam(
        role="user",
        content=content_to_wire(message.content),
    )


def tool_to_wire(tool: Tool) -> ChatCompletionFunctionToolParam:
    function = FunctionDefinition(
        name=tool.name,
        description=tool.description,
        parameters=tool.parameters,
    )
    if tool.strict is not None:
        function["strict"] = tool.strict
    return ChatCompletionFunctionToolParam(type="function", function=function)


def tokens_from_wire(completion: ChatCompletion, choice: Choice) -> TurnTokens | None:
    """Parse vLLM's token ids + sampling logprobs into `TurnTokens`, for training.

    vLLM surfaces the completion ids on the choice (`return_token_ids`), the prompt
    ids on the completion, and the sampled logprobs as one `logprobs.content` entry
    per generated token (`logprobs=True`). All are absent on providers that don't
    return them, so this is best-effort: no completion ids means no `tokens`.
    """
    completion_ids = cast(list[int] | None, (choice.model_extra or {}).get("token_ids"))
    if not completion_ids:
        return None
    content = choice.logprobs.content if choice.logprobs else None
    return TurnTokens(
        prompt_ids=cast(
            list[int], (completion.model_extra or {}).get("prompt_token_ids") or []
        ),
        completion_ids=completion_ids,
        completion_logprobs=[lp.logprob for lp in content] if content else [],
    )


def response_from_wire(completion: ChatCompletion) -> Response:
    choice = completion.choices[0]
    message = choice.message
    extra = message.model_extra or {}
    tool_calls = [
        ToolCall(id=tc.id, name=tc.function.name, arguments=tc.function.arguments)
        for tc in (message.tool_calls or [])
        if isinstance(tc, ChatCompletionMessageFunctionToolCall)
    ] or None
    finish = cast(
        FinishReason,
        choice.finish_reason
        if choice.finish_reason in ("stop", "length", "tool_calls")
        else None,
    )
    usage = (
        Usage(
            prompt_tokens=completion.usage.prompt_tokens,
            completion_tokens=completion.usage.completion_tokens,
        )
        if completion.usage
        else None
    )
    return Response(
        id=completion.id,
        created=completion.created,
        model=completion.model,
        message=AssistantMessage(
            content=message.content,
            reasoning_content=cast(
                str | None,
                extra.get("reasoning_content") or extra.get("reasoning"),
            ),
            tool_calls=tool_calls,
            provider_state=cast(
                list[dict[str, Any]] | None, extra.get("reasoning_details")
            ),
        ),
        finish_reason=finish,
        usage=usage,
        tokens=tokens_from_wire(completion, choice),
    )


class OpenAIChatCompletionsClient(Client):
    def __init__(self, openai: AsyncOpenAI) -> None:
        self.openai = openai

    async def get_response(
        self,
        prompt: Messages,
        model: str,
        sampling_args: SamplingConfig,
        tools: list[Tool] | None = None,
    ) -> Response:
        sampling: dict[str, Any] = sampling_args.model_dump(exclude_none=True)
        streaming = bool(sampling.pop("stream", False))
        if streaming:
            # Usage only arrives on the final chunk if asked for.
            sampling["stream_options"] = {
                "include_usage": True,
                **(sampling.get("stream_options") or {}),
            }
        body: dict[str, Any] = {
            "model": model,
            "messages": [message_to_wire(m) for m in prompt],
            **sampling,
        }
        if tools:
            body["tools"] = [tool_to_wire(t) for t in tools]
        try:
            if streaming:
                async with self.openai.chat.completions.stream(**body) as stream:
                    completion = await stream.get_final_completion()
            else:
                completion = await self.openai.chat.completions.create(**body)
        except OpenAIError as e:
            raise model_error(e) from e
        return response_from_wire(completion)

    async def close(self) -> None:
        await self.openai.close()
