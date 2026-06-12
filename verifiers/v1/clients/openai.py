"""OpenAI-compatible chat-completions client.

Distilled from v1's 545-line client: message<->wire translation, tool schemas,
best-effort reasoning_content. Sampling args pass straight through; when the
response carries vLLM's token ids + sampling logprobs (the caller asked for
`logprobs` and `return_token_ids`), we parse them into the response's `tokens`
so MITO training needs no renderer. Routed-experts/audio handling stays dropped.
This is the one place raw provider dicts cross into our typed `Response`.
"""

from openai import AsyncOpenAI, OpenAIError

from verifiers.v1.clients.client import Client
from verifiers.v1.errors import ModelError, OverlongPromptError
from verifiers.v1.types import (
    AssistantMessage,
    FinishReason,
    Message,
    Messages,
    Response,
    SamplingConfig,
    Tool,
    ToolCall,
    TurnTokens,
    Usage,
)

FINISH_REASONS = frozenset({"stop", "length", "tool_calls"})

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


def _content_to_wire(content):
    """Plain text passes through; a content-part list becomes OpenAI wire dicts (so the
    provider / renderer sees the native `image_url` shape)."""
    if isinstance(content, str):
        return content
    return [part.model_dump() for part in content]


def message_to_wire(message: Message) -> dict:
    if message.role == "assistant":
        wire: dict = {"role": "assistant", "content": message.content}
        # Reasoning models (DeepSeek V4, Kimi K2 Thinking, ...) require the prior turns'
        # `reasoning_content` to be sent back as a message-level field — stripping it breaks
        # multi-turn ("reasoning_content ... must be passed back to the API"). Carry it
        # through whenever the model produced it; providers that don't use it ignore it.
        if message.reasoning_content is not None:
            wire["reasoning_content"] = message.reasoning_content
        if message.tool_calls:
            wire["tool_calls"] = [
                {
                    "id": call.id,
                    "type": "function",
                    "function": {"name": call.name, "arguments": call.arguments},
                }
                for call in message.tool_calls
            ]
        return wire
    if message.role == "tool":
        return {
            "role": "tool",
            "tool_call_id": message.tool_call_id,
            "content": message.content,
        }
    return {"role": message.role, "content": _content_to_wire(message.content)}


def tool_to_wire(tool: Tool) -> dict:
    function: dict = {
        "name": tool.name,
        "description": tool.description,
        "parameters": tool.parameters,
    }
    if tool.strict is not None:
        function["strict"] = tool.strict
    return {"type": "function", "function": function}


def tokens_from_wire(completion, choice) -> TurnTokens | None:
    """Parse vLLM's token ids + sampling logprobs into `TurnTokens`, for training.

    vLLM surfaces the completion ids on the choice (`return_token_ids`), the prompt
    ids on the completion, and the sampled logprobs as one `logprobs.content` entry
    per generated token (`logprobs=True`). All are absent on providers that don't
    return them, so this is best-effort: no completion ids means no `tokens`.
    """
    completion_ids = getattr(choice, "token_ids", None)
    if not completion_ids:
        return None
    content = choice.logprobs.content if choice.logprobs else None
    return TurnTokens(
        prompt_ids=list(getattr(completion, "prompt_token_ids", None) or []),
        completion_ids=list(completion_ids),
        completion_logprobs=[lp.logprob for lp in content] if content else [],
    )


def response_from_wire(completion) -> Response:
    choice = completion.choices[0]
    message = choice.message
    tool_calls = [
        ToolCall(id=tc.id, name=tc.function.name, arguments=tc.function.arguments)
        for tc in (message.tool_calls or [])
    ] or None
    finish: FinishReason = (
        choice.finish_reason if choice.finish_reason in FINISH_REASONS else None
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
            reasoning_content=getattr(message, "reasoning_content", None),
            tool_calls=tool_calls,
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
        body: dict = {
            "model": model,
            "messages": [message_to_wire(m) for m in prompt],
            **sampling_args.model_dump(exclude_none=True),
        }
        if tools:
            body["tools"] = [tool_to_wire(t) for t in tools]
        try:
            completion = await self.openai.chat.completions.create(**body)
        except OpenAIError as e:
            raise model_error(e) from e
        return response_from_wire(completion)

    async def close(self) -> None:
        await self.openai.close()
