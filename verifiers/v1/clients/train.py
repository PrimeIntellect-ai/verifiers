"""Renderer client: client-side tokenization via the `renderers` package.

A drop-in alternative to the chat-completions client: instead of sending messages
as JSON text, it renders them to token ids with a HF chat template and calls a
vLLM `/inference/v1/generate` engine, so every response carries token ids +
sampling logprobs (recorded on the trace's per-turn `tokens`) for training. It
reuses the chat client's wire translation (message/tool shapes are the same), and
needs a running vLLM engine.
"""

import json
from collections.abc import Mapping
from typing import Any, cast

from openai import AsyncOpenAI, OpenAIError
from openai.types.chat import (
    ChatCompletion,
    ChatCompletionFunctionToolParam,
    ChatCompletionMessage,
)
from openai.types.chat.chat_completion import Choice
from openai.types.chat.chat_completion_message_function_tool_call import (
    ChatCompletionMessageFunctionToolCall,
    Function,
)
from openai.types.completion_usage import CompletionUsage
from openai.types.shared_params import FunctionDefinition
from renderers import OverlongPromptError as RendererOverlongPromptError
from renderers import RendererConfig, ToolSpec

from verifiers.v1.clients.client import Client
from verifiers.v1.dialects import FINISH_REASONS, ChatDialect, Dialect
from verifiers.v1.dialects.chat import message_to_wire, response_from_wire
from verifiers.v1.errors import OverlongPromptError, model_error
from verifiers.v1.types import (
    FinishReason,
    Response,
    SamplingConfig,
    Tool,
    TurnTokens,
)


def tool_to_renderer(tool: Tool) -> ToolSpec:
    """A vf tool -> the OpenAI tool envelope accepted by the renderer."""
    function = FunctionDefinition(
        name=tool.name,
        description=tool.description,
        parameters=tool.parameters,
    )
    if tool.strict is not None:
        function["strict"] = tool.strict
    return cast(
        ToolSpec,
        ChatCompletionFunctionToolParam(type="function", function=function),
    )


def completion_from_generate(
    result: dict[str, Any], model: str
) -> tuple[ChatCompletion, TurnTokens]:
    """Convert the renderer result into the native chat response plus training tokens."""
    tool_calls = [
        ChatCompletionMessageFunctionToolCall(
            id=tc.id or f"call_{i}",
            type="function",
            function=Function(
                name=tc.name,
                arguments=tc.arguments
                if isinstance(tc.arguments, str)
                else json.dumps(tc.arguments or {}),
            ),
        )
        for i, tc in enumerate(result.get("tool_calls") or [])
        if tc.name
    ] or None
    prompt_ids = result.get("prompt_ids") or []
    completion_ids = result.get("completion_ids") or []
    attribution = result.get("prompt_attribution")
    message_spans = (
        attribution.message_token_spans() if attribution is not None else None
    )
    finish_reason: FinishReason = (
        result["finish_reason"]
        if result.get("finish_reason") in FINISH_REASONS
        else None
    )
    completion = ChatCompletion(
        id=result.get("request_id", ""),
        choices=[
            Choice(
                index=0,
                logprobs=None,
                finish_reason=finish_reason or "stop",
                message=ChatCompletionMessage(
                    role="assistant",
                    content=result.get("content") or None,
                    reasoning_content=result.get("reasoning_content"),
                    tool_calls=tool_calls,
                ),
            )
        ],
        created=0,
        model=model,
        object="chat.completion",
        usage=CompletionUsage(
            prompt_tokens=len(prompt_ids),
            completion_tokens=len(completion_ids),
            total_tokens=len(prompt_ids) + len(completion_ids),
        ),
    )
    return (
        completion,
        TurnTokens(
            prompt_ids=prompt_ids,
            completion_ids=completion_ids,
            completion_logprobs=result.get("completion_logprobs") or [],
            message_spans=message_spans,
            multi_modal_data=result.get("multi_modal_data"),
        ),
    )


class TrainClient(Client):
    """Renders prompts to token ids and calls a vLLM `/inference/v1/generate` engine."""

    def __init__(
        self,
        openai: AsyncOpenAI,
        pool_size: int = 1,
        config: RendererConfig | None = None,
        renderer_model_name: str | None = None,
    ) -> None:
        self.openai = openai
        self.pool_size = pool_size
        self.config = config
        self.renderer_model_name = renderer_model_name
        self._pool = None

    def _renderer_pool(self, model: str):
        if self._pool is None:
            from renderers import create_renderer_pool

            self._pool = create_renderer_pool(
                self.renderer_model_name or model, self.config, size=self.pool_size
            )
        return self._pool

    async def get_response(
        self,
        dialect: Dialect,
        body: dict,
        model: str,
        sampling_args: SamplingConfig,
        _request_headers: Mapping[str, str] | None = None,
    ) -> Response:
        # The renderer tokenizes the typed prompt for training (it needs per-token ids + logprobs
        # back), so it can't forward the raw request — it parses `body` via the dialect and renders
        # it with a chat template. It leaves `Response.raw` unset; the interception server serializes
        # its `Response` for the program instead of relaying provider bytes.
        if not isinstance(dialect, ChatDialect):
            # The renderer renders a chat template, so it's only validated for chat-completions
            # input; other dialects' semantics (Responses reasoning items, Anthropic thinking) may
            # not round-trip faithfully through chat-template tokenization. Refuse them explicitly.
            raise NotImplementedError(
                f"The renderer client only supports the chat-completions dialect, got "
                f"{type(dialect).__name__}. Use the proxy client for this dialect, or add "
                f"renderer support for it."
            )
        prompt, tools = dialect.parse_request(body)
        renderer = self._renderer_pool(model)
        from renderers.client import generate

        try:
            result = await generate(
                client=self.openai,
                renderer=renderer,
                messages=[message_to_wire(m) for m in prompt],
                model=model,
                tools=[tool_to_renderer(t) for t in tools] if tools else None,
                sampling_params=sampling_args.model_dump(exclude_none=True),
            )
        except RendererOverlongPromptError as e:
            raise OverlongPromptError(str(e)) from e
        except OpenAIError as e:
            raise model_error(e) from e
        completion, tokens = completion_from_generate(result, model)
        response = response_from_wire(completion)
        # The wire response needs OpenAI fallbacks, while the trace keeps the renderer's original
        # empty id and unknown finish reason semantics.
        response.message.reasoning_content = result.get("reasoning_content")
        response.finish_reason = (
            result["finish_reason"]
            if result.get("finish_reason") in FINISH_REASONS
            else None
        )
        response.tokens = tokens
        response.raw = completion.model_dump(exclude_none=True)
        response.raw["id"] = response.raw["id"] or "vf-intercept"
        response.raw["choices"][0]["message"]["content"] = response.message.content
        return response

    async def close(self) -> None:
        await self.openai.close()
