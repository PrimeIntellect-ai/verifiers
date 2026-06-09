"""Renderer client: client-side tokenization via the `renderers` package.

A drop-in alternative to the chat-completions client: instead of sending messages
as JSON text, it renders them to token ids with a HF chat template and calls a
vLLM `/inference/v1/generate` engine, so every response carries token ids +
sampling logprobs (recorded on the trace's per-turn `tokens`) for training. It
reuses the chat client's wire translation (message/tool shapes are the same), and
needs a running vLLM engine.
"""

import json

from openai import AsyncOpenAI, OpenAIError
from renderers import RendererConfig

from verifiers.v1.clients.client import Client
from verifiers.v1.clients.openai import FINISH_REASONS
from verifiers.v1.clients.openai import message_to_wire as chat_message_to_wire
from verifiers.v1.clients.openai import tool_to_wire
from verifiers.v1.errors import ModelError
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


def message_to_wire(message: Message) -> dict:
    """The chat-completions wire form, plus `reasoning_content` (the chat template
    renders it back in), which the renderer needs but the chat client drops."""
    wire = chat_message_to_wire(message)
    if message.role == "assistant" and message.reasoning_content is not None:
        wire["reasoning_content"] = message.reasoning_content
    return wire


def response_from_generate(result: dict, model: str) -> Response:
    """Parse a `renderers.client.generate` result dict into a typed `Response`,
    mirroring the chat client's `response_from_wire` (plus the token encoding)."""
    finish: FinishReason = (
        result["finish_reason"]
        if result.get("finish_reason") in FINISH_REASONS
        else None
    )
    tool_calls = [
        ToolCall(
            id=tc.id or f"call_{i}",
            name=tc.name,
            arguments=tc.arguments
            if isinstance(tc.arguments, str)
            else json.dumps(tc.arguments or {}),
        )
        for i, tc in enumerate(result.get("tool_calls") or [])
        if getattr(tc, "name", None)
    ] or None
    prompt_ids = result.get("prompt_ids") or []
    completion_ids = result.get("completion_ids") or []
    return Response(
        id=result.get("request_id", ""),
        created=0,
        model=model,
        message=AssistantMessage(
            content=result.get("content") or None,
            reasoning_content=result.get("reasoning_content"),
            tool_calls=tool_calls,
        ),
        finish_reason=finish,
        usage=Usage(
            prompt_tokens=len(prompt_ids), completion_tokens=len(completion_ids)
        ),
        tokens=TurnTokens(
            prompt_ids=prompt_ids,
            completion_ids=completion_ids,
            completion_logprobs=result.get("completion_logprobs") or [],
        ),
    )


class RendererClient(Client):
    """Renders prompts to token ids and calls a vLLM `/inference/v1/generate` engine."""

    def __init__(
        self,
        openai: AsyncOpenAI,
        pool_size: int = 1,
        config: RendererConfig | None = None,
    ) -> None:
        self.openai = openai
        self.pool_size = pool_size
        self.config = config
        self._pool = None

    def _renderer_pool(self, model: str):
        if self._pool is None:
            from renderers import create_renderer_pool

            self._pool = create_renderer_pool(model, self.config, size=self.pool_size)
        return self._pool

    async def get_response(
        self,
        prompt: Messages,
        model: str,
        sampling_args: SamplingConfig,
        tools: list[Tool] | None = None,
    ) -> Response:
        renderer = self._renderer_pool(model)
        from renderers.client import generate

        try:
            result = await generate(
                client=self.openai,
                renderer=renderer,
                messages=[message_to_wire(m) for m in prompt],
                model=model,
                tools=[tool_to_wire(t) for t in tools] if tools else None,
                sampling_params=sampling_args.model_dump(exclude_none=True),
            )
        except OpenAIError as e:
            raise ModelError(str(e)) from e
        return response_from_generate(result, model)

    async def close(self) -> None:
        await self.openai.close()
