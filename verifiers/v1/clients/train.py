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

from openai import AsyncOpenAI, OpenAIError
from renderers import OverlongPromptError as RendererOverlongPromptError
from renderers import RendererConfig

from verifiers.v1.clients.client import Client
from verifiers.v1.dialects import FINISH_REASONS, ChatDialect, Dialect
from verifiers.v1.dialects.chat import message_to_wire
from verifiers.v1.errors import OverlongPromptError, model_error
from verifiers.v1.types import (
    AssistantMessage,
    FinishReason,
    Response,
    SamplingConfig,
    Tool,
    ToolCall,
    TurnTokens,
    Usage,
)


def tool_to_wire(tool: Tool) -> dict:
    """A vf tool -> the OpenAI chat wire dict (the renderer's generate request)."""
    function: dict = {
        "name": tool.name,
        "description": tool.description,
        "parameters": tool.parameters,
    }
    if tool.strict is not None:
        function["strict"] = tool.strict
    return {"type": "function", "function": function}


def serialize_completion(response: Response, model: str) -> dict:
    """A vf `Response` -> an OpenAI chat.completion dict the program's SDK expects. The renderer
    sets this on `Response.raw` (it generates, so has no provider response to relay)."""
    message: dict = {"role": "assistant", "content": response.message.content}
    if response.message.reasoning_content is not None:
        message["reasoning_content"] = response.message.reasoning_content
    if response.message.tool_calls:
        message["tool_calls"] = [
            {
                "id": c.id,
                "type": "function",
                "function": {"name": c.name, "arguments": c.arguments},
            }
            for c in response.message.tool_calls
        ]
    usage = (
        {
            "prompt_tokens": response.usage.prompt_tokens,
            "completion_tokens": response.usage.completion_tokens,
            "total_tokens": response.usage.total_tokens,
        }
        if response.usage
        else None
    )
    return {
        "id": response.id or "vf-intercept",
        "object": "chat.completion",
        "created": response.created,
        "model": response.model or model,
        "choices": [
            {
                "index": 0,
                "message": message,
                "finish_reason": response.finish_reason or "stop",
            }
        ],
        "usage": usage,
    }


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
    # Per-message token spans (the renderer's attribution) let the trace graph store each
    # message's tokens once; carried transiently on TurnTokens and consumed by `graph.add_turn`.
    attribution = result.get("prompt_attribution")
    message_spans = (
        attribution.message_token_spans() if attribution is not None else None
    )
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
        headers: Mapping[str, str] | None = None,
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
                tools=[tool_to_wire(t) for t in tools] if tools else None,
                sampling_params=sampling_args.model_dump(exclude_none=True),
            )
        except RendererOverlongPromptError as e:
            raise OverlongPromptError(str(e)) from e
        except OpenAIError as e:
            raise model_error(e) from e
        response = response_from_generate(result, model)
        # No provider response to relay (we generated), so serialize one for the program; the
        # interception server hands `Response.raw` back regardless of client.
        response.raw = serialize_completion(response, model)
        return response

    async def close(self) -> None:
        await self.openai.close()
