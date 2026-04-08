"""Renderer-based client.

All tokenization happens client-side via a Renderer from the renderers package.
No prefix matching, no /tokenize calls, no suffix stitching.
"""

from __future__ import annotations

from typing import Any

from renderers import Renderer, create_renderer
from renderers.client import completions_request

from verifiers.clients.openai_chat_completions_client import (
    OpenAIChatCompletionsClient,
    OpenAIChatMessages,
    OpenAITool,
    handle_openai_overlong_prompt,
)
from verifiers.types import (
    FinishReason,
    Response,
    ResponseMessage,
    ResponseTokens,
    SamplingArgs,
    ToolCall,
    Usage,
)


class RendererClient(OpenAIChatCompletionsClient):
    """Client that tokenizes prompts client-side via a Renderer.

    Every turn: Renderer renders messages → sends token IDs to vLLM /v1/completions
    → gets completion tokens → Renderer parses back to structured message.

    The Renderer is created lazily from the model name on first use.
    """

    def __init__(self, config, renderer: Renderer | None = None):
        super().__init__(config)
        self._renderer = renderer

    def _get_renderer(self, model: str) -> Renderer:
        if self._renderer is None:
            from transformers import AutoTokenizer

            tokenizer = AutoTokenizer.from_pretrained(model, trust_remote_code=True)
            self._renderer = create_renderer(tokenizer)
        return self._renderer

    @handle_openai_overlong_prompt
    async def get_native_response(
        self,
        prompt: OpenAIChatMessages,
        model: str,
        sampling_args: SamplingArgs,
        tools: list[OpenAITool] | None = None,
        **kwargs,
    ) -> dict[str, Any]:
        """Override: render messages → call /v1/generate → return raw result dict."""
        renderer = self._get_renderer(model)

        args = dict(sampling_args)
        if "max_tokens" in args:
            args["max_completion_tokens"] = args.pop("max_tokens")

        result = await completions_request(
            client=self.client,
            renderer=renderer,
            messages=prompt,
            model=model,
            tools=tools,
            **args,
        )
        return result

    async def from_native_response(self, response: dict[str, Any]) -> Response:
        """Parse the completions_request result dict into a verifiers Response."""
        content = response.get("content", "")
        reasoning_content = response.get("reasoning_content")
        finish_reason = _parse_finish_reason(response.get("finish_reason"))

        tool_calls = None
        raw_tcs = response.get("tool_calls")
        if raw_tcs:
            tool_calls = [
                ToolCall(
                    id=f"call_{i}",
                    name=tc["function"]["name"],
                    arguments=(
                        tc["function"]["arguments"]
                        if isinstance(tc["function"]["arguments"], str)
                        else __import__("json").dumps(tc["function"]["arguments"])
                    ),
                )
                for i, tc in enumerate(raw_tcs)
            ]

        prompt_ids = response.get("prompt_ids", [])
        completion_ids = response.get("completion_ids", [])
        completion_logprobs = response.get("completion_logprobs", [])

        tokens = ResponseTokens(
            prompt_ids=prompt_ids,
            prompt_mask=[0] * len(prompt_ids),
            completion_ids=completion_ids,
            completion_mask=[1] * len(completion_ids),
            completion_logprobs=completion_logprobs,
            routed_experts=response.get("routed_experts"),
        )

        usage_data = response.get("usage") or {}
        usage = Usage(
            prompt_tokens=usage_data.get("prompt_tokens", len(prompt_ids)),
            reasoning_tokens=0,
            completion_tokens=usage_data.get("completion_tokens", len(completion_ids)),
            total_tokens=usage_data.get(
                "total_tokens", len(prompt_ids) + len(completion_ids)
            ),
        )

        return Response(
            id=response.get("id", ""),
            created=response.get("created", 0),
            model=response.get("model", ""),
            usage=usage,
            message=ResponseMessage(
                content=content,
                reasoning_content=reasoning_content,
                finish_reason=finish_reason,
                is_truncated=finish_reason == "length",
                tokens=tokens,
                tool_calls=tool_calls,
            ),
        )


def _parse_finish_reason(raw: str | None) -> FinishReason:
    match raw:
        case "stop":
            return "stop"
        case "length":
            return "length"
        case "tool_calls":
            return "tool_calls"
        case _:
            return None
