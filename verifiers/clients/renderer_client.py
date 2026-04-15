"""Renderer-based client.

All tokenization happens client-side via a Renderer from the renderers package.
For multi-turn rollouts, the client preserves exact sampled completion tokens
and only renders the newly appended environment messages.

A shared RendererPool (one per model) offloads sync tokenization to threads so
concurrent rollouts tokenize in parallel instead of blocking the event loop.
"""

from __future__ import annotations

import asyncio
import json
import threading
from collections.abc import Mapping
from typing import Any, ClassVar, cast

from openai import AsyncOpenAI

from renderers import Message as RendererMessage
from renderers import (
    Renderer,
    RendererPool,
    ToolSpec,
    build_incremental_prompt_ids,
    create_renderer,
)
from renderers import ToolCall as RendererToolCall
from renderers import ToolCallFunction
from renderers.client import completions_request

from verifiers.clients.client import Client
from verifiers.errors import EmptyModelResponseError
from verifiers.types import (
    AssistantMessage,
    ClientConfig,
    FinishReason,
    Message,
    Messages,
    Response,
    ResponseMessage,
    ResponseTokens,
    SamplingArgs,
    SystemMessage,
    TextMessage,
    Tool,
    ToolCall,
    ToolMessage,
    Usage,
    UserMessage,
)
from verifiers.utils.client_utils import setup_openai_client
from verifiers.utils.message_utils import maybe_normalize_messages

_DEFAULT_POOL_SIZE = 32


class RendererClient(
    Client[AsyncOpenAI, list[RendererMessage], dict[str, Any], ToolSpec]
):
    """Client that tokenizes prompts client-side via a Renderer.

    First turn: Renderer renders messages → sends token IDs to vLLM /v1/generate.
    Later turns reuse exact sampled tokens and render only new environment messages.

    A class-level RendererPool (keyed by model) is shared across all instances
    so that concurrent rollouts tokenize in parallel threads.
    """

    _shared_pools: ClassVar[dict[tuple[str, str], RendererPool]] = {}
    _shared_pools_lock: ClassVar[threading.Lock] = threading.Lock()

    def __init__(
        self,
        config: ClientConfig,
        renderer: Renderer | None = None,
        pool_size: int = _DEFAULT_POOL_SIZE,
    ):
        super().__init__(config)
        self._renderer = renderer
        self._pool_size = pool_size

    def setup_client(self, config: ClientConfig) -> AsyncOpenAI:
        return setup_openai_client(config)

    async def close(self) -> None:
        await self.client.close()

    # ── Renderer management ─────────────────────────────────────────

    def _get_renderer_or_pool(self, model: str) -> Renderer | RendererPool:
        if self._renderer is not None:
            return self._renderer

        renderer_name = self._config.renderer if self._config is not None else "auto"
        renderer_model = (
            self._config.renderer_model_name
            if self._config is not None and self._config.renderer_model_name is not None
            else model
        )
        cache_key = (renderer_model, renderer_name)

        with self._shared_pools_lock:
            if cache_key not in self._shared_pools:

                def factory(_name=renderer_name, _model=renderer_model) -> Renderer:
                    from transformers import AutoTokenizer

                    tokenizer = AutoTokenizer.from_pretrained(
                        _model, trust_remote_code=True
                    )
                    return create_renderer(tokenizer, renderer=_name)

                self._shared_pools[cache_key] = RendererPool(
                    factory, size=self._pool_size
                )

        return self._shared_pools[cache_key]

    # ── Type conversions ────────────────────────────────────────────

    async def to_native_prompt(
        self, messages: Messages
    ) -> tuple[list[RendererMessage], dict]:
        messages = maybe_normalize_messages(messages, field_name="prompt")
        return [_to_renderer_message(m) for m in messages], {}

    async def to_native_tool(self, tool: Tool) -> ToolSpec:
        return ToolSpec(
            name=tool.name,
            description=tool.description or "",
            parameters=tool.parameters or {},
        )

    # ── Core request cycle ──────────────────────────────────────────

    async def get_native_response(
        self,
        prompt: list[RendererMessage],
        model: str,
        sampling_args: SamplingArgs,
        tools: list[ToolSpec] | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        renderer = self._get_renderer_or_pool(model)

        args = dict(sampling_args)
        if "max_tokens" in args:
            args["max_completion_tokens"] = args.pop("max_tokens")

        prompt_ids = await _get_incremental_prompt_ids(
            renderer=renderer,
            prompt=prompt,
            state=kwargs.get("state"),
            tools=tools,
        )

        return await completions_request(
            client=self.client,
            renderer=renderer,
            messages=prompt,
            model=model,
            tools=tools,
            prompt_ids=prompt_ids,
            **args,
        )

    async def raise_from_native_response(self, response: dict[str, Any]) -> None:
        if response is None:
            raise EmptyModelResponseError("Model returned no response")

        has_content = bool(response.get("content"))
        has_tool_calls = bool(response.get("tool_calls"))
        has_reasoning = bool(response.get("reasoning_content"))
        if not (has_content or has_tool_calls or has_reasoning):
            raise EmptyModelResponseError(
                "Model returned no content, reasoning, and did not call any tools"
            )

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
                        else json.dumps(tc["function"]["arguments"])
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


# ── Helpers ─────────────────────────────────────────────────────────


async def _run_with_renderer(renderer: Renderer | RendererPool, fn):
    if isinstance(renderer, RendererPool):

        def _work():
            with renderer.checkout() as r:
                return fn(r)

        return await asyncio.to_thread(_work)
    return fn(renderer)


def _get_value(obj: Any, key: str, default: Any = None) -> Any:
    if isinstance(obj, Mapping):
        return obj.get(key, default)
    return getattr(obj, key, default)


def _normalize_for_comparison(value: Any) -> Any:
    if hasattr(value, "model_dump"):
        return _normalize_for_comparison(value.model_dump(exclude_none=True))
    if isinstance(value, Mapping):
        return {
            str(k): _normalize_for_comparison(v)
            for k, v in value.items()
            if v is not None
        }
    if isinstance(value, list):
        return [_normalize_for_comparison(v) for v in value]
    return value


def _coerce_renderer_message(message: Any) -> RendererMessage:
    if isinstance(message, Mapping):
        return cast(
            RendererMessage,
            {
                str(k): _normalize_content(v)
                for k, v in message.items()
                if v is not None
            },
        )
    return _to_renderer_message(cast(Message, message))


def _message_role(message: Any) -> str | None:
    role = _get_value(message, "role")
    return role if isinstance(role, str) else None


def _is_valid_incremental_tail(messages: list[RendererMessage]) -> bool:
    if not messages:
        return False

    roles = [_message_role(message) for message in messages]
    if roles[-1] == "user":
        return all(role == "tool" for role in roles[:-1])
    return all(role == "tool" for role in roles)


def _has_multimodal_content(messages: list[RendererMessage]) -> bool:
    for message in messages:
        content = _get_value(message, "content")
        if not isinstance(content, list):
            continue
        for part in content:
            if _get_value(part, "type") != "text":
                return True
    return False


def _step_is_truncated(step: Any) -> bool:
    if bool(_get_value(step, "is_truncated", False)):
        return True

    tokens = _get_value(step, "tokens")
    if tokens is not None and bool(_get_value(tokens, "is_truncated", False)):
        return True

    response = _get_value(step, "response")
    message = _get_value(response, "message")
    return bool(_get_value(message, "is_truncated", False))


def _step_token_ids(step: Any) -> tuple[list[int], list[int]] | None:
    tokens = _get_value(step, "tokens")
    if tokens is None:
        return None

    prompt_ids = _get_value(tokens, "prompt_ids")
    completion_ids = _get_value(tokens, "completion_ids")
    if not prompt_ids or not completion_ids:
        return None
    return list(prompt_ids), list(completion_ids)


def _step_rendered_messages(step: Any) -> list[RendererMessage]:
    prompt = list(_get_value(step, "prompt", []) or [])
    completion = list(_get_value(step, "completion", []) or [])
    return [_coerce_renderer_message(message) for message in prompt + completion]


async def _get_incremental_prompt_ids(
    *,
    renderer: Renderer | RendererPool,
    prompt: list[RendererMessage],
    state: Any,
    tools: list[ToolSpec] | None,
) -> list[int] | None:
    if not state or _has_multimodal_content(prompt):
        return None

    trajectory = _get_value(state, "trajectory")
    if not trajectory:
        return None

    normalized_prompt = _normalize_for_comparison(prompt)
    for step in reversed(list(trajectory)):
        if _step_is_truncated(step):
            continue

        token_ids = _step_token_ids(step)
        if token_ids is None:
            continue

        previous_messages = _step_rendered_messages(step)
        if not previous_messages or len(previous_messages) >= len(prompt):
            continue
        if _has_multimodal_content(previous_messages):
            continue

        prefix_len = len(previous_messages)
        if normalized_prompt[:prefix_len] != _normalize_for_comparison(
            previous_messages
        ):
            continue

        tail = prompt[prefix_len:]
        if not _is_valid_incremental_tail(tail):
            continue

        previous_prompt_ids, previous_completion_ids = token_ids
        return await _run_with_renderer(
            renderer,
            lambda r: build_incremental_prompt_ids(
                r,
                previous_prompt_ids,
                previous_completion_ids,
                tail,
                tools=tools,
            ),
        )

    return None


def _normalize_content(content: Any) -> Any:
    """Convert Pydantic content parts to plain dicts."""
    if isinstance(content, list):
        return [
            dict(p)
            if isinstance(p, Mapping)
            else cast(dict, p.model_dump())
            if hasattr(p, "model_dump")
            else p
            for p in content
        ]
    return content


def _to_renderer_message(message: Message) -> RendererMessage:
    """Convert a verifiers Message (Pydantic model) to a renderer Message (TypedDict)."""
    if isinstance(message, SystemMessage):
        return RendererMessage(
            role="system", content=_normalize_content(message.content)
        )
    elif isinstance(message, UserMessage):
        return RendererMessage(role="user", content=_normalize_content(message.content))
    elif isinstance(message, AssistantMessage):
        msg = RendererMessage(
            role="assistant",
            content=_normalize_content(message.content),
        )
        if message.reasoning_content is not None:
            msg["reasoning_content"] = message.reasoning_content
        if message.tool_calls is not None:
            msg["tool_calls"] = [
                RendererToolCall(
                    type="function",
                    id=tc.id,
                    function=ToolCallFunction(name=tc.name, arguments=tc.arguments),
                )
                for tc in message.tool_calls
            ]
        return msg
    elif isinstance(message, ToolMessage):
        return RendererMessage(
            role="tool",
            content=_normalize_content(message.content),
            tool_call_id=message.tool_call_id,
        )
    elif isinstance(message, TextMessage):
        return RendererMessage(role="user", content=message.content)
    else:
        raise ValueError(f"Unknown message type: {type(message)}")


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
