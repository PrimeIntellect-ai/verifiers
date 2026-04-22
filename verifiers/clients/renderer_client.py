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
import logging
import threading
from collections.abc import Mapping
from typing import Any, ClassVar, cast

# Dedicated logger for extension-break diagnostics. Set to DEBUG to see the
# two token streams at the divergence point; left at the default WARNING
# so there's no overhead in production.
_incremental_logger = logging.getLogger("verifiers.renderer_client.extension_break")

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

# Size 1 by default. HF fast tokenizers encode a short chat prompt in a few
# tens of microseconds, so even 2k rollouts tokenize serially in ~100ms — far
# cheaper than dispatching each one through asyncio.to_thread and queueing on
# a multi-slot pool. Larger pools mostly just inflate startup time: each slot
# instantiates its own AutoTokenizer (300-600ms each, and GIL-bound, so extra
# workers don't parallelize well). Callers with genuinely long prompts or
# big tokenizers can bump this per-client.
_DEFAULT_POOL_SIZE = 1


class RendererClient(
    Client[AsyncOpenAI, list[RendererMessage], dict[str, Any], ToolSpec]
):
    """Client that tokenizes prompts client-side via a Renderer.

    First turn: Renderer renders messages → sends token IDs to vLLM /v1/generate.
    Later turns reuse exact sampled tokens and render only new environment messages.

    A class-level RendererPool (keyed by model) is shared across all instances
    so that concurrent rollouts tokenize in parallel threads.
    """

    # Cache key is (renderer_model_name, renderer_name, tool_parser,
    # reasoning_parser, pool_size) so that different parser configs or pool
    # sizes for the same model don't collide.
    _shared_pools: ClassVar[dict[tuple[str, str, str | None, str | None, int], RendererPool]] = {}
    _shared_pools_lock: ClassVar[threading.Lock] = threading.Lock()

    def __init__(
        self,
        config: ClientConfig,
        renderer: Renderer | None = None,
        pool_size: int = _DEFAULT_POOL_SIZE,
    ):
        super().__init__(config)
        self._renderer = renderer
        # ClientConfig.renderer_pool_size wins over the constructor default so
        # callers can tune pool size via config without subclassing.
        cfg_size = getattr(config, "renderer_pool_size", None)
        self._pool_size = cfg_size if cfg_size is not None else pool_size

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
        tool_parser = self._config.tool_parser if self._config is not None else None
        reasoning_parser = (
            self._config.reasoning_parser if self._config is not None else None
        )
        cache_key = (renderer_model, renderer_name, tool_parser, reasoning_parser, self._pool_size)

        with self._shared_pools_lock:
            if cache_key not in self._shared_pools:

                def factory(
                    _name=renderer_name,
                    _model=renderer_model,
                    _tool_parser=tool_parser,
                    _reasoning_parser=reasoning_parser,
                ) -> Renderer:
                    from transformers import AutoTokenizer

                    tokenizer = AutoTokenizer.from_pretrained(
                        _model, trust_remote_code=True
                    )
                    return create_renderer(
                        tokenizer,
                        renderer=_name,
                        tool_parser=_tool_parser,
                        reasoning_parser=_reasoning_parser,
                    )

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


def _log_extension_break(
    renderer: Renderer,
    *,
    turn_n: int,
    prev_prompt_ids: list[int],
    prev_completion_ids: list[int],
    full_prompt: list[RendererMessage],
    tools: list[ToolSpec] | None,
) -> None:
    """Emit a DEBUG log showing the two token streams that diverge when the
    bridge trick fails at the start of turn ``turn_n``.

    Stream A = ``prev_prompt_ids + prev_completion_ids`` — the exact tokens
    vLLM saw and sampled at turn N-1. Stream B = ``render_ids(full_prompt,
    add_generation_prompt=True)`` — what the client will now send as a fresh
    render. Their longest common prefix shows how much of the previous
    turn's history the renderer still honors bitwise; tokens after that
    point are where the assistant history got "rewritten".
    """
    try:
        rerender_ids = renderer.render_ids(
            full_prompt, tools=tools, add_generation_prompt=True
        )
    except Exception as exc:  # pragma: no cover — diagnostic only
        _incremental_logger.debug(
            f"extension break at turn {turn_n}: re-render failed ({exc!r}); "
            f"prev_combined_len={len(prev_prompt_ids) + len(prev_completion_ids)}"
        )
        return

    prev_combined = list(prev_prompt_ids) + list(prev_completion_ids)
    max_check = min(len(prev_combined), len(rerender_ids))
    div = 0
    while div < max_check and prev_combined[div] == rerender_ids[div]:
        div += 1

    if div >= len(prev_prompt_ids):
        reason = "renderer rewrote assistant history"
    else:
        reason = "prompt tokens diverged (unusual — investigate tokenizer stability)"

    # Show a small window around the divergence so the log is scannable
    # but not overwhelming. Full arrays are still logged after.
    window = 8
    lo = max(0, div - window)
    hi_prev = min(len(prev_combined), div + window)
    hi_rer = min(len(rerender_ids), div + window)

    prev_window = prev_combined[lo:hi_prev]
    rer_window = rerender_ids[lo:hi_rer]

    tokenizer = getattr(renderer, "_tokenizer", None) or getattr(
        renderer, "tokenizer", None
    )

    def _decode(ids: list[int]) -> str:
        if tokenizer is None:
            return "<tokenizer unavailable>"
        try:
            return tokenizer.decode(ids, skip_special_tokens=False)
        except Exception as exc:  # pragma: no cover
            return f"<decode failed: {exc!r}>"

    _incremental_logger.debug(
        "extension break at turn %d: %s\n"
        "  diverged at token %d "
        "(prev_prompt=%d, prev_completion=%d, rerender=%d)\n"
        "  prev[%d:%d] text:     %r\n"
        "  prev[%d:%d] ids:      %s\n"
        "  rerender[%d:%d] text: %r\n"
        "  rerender[%d:%d] ids:  %s",
        turn_n,
        reason,
        div,
        len(prev_prompt_ids),
        len(prev_completion_ids),
        len(rerender_ids),
        lo,
        hi_prev,
        _decode(prev_window),
        lo,
        hi_prev,
        prev_window,
        lo,
        hi_rer,
        _decode(rer_window),
        lo,
        hi_rer,
        rer_window,
    )


async def _get_incremental_prompt_ids(
    *,
    renderer: Renderer | RendererPool,
    prompt: list[RendererMessage],
    state: Any,
    tools: list[ToolSpec] | None,
) -> list[int] | None:
    if not state:
        return None

    trajectory = _get_value(state, "trajectory")
    if not trajectory:
        return None

    # Each renderer's bridge_to_next_turn (or the generic fallback) decides
    # how to handle a truncated anchor, so we don't special-case truncation
    # here. When the bridge can't extend (e.g. DefaultRenderer without
    # synthesize_close_on_truncation), it returns None and the caller falls
    # back to a full re-render — matching main's TITO-on-truncation behavior.
    trajectory_list = list(trajectory)

    normalized_prompt = _normalize_for_comparison(prompt)
    for step in reversed(trajectory_list):
        token_ids = _step_token_ids(step)
        if token_ids is None:
            continue

        previous_messages = _step_rendered_messages(step)
        if not previous_messages or len(previous_messages) >= len(prompt):
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
        # Prefer a renderer's own bridge_to_next_turn if it defines one. Hand-
        # coded renderers implement this with template-aware logic (e.g. GLM
        # knows its next-turn marker, Qwen3.5 knows its thinking-block rules);
        # DefaultRenderer's implementation delegates to the generic algorithm.
        # Renderers that don't define the method fall back to the generic
        # helper here so we can migrate incrementally.
        bridged = await _run_with_renderer(
            renderer,
            lambda r: (
                r.bridge_to_next_turn(
                    previous_prompt_ids,
                    previous_completion_ids,
                    tail,
                    tools=tools,
                )
                if hasattr(r, "bridge_to_next_turn")
                else build_incremental_prompt_ids(
                    r,
                    previous_prompt_ids,
                    previous_completion_ids,
                    tail,
                    tools=tools,
                )
            ),
        )
        if bridged is None and _incremental_logger.isEnabledFor(logging.DEBUG):
            await _run_with_renderer(
                renderer,
                lambda r: _log_extension_break(
                    r,
                    turn_n=len(list(trajectory)),
                    prev_prompt_ids=list(previous_prompt_ids),
                    prev_completion_ids=list(previous_completion_ids),
                    full_prompt=prompt,
                    tools=tools,
                ),
            )
        return bridged

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
