"""Renderer-based client.

All tokenization happens client-side via a Renderer from the renderers package.
For multi-turn rollouts, the client preserves exact sampled completion tokens
and only renders the newly appended environment messages.

A shared RendererPool (one per model) offloads sync tokenization to threads so
concurrent rollouts tokenize in parallel instead of blocking the event loop.
"""

import asyncio
import json
import threading
from collections.abc import Mapping
from typing import Any, ClassVar, cast

from openai import AsyncOpenAI
from renderers import Message as RendererMessage
from renderers import (
    MultimodalRenderer,
    RenderedTokens,
    Renderer,
    RendererPool,
    ToolCallFunction,
    ToolSpec,
    create_renderer_pool,
    is_multimodal,
)
from renderers import ToolCall as RendererToolCall
from renderers.client import generate

from verifiers.clients.client import Client
from verifiers.clients.openai_chat_completions_client import (
    handle_openai_overlong_prompt,
)
from verifiers.clients.routed_experts import parse_routed_experts
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

# Module-level bridge counters. Incremented by every RendererClient instance
# that tries to stitch a multi-turn prompt; callers (e.g. prime-rl's
# orchestrator) can read and reset these per training step to surface a
# bridge_break_rate metric.
_bridge_metrics_lock = threading.Lock()
_bridge_metrics: dict[str, int] = {"attempts": 0, "successes": 0, "failures": 0}


def get_bridge_metrics() -> dict[str, int]:
    """Snapshot the in-memory bridge counters (attempts/successes/failures)."""
    with _bridge_metrics_lock:
        return dict(_bridge_metrics)


def reset_bridge_metrics() -> None:
    """Zero the in-memory bridge counters."""
    with _bridge_metrics_lock:
        for k in _bridge_metrics:
            _bridge_metrics[k] = 0


def _record_bridge(success: bool) -> None:
    with _bridge_metrics_lock:
        _bridge_metrics["attempts"] += 1
        _bridge_metrics["successes" if success else "failures"] += 1


# Size 1 by default. HF fast tokenizers encode a short chat prompt in a few
# tens of microseconds, so even 2k rollouts tokenize serially in ~100ms — far
# cheaper than dispatching each one through asyncio.to_thread and queueing on
# a multi-slot pool. Larger pools mostly just inflate startup time: each slot
# instantiates its own AutoTokenizer (300-600ms each, and GIL-bound, so extra
# workers don't parallelize well). Callers with genuinely long prompts or
# big tokenizers can bump this per-client.
_DEFAULT_POOL_SIZE = 1


# ── Helpers ─────────────────────────────────────────────────────────


async def _maybe_offload(renderer: Renderer | RendererPool, fn):
    """Run sync renderer work on a thread iff ``renderer`` is a pool.

    Pool methods can block on the internal queue/lock; we offload to keep
    the event loop responsive. A bare ``Renderer`` runs inline.
    """
    if isinstance(renderer, RendererPool):
        return await asyncio.to_thread(fn)
    return fn()


def _get_value(obj: Any, key: str, default: Any = None) -> Any:
    if isinstance(obj, Mapping):
        return obj.get(key, default)
    return getattr(obj, key, default)


def _normalize_for_comparison(value: Any, _key: str | None = None) -> Any:
    # tool_call.arguments is serialized as a string on one side (our trajectory
    # uses json.dumps with default separators) and often comes back from
    # upstream scaffolds re-stringified with JS JSON.stringify (compact, no
    # spaces). Both encode the same dict; parse and normalize structurally so
    # pure-format drift doesn't block incremental prompt matching.
    if _key == "arguments" and isinstance(value, str):
        try:
            value = json.loads(value)
        except (json.JSONDecodeError, ValueError):
            pass
    if hasattr(value, "model_dump"):
        return _normalize_for_comparison(value.model_dump(exclude_none=True))
    if isinstance(value, Mapping):
        # Treat content="" as equivalent to content=None (absent): tool-call-only
        # assistant messages get serialized either way depending on the upstream
        # pipeline (e.g., reasoning parsers strip text content to "" while other
        # paths leave it as None), and the prefix-match must be unaffected.
        return {
            str(k): _normalize_for_comparison(v, _key=str(k))
            for k, v in value.items()
            if v is not None and not (str(k) == "content" and v == "")
        }
    if isinstance(value, list):
        return [_normalize_for_comparison(v) for v in value]
    return value


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


def _attach_tool_call_names(
    messages: list[RendererMessage],
) -> list[RendererMessage]:
    """Fill ``name`` on tool-role messages from prior assistant ``tool_calls``.

    The verifiers ``ToolMessage`` schema has ``role``/``content``/``tool_call_id``
    but no ``name`` field. Some renderers use the function name when emitting
    tool results — notably GPT-OSS Harmony, which prefixes results with
    ``<|start|>functions.{name} to=assistant``. Without recovery, every result
    falls back to ``functions.unknown``.

    We walk the converted-renderer-dict list once, build a ``tool_call_id →
    name`` map from assistant ``tool_calls`` entries, and set ``name`` on
    every subsequent tool message that doesn't already carry one. Validated
    end-to-end on GPT-OSS-20b.
    """
    lookup: dict[str, str] = {}
    for m in messages:
        role = m.get("role") if isinstance(m, Mapping) else None
        if role == "assistant":
            for tc in m.get("tool_calls") or []:
                if not isinstance(tc, Mapping):
                    continue
                tc_id = tc.get("id")
                fn = tc.get("function")
                tc_name = fn.get("name") if isinstance(fn, Mapping) else None
                if isinstance(tc_id, str) and isinstance(tc_name, str):
                    lookup[tc_id] = tc_name
        elif role == "tool" and "name" not in m:
            tcid = m.get("tool_call_id")
            if isinstance(tcid, str):
                name = lookup.get(tcid)
                if name is not None:
                    m["name"] = name
    return messages


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
    # Prefer step.tokens (post-parse_response_tokens) when populated. In
    # multi-turn rollouts, parse_response_tokens zeroes out completion_ids
    # whenever prompt_len > max_seq_len (training-budget enforcement) —
    # that destroys the anchor tokens this lookup needs for bridging. Fall
    # back to the raw response tokens in that case so the bridge can
    # continue to chain across turns; interleave_rollout still enforces
    # training budget at sample-assembly time.
    tokens = _get_value(step, "tokens")
    if tokens is not None:
        prompt_ids = _get_value(tokens, "prompt_ids")
        completion_ids = _get_value(tokens, "completion_ids")
        if prompt_ids and completion_ids:
            return list(prompt_ids), list(completion_ids)

    response = _get_value(step, "response")
    message = _get_value(response, "message")
    raw_tokens = _get_value(message, "tokens")
    if raw_tokens is None:
        return None
    prompt_ids = _get_value(raw_tokens, "prompt_ids")
    completion_ids = _get_value(raw_tokens, "completion_ids")
    if not prompt_ids or not completion_ids:
        return None
    return list(prompt_ids), list(completion_ids)


def _step_multi_modal_data(step: Any):
    """Recover the previous turn's ``MultiModalData`` for bridging.

    Mirrors :func:`_step_token_ids`: prefer ``step.tokens.multi_modal_data``
    (post-parse_response_tokens), fall back to ``step.response.message.tokens``.
    Returns ``None`` when no multimodal sidecar was emitted (text-only
    rollouts) — the bridge handles that branch transparently.
    """
    tokens = _get_value(step, "tokens")
    if tokens is not None:
        mm = _get_value(tokens, "multi_modal_data")
        if mm is not None:
            return mm

    response = _get_value(step, "response")
    message = _get_value(response, "message")
    raw_tokens = _get_value(message, "tokens")
    if raw_tokens is None:
        return None
    return _get_value(raw_tokens, "multi_modal_data")


def _step_rendered_messages(step: Any) -> list[RendererMessage]:
    prompt = list(_get_value(step, "prompt", []) or [])
    completion = list(_get_value(step, "completion", []) or [])
    return _attach_tool_call_names(
        [_coerce_renderer_message(message) for message in prompt + completion]
    )


async def _get_incremental_prompt_ids(
    *,
    renderer: Renderer | RendererPool,
    prompt: list[RendererMessage],
    state: Any,
    tools: list[ToolSpec] | None,
) -> "RenderedTokens | None":
    """Return the bridged prompt for the next turn as ``RenderedTokens``.

    Returns ``None`` when no prior trajectory step lines up with the new
    prompt's prefix or the renderer's ``bridge_to_next_turn`` can't extend
    — both cases fall back to a full re-render in :func:`generate`.
    """
    if not state:
        return None

    trajectory = _get_value(state, "trajectory")
    if not trajectory:
        return None

    # Each renderer's bridge_to_next_turn (or the generic fallback) decides
    # how to handle a truncated anchor, so we don't special-case truncation
    # here. When the bridge can't extend (e.g. DefaultRenderer, which
    # doesn't know its template's close), it returns None and the caller
    # falls back to a full re-render — matching main's TITO-on-truncation
    # behavior.
    normalized_prompt = _normalize_for_comparison(prompt)
    for step in reversed(list(trajectory)):
        token_ids = _step_token_ids(step)
        if token_ids is None:
            continue

        previous_messages = _step_rendered_messages(step)
        if not previous_messages or len(previous_messages) >= len(prompt):
            continue
        prefix_len = len(previous_messages)
        norm_prev = _normalize_for_comparison(previous_messages)
        if normalized_prompt[:prefix_len] != norm_prev:
            continue

        tail = prompt[prefix_len:]
        if not _is_valid_incremental_tail(tail):
            continue

        previous_prompt_ids, previous_completion_ids = token_ids
        previous_mm_data = _step_multi_modal_data(step)
        # Multimodal renderers' bridge accepts ``previous_multi_modal_data``
        # so earlier-turn images carry forward into the new prompt's
        # ``mm_placeholders``. Without that carry-forward, vLLM sees
        # placeholder counts that don't match the combined token sequence
        # and silently falls back to hash-cache lookup (or errors).
        # Text-only renderers' bridge signature doesn't include that
        # kwarg. ``is_multimodal`` is type-cached so this dispatch is a
        # dict lookup, not a runtime_checkable Protocol walk.
        if is_multimodal(renderer):
            mm_renderer = cast(MultimodalRenderer, renderer)
            bridge = lambda: mm_renderer.bridge_to_next_turn(  # noqa: E731
                previous_prompt_ids,
                previous_completion_ids,
                tail,
                tools=tools,
                previous_multi_modal_data=previous_mm_data,
            )
        else:
            bridge = lambda: renderer.bridge_to_next_turn(  # noqa: E731
                previous_prompt_ids,
                previous_completion_ids,
                tail,
                tools=tools,
            )
        bridged = await _maybe_offload(renderer, bridge)
        _record_bridge(success=bridged is not None)
        return bridged

    return None


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
    # reasoning_parser, pool_size, preserve_all_thinking,
    # preserve_thinking_between_tool_calls) so that different parser configs,
    # pool sizes, or preserve-thinking bindings for the same model don't
    # collide.
    _shared_pools: ClassVar[
        dict[
            tuple[str, str, str | None, str | None, int, bool, bool],
            RendererPool,
        ]
    ] = {}
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
        preserve_all_thinking = (
            self._config.preserve_all_thinking if self._config is not None else False
        )
        preserve_thinking_between_tool_calls = (
            self._config.preserve_thinking_between_tool_calls
            if self._config is not None
            else False
        )
        cache_key = (
            renderer_model,
            renderer_name,
            tool_parser,
            reasoning_parser,
            self._pool_size,
            preserve_all_thinking,
            preserve_thinking_between_tool_calls,
        )

        with self._shared_pools_lock:
            if cache_key not in self._shared_pools:
                self._shared_pools[cache_key] = create_renderer_pool(
                    renderer_model,
                    renderer=renderer_name,
                    size=self._pool_size,
                    tool_parser=tool_parser,
                    reasoning_parser=reasoning_parser,
                    preserve_all_thinking=preserve_all_thinking,
                    preserve_thinking_between_tool_calls=preserve_thinking_between_tool_calls,
                )

        return self._shared_pools[cache_key]

    # ── Type conversions ────────────────────────────────────────────

    async def to_native_prompt(
        self, messages: Messages
    ) -> tuple[list[RendererMessage], dict]:
        return (
            _attach_tool_call_names([_to_renderer_message(m) for m in messages]),
            {},
        )

    async def to_native_tool(self, tool: Tool) -> ToolSpec:
        function: dict[str, Any] = {
            "name": tool.name,
            "description": tool.description,
            "parameters": tool.parameters,
        }
        if tool.strict is not None:
            function["strict"] = tool.strict
        return cast(ToolSpec, {"type": "function", "function": function})

    # ── Core request cycle ──────────────────────────────────────────

    @handle_openai_overlong_prompt
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
        sampling_params: dict[str, Any] = dict(args.pop("extra_body", None) or {})
        for key in (
            "temperature",
            "top_p",
            "top_k",
            "min_p",
            "seed",
            "n",
            "repetition_penalty",
            "min_tokens",
        ):
            if args.get(key) is not None:
                sampling_params[key] = args[key]
        max_tokens = args.get("max_tokens") or args.get("max_completion_tokens")
        if max_tokens is not None:
            sampling_params["max_tokens"] = max_tokens
        if args.get("prompt_logprobs"):
            sampling_params["prompt_logprobs"] = 1

        bridged = await _get_incremental_prompt_ids(
            renderer=renderer,
            prompt=prompt,
            state=kwargs.get("state"),
            tools=tools,
        )
        # ``bridged`` is RenderedTokens | None. Unpack token_ids + mm_data
        # so multimodal renderers thread per-image features through to
        # /inference/v1/generate without re-rendering the whole turn.
        if bridged is not None:
            prompt_ids = bridged.token_ids
            multi_modal_data = bridged.multi_modal_data
        else:
            prompt_ids = None
            multi_modal_data = None

        return await generate(
            client=self.client,
            renderer=renderer,
            messages=prompt,
            model=model,
            prompt_ids=prompt_ids,
            multi_modal_data=multi_modal_data,
            tools=tools,
            sampling_params=sampling_params,
            cache_salt=args.get("cache_salt")
            or sampling_params.pop("cache_salt", None),
            priority=args.get("priority") or sampling_params.pop("priority", None),
            extra_headers=args.get("extra_headers"),
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
        """Parse the generate() result dict into a verifiers Response."""
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

        routed_experts = parse_routed_experts(response.get("routed_experts"))

        tokens = ResponseTokens(
            prompt_ids=prompt_ids,
            prompt_mask=[0] * len(prompt_ids),
            completion_ids=completion_ids,
            completion_mask=[1] * len(completion_ids),
            completion_logprobs=completion_logprobs,
            routed_experts=routed_experts,
            multi_modal_data=response.get("multi_modal_data"),
        )

        # /inference/v1/generate doesn't return usage; reconstruct from tokens.
        usage = Usage(
            prompt_tokens=len(prompt_ids),
            reasoning_tokens=0,
            completion_tokens=len(completion_ids),
            total_tokens=len(prompt_ids) + len(completion_ids),
        )

        return Response(
            id=response.get("request_id", ""),
            created=0,
            model="",
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
