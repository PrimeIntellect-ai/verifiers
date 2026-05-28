"""Renderer-based client.

All tokenization happens client-side via a Renderer from the renderers package.
For multi-turn rollouts, the client preserves exact sampled completion tokens
and only renders the newly appended environment messages.

A shared RendererPool (one per model) offloads sync tokenization to threads so
concurrent rollouts tokenize in parallel instead of blocking the event loop.
"""

import ctypes
import gc
import json
import logging
import os
import threading
import time
from collections.abc import Mapping
from contextvars import ContextVar
from multiprocessing import current_process
from typing import Any, ClassVar, cast

from openai import AsyncOpenAI

from renderers import Message as RendererMessage
from renderers import OverlongPromptError as RendererOverlongPromptError
from renderers import (
    AutoRendererConfig,
    MultimodalRenderer,
    ParsedToolCall,
    RenderedTokens,
    Renderer,
    RendererConfig,
    RendererPool,
    ToolSpec,
    config_from_name,
    create_renderer_pool,
    is_multimodal,
)
from renderers import ToolCall as RendererToolCall
from renderers import ToolCallFunction
from renderers.base import MODEL_RENDERER_MAP
import renderers.client as _renderer_client_module
from renderers.client import _maybe_offload, generate

from verifiers.clients.client import Client
from verifiers.clients.openai_chat_completions_client import (
    handle_openai_overlong_prompt,
)
from verifiers.errors import EmptyModelResponseError, OverlongPromptError
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

_mm_logger = logging.getLogger("verifiers.clients.renderer_ephemeral_mm")
_mm_patch_installed = False
_mm_patch_lock = threading.Lock()
_mm_cleanup_lock = threading.Lock()
_mm_libc: Any | None = None
_DEFAULT_MM_CLEANUP_MODE = "auto"
_DEFAULT_MM_CLEANUP_MIN_DELTA_MB = 64.0
_HASH_PREVIEW_CHARS = 8
_HASH_PREVIEW_LIMIT = 8

_mm_force_full_payloads: ContextVar[bool] = ContextVar(
    "renderer_mm_force_full_payloads", default=False
)
_mm_used_hash_only: ContextVar[bool] = ContextVar(
    "renderer_mm_used_hash_only", default=False
)
_mm_full_payload_count: ContextVar[int] = ContextVar(
    "renderer_mm_full_payload_count", default=0
)
_mm_request_id: ContextVar[int | None] = ContextVar("renderer_mm_request_id", default=None)
_mm_log_context: ContextVar[dict[str, Any] | None] = ContextVar(
    "renderer_mm_log_context", default=None
)
_mm_request_counter = 0
_mm_inflight = 0
_mm_counter_lock = threading.Lock()


def get_bridge_metrics() -> dict[str, int]:
    """Snapshot the in-memory bridge counters (attempts/successes/failures)."""
    with _bridge_metrics_lock:
        return dict(_bridge_metrics)


def reset_bridge_metrics() -> None:
    """Zero the in-memory bridge counters."""
    with _bridge_metrics_lock:
        for k in _bridge_metrics:
            _bridge_metrics[k] = 0


# Size 1 by default. HF fast tokenizers encode a short chat prompt in a few
# tens of microseconds, so even 2k rollouts tokenize serially in ~100ms — far
# cheaper than dispatching each one through asyncio.to_thread and queueing on
# a multi-slot pool. Larger pools mostly just inflate startup time: each slot
# instantiates its own AutoTokenizer (300-600ms each, and GIL-bound, so extra
# workers don't parallelize well). Callers with genuinely long prompts or
# big tokenizers can bump this per-client.
_DEFAULT_POOL_SIZE = 1


# ── Helpers ─────────────────────────────────────────────────────────


def _env_flag(name: str) -> str:
    return os.environ.get(name, "").strip().lower()


def _is_off_flag(value: str) -> bool:
    return value in {"0", "false", "off", "disabled", "none", "no"}


def _rss_mb() -> float:
    try:
        with open("/proc/self/status") as f:
            for line in f:
                if line.startswith("VmRSS:"):
                    return int(line.split()[1]) / 1024.0
    except OSError:
        pass
    return 0.0


def _os_thread_count() -> int:
    try:
        return len(os.listdir("/proc/self/task"))
    except OSError:
        return 0


def _cleanup_min_delta_mb() -> float:
    raw = os.environ.get("VF_RENDERER_MM_CLEANUP_MIN_DELTA_MB")
    if raw:
        try:
            return max(0.0, float(raw))
        except ValueError:
            _mm_logger.warning(
                "Ignoring invalid renderer MM cleanup min delta value: %r", raw
            )
    return _DEFAULT_MM_CLEANUP_MIN_DELTA_MB


def _cleanup_mode() -> str:
    return _env_flag("VF_RENDERER_MM_CLEANUP") or _DEFAULT_MM_CLEANUP_MODE


def _maybe_cleanup_mm_request(req_id: int, full_payload_count: int, start_rss: float) -> None:
    """Return transient multimodal payload memory to the OS when possible."""

    mode = _cleanup_mode()
    if _is_off_flag(mode):
        return
    if mode not in {"auto", "always", "1", "true", "on", "enabled"}:
        _mm_logger.warning("Ignoring invalid renderer MM cleanup mode: %r", mode)
        return
    if full_payload_count <= 0 and mode != "always":
        return

    before = _rss_mb()
    if mode == "auto" and before - start_rss < _cleanup_min_delta_mb():
        return

    with _mm_cleanup_lock:
        before = _rss_mb()
        gc_collected = gc.collect()
        after_gc = _rss_mb()
        trim_ok: int | None = None
        trim_error: str | None = None
        try:
            global _mm_libc
            if _mm_libc is None:
                _mm_libc = ctypes.CDLL("libc.so.6")
            trim_ok = int(_mm_libc.malloc_trim(0))
        except Exception as exc:  # pragma: no cover - platform/libc dependent.
            trim_error = type(exc).__name__
        after_trim = _rss_mb()

    _mm_logger.info(
        "renderer_mm_cleanup req=%d mode=%s full_payloads=%d "
        "rss_start_mb=%.1f rss_before_mb=%.1f rss_after_gc_mb=%.1f "
        "rss_after_trim_mb=%.1f freed_mb=%.1f gc_collected=%d "
        "trim_ok=%s trim_error=%s ctx=%s",
        req_id,
        mode,
        full_payload_count,
        start_rss,
        before,
        after_gc,
        after_trim,
        max(0.0, before - after_trim),
        gc_collected,
        "-" if trim_ok is None else trim_ok,
        "-" if trim_error is None else trim_error,
        _mm_context_str(),
    )


def _value_nbytes(value: Any) -> int:
    nbytes = getattr(value, "nbytes", None)
    if isinstance(nbytes, int):
        return nbytes
    element_size = getattr(value, "element_size", None)
    nelement = getattr(value, "nelement", None)
    if callable(element_size) and callable(nelement):
        try:
            return int(element_size() * nelement())
        except Exception:
            return 0
    if isinstance(value, (bytes, bytearray, memoryview, str)):
        return len(value)
    if isinstance(value, Mapping):
        return sum(_value_nbytes(v) for v in value.values())
    if isinstance(value, (list, tuple)):
        return sum(_value_nbytes(v) for v in value)
    return 0


def _mm_data_stats(mm_data: Any) -> tuple[int, float]:
    image_items = getattr(mm_data, "mm_items", {}).get("image") or []
    return len(image_items), _value_nbytes(image_items) / (1024.0 * 1024.0)


def _features_encoded_mb(features: Mapping[str, Any] | None) -> float:
    if not features:
        return 0.0
    kwargs_data = features.get("kwargs_data")
    if not isinstance(kwargs_data, Mapping):
        return 0.0
    return _value_nbytes(kwargs_data) / (1024.0 * 1024.0)


def _preview_hashes(hashes: list[str]) -> str:
    if not hashes:
        return "-"
    clipped = [str(h)[:_HASH_PREVIEW_CHARS] for h in hashes[:_HASH_PREVIEW_LIMIT]]
    suffix = (
        f",+{len(hashes) - _HASH_PREVIEW_LIMIT}"
        if len(hashes) > _HASH_PREVIEW_LIMIT
        else ""
    )
    return ",".join(clipped) + suffix


def _mm_context_str() -> str:
    ctx = _mm_log_context.get() or {}
    parts = [f"pid={os.getpid()}", f"proc={current_process().name}"]
    for key in (
        "session_id",
        "trajectory_id",
        "prior_turns",
        "prompt_msgs",
        "model",
        "cache_salt",
        "client_idx",
    ):
        value = ctx.get(key)
        if value is not None:
            parts.append(f"{key}={value}")
    return " ".join(parts)


def _image_feature_summary(
    features: Mapping[str, Any] | None,
) -> tuple[int, int, str, str]:
    if not features:
        return 0, 0, "-", "-"
    hashes = list((features.get("mm_hashes") or {}).get("image") or [])
    kwargs_data = features.get("kwargs_data")
    hash_only: list[str] = []
    full_payloads: list[str] = []
    if hashes and isinstance(kwargs_data, Mapping):
        image_payloads = kwargs_data.get("image")
        if isinstance(image_payloads, list):
            for i, h in enumerate(hashes):
                if i < len(image_payloads) and image_payloads[i] is not None:
                    full_payloads.append(h)
                else:
                    hash_only.append(h)
    elif hashes and kwargs_data is None:
        hash_only = hashes
    return (
        len(full_payloads),
        len(hash_only),
        _preview_hashes(full_payloads),
        _preview_hashes(hash_only),
    )


def _mark_hash_only_used() -> None:
    _mm_used_hash_only.set(True)


def _increment_full_payload_count(count: int) -> None:
    if count > 0:
        _mm_full_payload_count.set(_mm_full_payload_count.get() + count)


def _materialize_pixels_for_ephemeral_mm(renderer: Any, mm_data: Any, messages: list[Any]):
    from dataclasses import replace

    from renderers.qwen3_vl import _grids_equal, _iter_image_parts

    image_items = getattr(mm_data, "mm_items", {}).get("image") or []
    if not image_items:
        return mm_data
    hashes = list(getattr(mm_data, "mm_hashes", {}).get("image") or [])
    if len(hashes) != len(image_items):
        raise ValueError(
            "materialize_pixels: mm_hashes/mm_items length mismatch "
            f"({len(hashes)} vs {len(image_items)})"
        )
    force_full_payloads = _mm_force_full_payloads.get()

    missing = {
        hashes[i]
        for i, item in enumerate(image_items)
        if force_full_payloads and item.get("pixel_values") is None
    }

    resolved: dict[str, dict[str, Any]] = {}
    if missing:
        for part in _iter_image_parts(messages):
            if not missing:
                break
            _, out, _, h = renderer._process_image(part)
            if h in missing:
                resolved[h] = out
                missing.discard(h)
        if missing:
            raise ValueError(
                f"materialize_pixels: {len(missing)} image hash(es) not "
                "found in messages; cannot reconstruct pixel_values"
            )

    new_image_items: list[dict[str, Any]] = []
    for i, item in enumerate(image_items):
        h = hashes[i]
        if item.get("pixel_values") is not None:
            new_image_items.append(item)
            continue
        if not force_full_payloads:
            _mark_hash_only_used()
            new_image_items.append({k: v for k, v in item.items() if k != "pixel_values"})
            continue
        out = resolved[h]
        if not _grids_equal(out["image_grid_thw"], item.get("image_grid_thw")):
            raise ValueError(
                "materialize_pixels: reconstructed image_grid_thw "
                f"{out['image_grid_thw']!r} != descriptor "
                f"{item.get('image_grid_thw')!r}"
            )
        new_image_items.append(
            {
                "pixel_values": out["pixel_values"],
                "image_grid_thw": out["image_grid_thw"],
            }
        )

    new_items = dict(mm_data.mm_items)
    new_items["image"] = new_image_items
    return replace(mm_data, mm_items=new_items)


def _build_qwen_vl_features_with_hash_only(
    mm_data: Any, *, spatial_merge_size: int
) -> dict[str, Any]:
    try:
        import torch
        from transformers.feature_extraction_utils import BatchFeature
        from vllm.entrypoints.serve.disagg.mm_serde import encode_mm_kwargs_item
        from vllm.model_executor.models.qwen2_vl import _create_qwen2vl_field_factory
        from vllm.multimodal.inputs import MultiModalKwargsItems
    except ImportError as exc:
        raise RuntimeError(
            "Multimodal generate via /inference/v1/generate requires `vllm` "
            "and `torch` to encode the features payload."
        ) from exc

    out: dict[str, Any] = {
        "mm_hashes": {},
        "mm_placeholders": {},
        "kwargs_data": {},
    }

    image_items = mm_data.mm_items.get("image") or []
    if image_items:
        hashes = list(mm_data.mm_hashes.get("image") or [])
        encoded: list[str | None] = [None] * len(image_items)
        encode_indices: list[int] = []
        encode_items: list[dict[str, Any]] = []
        for i, item in enumerate(image_items):
            if item.get("pixel_values") is None:
                _mark_hash_only_used()
                continue
            encode_indices.append(i)
            encode_items.append(item)

        if encode_items:
            pixel_values = torch.cat(
                [torch.as_tensor(it["pixel_values"]) for it in encode_items], dim=0
            )
            image_grid_thw = torch.cat(
                [torch.as_tensor(it["image_grid_thw"]) for it in encode_items], dim=0
            )
            hf_inputs = BatchFeature(
                data={"pixel_values": pixel_values, "image_grid_thw": image_grid_thw}
            )
            config = _create_qwen2vl_field_factory(spatial_merge_size)(hf_inputs)
            kwargs_items = MultiModalKwargsItems.from_hf_inputs(hf_inputs, config)
            encoded_items = [encode_mm_kwargs_item(it) for it in kwargs_items["image"]]
            for i, encoded_item in zip(encode_indices, encoded_items):
                encoded[i] = encoded_item
            _increment_full_payload_count(len(encode_items))

        out["kwargs_data"]["image"] = encoded
        out["mm_hashes"]["image"] = hashes
        out["mm_placeholders"]["image"] = [
            {"offset": p.offset, "length": p.length}
            for p in mm_data.mm_placeholders.get("image") or []
        ]

    if not any(
        any(item is not None for item in items) for items in out["kwargs_data"].values()
    ):
        out["kwargs_data"] = None

    return out


def _install_ephemeral_mm_patch() -> None:
    global _mm_patch_installed
    if _mm_patch_installed:
        return
    with _mm_patch_lock:
        if _mm_patch_installed:
            return

        from renderers.qwen3_vl import Qwen3VLRenderer
        from renderers.qwen35 import Qwen35Renderer

        def wrapped_build_qwen_vl_features(mm_data: Any, *, spatial_merge_size: int):
            images, mm_mb = _mm_data_stats(mm_data)
            t0 = time.monotonic()
            features = _build_qwen_vl_features_with_hash_only(
                mm_data, spatial_merge_size=spatial_merge_size
            )
            elapsed_ms = (time.monotonic() - t0) * 1000.0
            encoded_mb = _features_encoded_mb(features)
            full_payloads, hash_only, full_hashes, hash_only_hashes = (
                _image_feature_summary(features)
            )
            _mm_logger.info(
                "renderer_mm_features req=%s images=%d full=%d hash_only=%d "
                "full_hashes=%s hash_only_hashes=%s mm_mb=%.1f payload_mb=%.1f "
                "rss_mb=%.1f threads=%d elapsed_ms=%.1f ctx=%s",
                _mm_request_id.get() or "-",
                images,
                full_payloads,
                hash_only,
                full_hashes,
                hash_only_hashes,
                mm_mb,
                encoded_mb,
                _rss_mb(),
                _os_thread_count(),
                elapsed_ms,
                _mm_context_str(),
            )
            return features

        _renderer_client_module._build_qwen_vl_features = wrapped_build_qwen_vl_features
        Qwen35Renderer.materialize_pixels = _materialize_pixels_for_ephemeral_mm
        Qwen3VLRenderer.materialize_pixels = _materialize_pixels_for_ephemeral_mm
        _mm_patch_installed = True


def _is_retryable_hash_only_failure(exc: Exception) -> bool:
    status = getattr(exc, "status_code", None)
    response = getattr(exc, "response", None)
    if status is None and response is not None:
        status = getattr(response, "status_code", None)
    if isinstance(status, int) and status >= 500:
        return True
    text = repr(exc)
    return "Expected a cached item" in text or "mm_hash" in text


async def _generate_with_ephemeral_mm_retry(**kwargs: Any) -> dict[str, Any]:
    log_context = kwargs.pop("_mm_log_context", None)
    _install_ephemeral_mm_patch()

    global _mm_request_counter, _mm_inflight
    with _mm_counter_lock:
        _mm_request_counter += 1
        req_id = _mm_request_counter
        _mm_inflight += 1
        inflight = _mm_inflight

    force_token = _mm_force_full_payloads.set(False)
    used_token = _mm_used_hash_only.set(False)
    payload_token = _mm_full_payload_count.set(0)
    req_token = _mm_request_id.set(req_id)
    log_token = _mm_log_context.set(
        cast(dict[str, Any] | None, log_context)
        if isinstance(log_context, dict)
        else None
    )
    start_rss = _rss_mb()
    start = time.monotonic()
    try:
        _mm_logger.debug(
            "renderer_mm_generate_start req=%d inflight=%d rss_mb=%.1f "
            "threads=%d ctx=%s",
            req_id,
            inflight,
            start_rss,
            _os_thread_count(),
            _mm_context_str(),
        )
        try:
            return await generate(**kwargs)
        except Exception as exc:
            retryable = _is_retryable_hash_only_failure(exc)
            used_hash_only = _mm_used_hash_only.get()
            _mm_logger.info(
                "renderer_mm_exception req=%d hash_only=%d retryable=%d "
                "full_payloads=%d exc_type=%s ctx=%s",
                req_id,
                int(used_hash_only),
                int(retryable),
                _mm_full_payload_count.get(),
                type(exc).__name__,
                _mm_context_str(),
            )
            if not (used_hash_only and retryable):
                raise

            _mm_logger.warning(
                "renderer_mm_hash_only_failed req=%d; retrying with all "
                "images materialized: %r ctx=%s",
                req_id,
                exc,
                _mm_context_str(),
            )
            _mm_force_full_payloads.set(True)
            _mm_used_hash_only.set(False)
            _mm_full_payload_count.set(0)
            return await generate(**kwargs)
    finally:
        with _mm_counter_lock:
            _mm_inflight -= 1
            inflight = _mm_inflight
        full_payload_count = _mm_full_payload_count.get()
        _maybe_cleanup_mm_request(req_id, full_payload_count, start_rss)
        _mm_logger.debug(
            "renderer_mm_generate_end req=%d inflight=%d elapsed_s=%.1f "
            "rss_start_mb=%.1f rss_end_mb=%.1f threads=%d full_payloads=%d ctx=%s",
            req_id,
            inflight,
            time.monotonic() - start,
            start_rss,
            _rss_mb(),
            _os_thread_count(),
            full_payload_count,
            _mm_context_str(),
        )
        _mm_force_full_payloads.reset(force_token)
        _mm_used_hash_only.reset(used_token)
        _mm_full_payload_count.reset(payload_token)
        _mm_request_id.reset(req_token)
        _mm_log_context.reset(log_token)


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


def _is_valid_incremental_tail(messages: list[RendererMessage]) -> bool:
    if not messages:
        return False

    roles = []
    for message in messages:
        role = _get_value(message, "role")
        roles.append(role if isinstance(role, str) else None)
    if roles[-1] == "user":
        return all(role == "tool" for role in roles[:-1])
    return all(role == "tool" for role in roles)


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

        step_prompt = list(_get_value(step, "prompt", []) or [])
        step_completion = list(_get_value(step, "completion", []) or [])
        previous_messages = _attach_tool_call_names(
            [
                _coerce_renderer_message(message)
                for message in step_prompt + step_completion
            ]
        )
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
        with _bridge_metrics_lock:
            _bridge_metrics["attempts"] += 1
            _bridge_metrics["successes" if bridged is not None else "failures"] += 1
        return bridged

    return None


def _resolve_renderer_config(
    base: RendererConfig | None,
    chat_template_kwargs: Mapping[str, Any] | None,
    *,
    renderer_model: str,
) -> RendererConfig | None:
    """Merge ``chat_template_kwargs`` into a typed ``RendererConfig``.

    When ``base`` is ``None`` or ``AutoRendererConfig`` (would auto-resolve
    inside ``renderers.create_renderer``), we pull resolution forward via
    ``MODEL_RENDERER_MAP`` so kwargs land on the concrete config variant
    and pydantic validates them against the actual renderer's schema —
    ``AutoRendererConfig`` intentionally carries only ``preserve_*`` and
    would reject template kwargs like ``enable_thinking``. ``renderer_model``
    must match what the pool will tokenize with (i.e.
    ``ClientConfig.renderer_model_name`` when set, else the request model),
    so resolution agrees with the tokenizer the renderer will hold.

    Kwargs override fields with the same name on the (resolved) base.
    Typed configs (``extra="forbid"``) reject unknown keys with a
    field-path error; ``DefaultRendererConfig`` (``extra="allow"``) keeps
    the escape hatch for arbitrary jinja kwargs.
    """
    if not chat_template_kwargs:
        return base

    # Resolve auto → concrete (mirrors ``renderers._resolve_auto``) so
    # ``enable_thinking`` etc. validate against the right schema instead of
    # ``AutoRendererConfig``'s minimal one. Carries ``preserve_*`` across.
    if base is None or isinstance(base, AutoRendererConfig):
        renderer_name = MODEL_RENDERER_MAP.get(renderer_model, "default")
        # ``config_from_name`` returns ``None`` only for ``"auto"``, which
        # ``MODEL_RENDERER_MAP.get(..., "default")`` excludes — assert for ty.
        concrete = config_from_name(renderer_name)
        assert concrete is not None
        if isinstance(base, AutoRendererConfig):
            concrete = concrete.model_copy(
                update={
                    "preserve_all_thinking": base.preserve_all_thinking,
                    "preserve_thinking_between_tool_calls": base.preserve_thinking_between_tool_calls,
                }
            )
        base = cast(RendererConfig, concrete)

    return type(base).model_validate({**base.model_dump(), **chat_template_kwargs})


class RendererClient(
    Client[AsyncOpenAI, list[RendererMessage], dict[str, Any], ToolSpec]
):
    """Client that tokenizes prompts client-side via a Renderer.

    First turn: Renderer renders messages → sends token IDs to vLLM /v1/generate.
    Later turns reuse exact sampled tokens and render only new environment messages.

    A class-level RendererPool (keyed by model) is shared across all instances
    so that concurrent rollouts tokenize in parallel threads.
    """

    # Cache key is ``(renderer_model_name, pool_size, renderer_config_json)``.
    # ``renderer_config`` is a frozen pydantic model so it's hashable directly,
    # but we serialize it via ``model_dump_json()`` for a stable, deterministic
    # key shape that's safe across pydantic version bumps.
    _shared_pools: ClassVar[
        dict[
            tuple[str, int, str | None],
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

    def _get_renderer_or_pool(
        self,
        model: str,
        *,
        renderer_config: RendererConfig | None = None,
    ) -> Renderer | RendererPool:
        if self._renderer is not None:
            return self._renderer

        if renderer_config is None:
            renderer_config = (
                self._config.renderer_config if self._config is not None else None
            )
        renderer_model = (
            self._config.renderer_model_name
            if self._config is not None and self._config.renderer_model_name is not None
            else model
        )
        cfg_key = (
            renderer_config.model_dump_json() if renderer_config is not None else None
        )
        cache_key = (renderer_model, self._pool_size, cfg_key)

        with self._shared_pools_lock:
            if cache_key not in self._shared_pools:
                self._shared_pools[cache_key] = create_renderer_pool(
                    renderer_model,
                    renderer_config,
                    size=self._pool_size,
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
        args = dict(sampling_args)
        extra_headers = {
            **dict(args.pop("extra_headers", None) or {}),
            **dict(kwargs.pop("extra_headers", None) or {}),
        }
        sampling_params: dict[str, Any] = dict(args.pop("extra_body", None) or {})

        # ``chat_template_kwargs`` belong to the renderer, not the engine —
        # peel them off the per-request sampling and fold them into the
        # typed RendererConfig. Pool cache key already includes the
        # effective config so identical kwargs reuse the same renderer.
        chat_template_kwargs = sampling_params.pop("chat_template_kwargs", None)
        # Auto-resolution must agree with the model the pool will tokenize
        # against — ``renderer_model_name`` overrides the request ``model``
        # when set (same precedence ``_get_renderer_or_pool`` uses below).
        renderer_model = (
            self._config.renderer_model_name
            if self._config is not None and self._config.renderer_model_name is not None
            else model
        )
        effective_cfg = _resolve_renderer_config(
            self._config.renderer_config if self._config is not None else None,
            chat_template_kwargs,
            renderer_model=renderer_model,
        )
        renderer = self._get_renderer_or_pool(model, renderer_config=effective_cfg)

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

        state = kwargs.get("state")
        trajectory = _get_value(state, "trajectory", []) if state is not None else []
        prior_turns = len(trajectory) if isinstance(trajectory, list) else None
        session_id = (
            extra_headers.get("X-Session-ID")
            or extra_headers.get("x-session-id")
            or (_get_value(state, "trajectory_id") if state is not None else None)
        )
        cache_salt = args.get("cache_salt") or sampling_params.pop("cache_salt", None)
        priority = args.get("priority") or sampling_params.pop("priority", None)

        bridged = await _get_incremental_prompt_ids(
            renderer=renderer,
            prompt=prompt,
            state=state,
            tools=tools,
        )
        # ``bridged`` is RenderedTokens | None. Unpack token_ids + mm_data
        # (multimodal feature pass-through) and prompt_attribution
        # (per-token mask sidecar). On the first turn (``bridged is None``),
        # ``generate`` renders and emits the attribution itself.
        if bridged is not None:
            prompt_ids = bridged.token_ids
            multi_modal_data = bridged.multi_modal_data
            prompt_attribution = bridged
        else:
            prompt_ids = None
            multi_modal_data = None
            prompt_attribution = None

        # ``renderers.client.generate`` discovers the engine's context-length
        # cap on its own (via ``GET /v1/models``, cached) and raises
        # ``renderers.OverlongPromptError`` on pre-flight overflow. Rebadge
        # that into the verifiers-native ``OverlongPromptError`` so the
        # ``MultiTurnEnv.prompt_too_long`` stop condition picks it up via
        # the ``vf.Error`` hierarchy. The ``@handle_openai_overlong_prompt``
        # decorator still handles the fallback case (cap unknown → engine
        # 4xx → vf.OverlongPromptError) for engines whose ``/v1/models``
        # doesn't expose ``max_model_len``.
        try:
            return await _generate_with_ephemeral_mm_retry(
                client=self.client,
                renderer=renderer,
                messages=prompt,
                model=model,
                prompt_ids=prompt_ids,
                multi_modal_data=multi_modal_data,
                prompt_attribution=prompt_attribution,
                tools=tools,
                sampling_params=sampling_params,
                cache_salt=cache_salt,
                priority=priority,
                extra_headers=extra_headers or None,
                _mm_log_context={
                    "session_id": session_id,
                    "trajectory_id": _get_value(state, "trajectory_id")
                    if state is not None
                    else None,
                    "prior_turns": prior_turns,
                    "prompt_msgs": len(prompt),
                    "model": model,
                    "cache_salt": cache_salt,
                    "client_idx": self._config.client_idx
                    if self._config is not None
                    else None,
                },
            )
        except RendererOverlongPromptError as exc:
            raise OverlongPromptError(str(exc)) from exc

    async def raise_from_native_response(self, response: dict[str, Any]) -> None:
        if response is None:
            raise EmptyModelResponseError("Model returned no response")

        has_content = bool(response.get("content"))
        # ``tool_calls`` is now ``list[ParsedToolCall]`` (renderers >=0.1.8.dev1)
        # — a non-empty list with only malformed attempts still counts as the
        # model having tried to call a tool, so we don't filter by status here.
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
        match response.get("finish_reason"):
            case "stop":
                finish_reason: FinishReason = "stop"
            case "length":
                finish_reason = "length"
            case "tool_calls":
                finish_reason = "tool_calls"
            case _:
                finish_reason = None

        # Forward any ``ParsedToolCall`` with a ``name`` regardless of
        # ``.status``: argument errors surface to the model as tool-role
        # validation messages via the env's ``call_tool`` for self-
        # correction, instead of an empty ``tool_calls`` that terminates
        # the rollout via ``no_tools_called``. ``status`` stays on each
        # call for downstream filtering.
        tool_calls = None
        raw_tcs = response.get("tool_calls") or []
        usable_tcs = [
            tc for tc in raw_tcs if isinstance(tc, ParsedToolCall) and tc.name
        ]
        if usable_tcs:
            tool_calls = [
                ToolCall(
                    id=tc.id or f"call_{i}",
                    name=tc.name or "",
                    arguments=(
                        tc.arguments
                        if isinstance(tc.arguments, str)
                        else json.dumps(tc.arguments or {})
                    ),
                )
                for i, tc in enumerate(usable_tcs)
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
            multi_modal_data=response.get("multi_modal_data"),
            prompt_attribution=response.get("prompt_attribution"),
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
