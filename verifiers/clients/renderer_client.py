"""Renderer-based client.

All tokenization happens client-side via a Renderer from the renderers package.
For multi-turn rollouts, the client preserves exact sampled completion tokens
and only renders the newly appended environment messages.

A shared RendererPool (one per model) offloads sync tokenization to threads so
concurrent rollouts tokenize in parallel instead of blocking the event loop.
"""

import asyncio
import base64
import ctypes
import gc
import hashlib
import json
import logging
import os
import threading
from collections.abc import Mapping
from multiprocessing import current_process
from pathlib import Path
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
from renderers.client import _maybe_offload, generate
from renderers.mm_store import image_asset_dir, run_id_from_env

from verifiers.clients.client import Client
from verifiers.clients.openai_chat_completions_client import (
    handle_openai_overlong_prompt,
)
from verifiers.errors import EmptyModelResponseError, OverlongPromptError
from verifiers.utils.loop_debug import looptime

# Instrument the sync-on-loop multimodal feature build inside renderers.generate
# (``_build_mm_features`` base64-encodes pixel kwargs and is NOT offloaded — it
# runs on the event loop per request, a prime on-loop-block suspect). Monkeypatch
# it here (module-level fn, looked up by name in generate) so its per-request loop
# time shows up as ``looptime rdr_build_mm_features`` without editing renderers.
# Fully defensive: any failure (renderers version drift / rename) is swallowed so
# it can NEVER break the env-worker import.
try:
    import renderers.client as _renderers_client

    if not getattr(_renderers_client, "_vf_looptime_patched", False):
        _vf_orig_build_mm = getattr(_renderers_client, "_build_mm_features", None)
        if callable(_vf_orig_build_mm):
            def _vf_timed_build_mm(*args, _orig=_vf_orig_build_mm, **kwargs):
                with looptime("rdr_build_mm_features"):
                    return _orig(*args, **kwargs)

            _renderers_client._build_mm_features = _vf_timed_build_mm
        _renderers_client._vf_looptime_patched = True
        logging.getLogger("vf.looptime").info(
            "renderers._build_mm_features looptime patch: %s",
            "applied" if callable(_vf_orig_build_mm) else "skipped (fn not found)",
        )

    # FIX: _build_mm_features runs SYNC on the event loop inside generate
    # (~1.2s/render, measured). Offload it by source-patching generate's call site
    # (sync -> ``await asyncio.to_thread``). Can't be done by function-monkeypatch
    # (the call site is sync). Fully defensive: on any failure we keep the original
    # generate, so the worst case is the pre-fix behaviour (no crash, no regression).
    if not getattr(_renderers_client, "_vf_generate_offload_patched", False):
        import asyncio as _vf_aio
        import inspect as _vf_inspect
        import textwrap as _vf_textwrap

        _vf_needle = "_build_mm_features(renderer, mm_data)"
        _vf_src = _vf_textwrap.dedent(_vf_inspect.getsource(_renderers_client.generate))
        if _vf_needle in _vf_src and "to_thread(_build_mm_features" not in _vf_src:
            _vf_src = _vf_src.replace(
                _vf_needle,
                "(await asyncio.to_thread(_build_mm_features, renderer, mm_data))",
            )
            exec(
                compile(_vf_src, _renderers_client.__file__, "exec"),
                _renderers_client.__dict__,
            )
            if _vf_aio.iscoroutinefunction(_renderers_client.generate):
                globals()["generate"] = _renderers_client.generate
                _renderers_client._vf_generate_offload_patched = True
                logging.getLogger("vf.looptime").warning(
                    "renderers.generate _build_mm_features offload: APPLIED"
                )
            else:
                logging.getLogger("vf.looptime").warning(
                    "renderers.generate offload: skipped (not a coroutine after patch)"
                )
        else:
            logging.getLogger("vf.looptime").warning(
                "renderers.generate offload: skipped (call site not found)"
            )
except Exception as _exc:  # pragma: no cover - never break import
    logging.getLogger("vf.looptime").warning(
        "renderers looptime monkeypatch skipped: %r", _exc
    )

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
_mm_cleanup_lock = threading.Lock()
_mm_libc: Any | None = None
_DEFAULT_MM_CLEANUP_MODE = "auto"
_DEFAULT_MM_CLEANUP_MIN_DELTA_MB = 64.0

_mm_request_counter = 0
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


# Default to a single slot. Multimodal prompts process image pixels per turn and
# can benefit from a larger pool, but each extra slot multiplies renderer/tokenizer
# memory per env-worker process, so keep the pool opt-in: image-heavy, high-inflight
# runs set ClientConfig.renderer_pool_size (or orchestrator.pool_size) explicitly.
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


def _maybe_cleanup_mm_request(
    req_id: int, built_full: bool, start_rss: float, context: Mapping[str, Any]
) -> None:
    """Return transient multimodal payload memory to the OS when possible.

    ``built_full`` is whether this request actually serialized any full pixel
    payload (a new-turn image or a materialize-all fallback). Hash-only-only
    requests allocate no large transient tensors, so in ``auto`` mode we skip
    the trim for them.
    """

    mode = _cleanup_mode()
    if _is_off_flag(mode):
        return
    if mode not in {"auto", "always", "1", "true", "on", "enabled"}:
        _mm_logger.warning("Ignoring invalid renderer MM cleanup mode: %r", mode)
        return
    if not built_full and mode != "always":
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
        "renderer_mm_cleanup req=%d mode=%s built_full=%d "
        "rss_start_mb=%.1f rss_before_mb=%.1f rss_after_gc_mb=%.1f "
        "rss_after_trim_mb=%.1f freed_mb=%.1f gc_collected=%d "
        "trim_ok=%s trim_error=%s ctx=%s",
        req_id,
        mode,
        int(built_full),
        start_rss,
        before,
        after_gc,
        after_trim,
        max(0.0, before - after_trim),
        gc_collected,
        "-" if trim_ok is None else trim_ok,
        "-" if trim_error is None else trim_error,
        _format_context(context),
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


def _mm_sidecar_stats(mm_data: Any) -> tuple[int, int, float]:
    if mm_data is None:
        return 0, 0, 0.0
    image_items = getattr(mm_data, "mm_items", {}).get("image") or []
    descriptor_count = 0
    payload_count = 0
    payload_bytes = 0
    for item in image_items:
        if not isinstance(item, Mapping):
            continue
        if item.get("pixel_values") is None:
            descriptor_count += 1
        else:
            payload_count += 1
            payload_bytes += _value_nbytes(item.get("pixel_values"))
            payload_bytes += _value_nbytes(item.get("image_grid_thw"))
    return descriptor_count, payload_count, payload_bytes / (1024.0 * 1024.0)


# ── Live image offload ───────────────────────────────────────────────────
# Screenshots arrive as ``data:image/...;base64,...`` URLs. Decoding them to a
# shared dir during the live rollout (and rewriting the URL to ``file://``)
# keeps the env worker from retaining the full base64 transcript across turns —
# the unbounded growth lives in ``state["trajectory"]``, which accumulates every
# turn's prompt. Renderers load ``file://`` paths transparently. The prompt is
# stored into ``state["trajectory"]`` after the request, so each trajectory step
# keeps the cheap references that were created while it was the live prompt.

_IMAGE_OFFLOAD_MODE_ENV = "VF_RENDERER_IMAGE_OFFLOAD"
_FILE_URL_PREFIX = "file://"
_MEDIA_TYPE_EXT = {
    "jpeg": ".jpg",
    "jpg": ".jpg",
    "png": ".png",
    "webp": ".webp",
    "gif": ".gif",
}


def _image_offload_enabled() -> bool:
    return not _is_off_flag(_env_flag(_IMAGE_OFFLOAD_MODE_ENV))


def _image_offload_dir() -> Path:
    # The env worker runs in a separate pod that can't inherit the orchestrator's
    # env, but the platform injects RUN_ID into every container — so derive the
    # shared run dir from it. Must equal the orchestrator's ``config.output_dir``
    # (``/data/outputs/run_<RUN_ID>/assets/images`` in prod), where
    # ``offload_images_to_disk`` writes and ``materialize_pixels`` reads back by
    # hash. Absolute by construction → the ``file://`` URL is always well-formed.
    return image_asset_dir(run_id_from_env())


def _media_type_ext(media_type: str) -> str:
    subtype = media_type.split("/", 1)[-1].split(";", 1)[0].strip().lower()
    return _MEDIA_TYPE_EXT.get(subtype, ".img")


def _offload_image_url(url: Any, offload_dir: Path) -> "tuple[str, int] | None":
    """Decode a base64 image data URI to ``offload_dir`` and return
    ``(file_url, decoded_bytes)``. Returns ``None`` for anything that isn't a
    base64 image data URI (already ``file://`` / a path / http, or non-image),
    leaving the caller's URL untouched.

    Content-addressed by ``sha256(decoded_bytes)`` so identical images share one
    file (and one path) regardless of which writer or turn produced them. Writes
    via a unique temp file + atomic ``os.replace`` so concurrent env workers on a
    shared (NFS) dir never see a partial file.
    """
    if not isinstance(url, str) or not url.startswith("data:image/"):
        return None
    marker = ";base64,"
    if marker not in url:
        return None
    header, b64 = url.split(marker, 1)
    media_type = header[len("data:") :]  # e.g. "image/jpeg"
    try:
        raw = base64.b64decode(b64)
    except Exception:
        return None
    digest = hashlib.sha256(raw).hexdigest()[:16]
    path = offload_dir / f"{digest}{_media_type_ext(media_type)}"
    if not path.exists():
        try:
            offload_dir.mkdir(parents=True, exist_ok=True)
            tmp = path.with_name(
                f".{path.name}.{os.getpid()}.{threading.get_ident()}.tmp"
            )
            tmp.write_bytes(raw)
            os.replace(tmp, path)  # atomic; last writer wins (identical content)
        except OSError as exc:
            # Best effort: if the shared dir isn't writable, leave the data URI
            # in place rather than dropping the image.
            _mm_logger.warning("renderer_image_offload write failed: %r", exc)
            return None
    else:
        # Recurring image already on disk: refresh mtime so a future last-use
        # sweep treats it as hot (consistent with the mm_feature writer). Images
        # aren't evicted today; best-effort, ignore a concurrent-sweep race.
        try:
            path.touch()
        except OSError:
            pass
    return f"{_FILE_URL_PREFIX}{path}", len(raw)


def _offload_image_parts_inplace(value: Any, offload_dir: Path) -> "tuple[int, int]":
    """Rewrite every base64 image URL reachable from ``value`` to ``file://`` in
    place; return ``(images_rewritten, bytes_offloaded)``.

    Handles plain-dict messages/parts (the native renderer prompt) and Pydantic
    ``Message`` / ``ContentPart`` models (stored trajectory prompts): for dicts
    we mutate ``item["image_url"]["url"]``; for content-part objects we set
    ``part.image_url.url``; messages are descended via their ``content`` list.
    """
    if isinstance(value, dict):
        count = nbytes = 0
        if value.get("type") == "image_url" and isinstance(
            value.get("image_url"), dict
        ):
            res = _offload_image_url(value["image_url"].get("url"), offload_dir)
            if res is not None:
                value["image_url"]["url"], n = res
                count += 1
                nbytes += n
        for child in value.values():
            c, b = _offload_image_parts_inplace(child, offload_dir)
            count += c
            nbytes += b
        return count, nbytes
    if isinstance(value, (list, tuple)):
        count = nbytes = 0
        for child in value:
            c, b = _offload_image_parts_inplace(child, offload_dir)
            count += c
            nbytes += b
        return count, nbytes
    # Pydantic image content part: ``part.type == "image_url"``, ``part.image_url.url``.
    if getattr(value, "type", None) == "image_url":
        src = getattr(value, "image_url", None)
        res = (
            _offload_image_url(getattr(src, "url", None), offload_dir)
            if src is not None
            else None
        )
        if res is not None:
            try:
                src.url = res[0]
                return 1, res[1]
            except Exception:  # frozen / validated model — leave it untouched
                return 0, 0
        return 0, 0
    # Pydantic message: descend into its content list.
    content = getattr(value, "content", None)
    if isinstance(content, (list, tuple)):
        return _offload_image_parts_inplace(content, offload_dir)
    return 0, 0


def _offload_prompt_images(prompt: Any) -> "dict[str, int]":
    """Offload base64 images in the current ``prompt`` to disk, in place."""
    offload_dir = _image_offload_dir()
    prompt_count, prompt_bytes = _offload_image_parts_inplace(prompt, offload_dir)
    return {
        "prompt_rewritten": prompt_count,
        "prompt_bytes": prompt_bytes,
    }


def _format_context(ctx: Mapping[str, Any]) -> str:
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


_RETRYABLE_MM_ERROR_TYPES = {
    "missing_mm_cache_item",
    "missing_mm_feature_artifact",
    "corrupt_mm_feature_artifact",
}


def _json_error_type(value: Any) -> str | None:
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except (TypeError, ValueError):
            return None
    if not isinstance(value, Mapping):
        return None
    error_type = value.get("error_type")
    return error_type if isinstance(error_type, str) else None


def _retryable_mm_error_type(exc: Exception) -> str | None:
    candidates: list[Any] = []
    body = getattr(exc, "body", None)
    if body is not None:
        candidates.append(body)
    response = getattr(exc, "response", None)
    if response is not None:
        try:
            candidates.append(response.json())
        except Exception:
            text = getattr(response, "text", None)
            if text is not None:
                candidates.append(text)

    for payload in candidates:
        if not isinstance(payload, Mapping):
            error_type = _json_error_type(payload)
            if error_type in _RETRYABLE_MM_ERROR_TYPES:
                return error_type
            continue
        error = payload.get("error")
        if isinstance(error, Mapping):
            error_type = error.get("type")
            if error_type in _RETRYABLE_MM_ERROR_TYPES:
                return cast(str, error_type)
            error_type = _json_error_type(error.get("message"))
            if error_type in _RETRYABLE_MM_ERROR_TYPES:
                return error_type
        error_type = _json_error_type(payload)
        if error_type in _RETRYABLE_MM_ERROR_TYPES:
            return error_type
    return None


async def _generate_with_mm_fallback(
    *, mm_log_context: Mapping[str, Any] | None = None, **kwargs: Any
) -> dict[str, Any]:
    """Send images hash-only first; fall back to full pixels on a cache miss.

    Prior-turn images reach ``generate`` descriptor-only and are serialized
    hash-only, assuming the engine still has them cached; the new turn's images
    carry ``pixel_values`` and are sent in full (see
    ``renderers.client.generate`` / ``_build_qwen_vl_features``). If the engine
    rejects the hash-only request because it evicted a hashed image, retry once
    with ``force_full_pixels=True`` so every image is re-materialized and sent.

    Both gates are inferred locally from ``multi_modal_data``: ``has_hash_only``
    (a cache-miss is worth retrying) is needed on the failure path where
    ``generate`` has no return value, and ``built_full`` (this request built
    full pixel payloads, so trim afterwards) follows from the same inspection —
    an image carries ``pixel_values`` iff ``generate`` sends it in full. Reading
    both here, not from state set inside ``generate`` (which runs the feature
    build on a pool thread — a copied context where a ``ContextVar.set`` would
    be invisible), is what keeps this correct.
    """
    ctx: Mapping[str, Any] = (
        mm_log_context if isinstance(mm_log_context, Mapping) else {}
    )
    mm_data = kwargs.get("multi_modal_data")
    descriptor_count, payload_count, _ = _mm_sidecar_stats(mm_data)
    has_hash_only = descriptor_count > 0
    built_full = payload_count > 0

    global _mm_request_counter
    with _mm_counter_lock:
        _mm_request_counter += 1
        req_id = _mm_request_counter

    start_rss = _rss_mb()
    try:
        with looptime("engine_generate"):
            return await generate(force_full_pixels=False, **kwargs)
    except Exception as exc:
        mm_error_type = _retryable_mm_error_type(exc)
        retryable = mm_error_type in _RETRYABLE_MM_ERROR_TYPES and (
            mm_error_type != "missing_mm_cache_item" or has_hash_only
        )
        if not retryable:
            raise

        _mm_logger.warning(
            "renderer_mm_repair_retry req=%d error_type=%s; retrying "
            "with all images materialized: %r ctx=%s",
            req_id,
            mm_error_type,
            exc,
            _format_context(ctx),
        )
        built_full = True
        with looptime("engine_generate_repair"):
            return await generate(force_full_pixels=True, **kwargs)
    finally:
        with looptime("mm_cleanup"):
            _maybe_cleanup_mm_request(req_id, built_full, start_rss, ctx)


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
) -> "tuple[RenderedTokens, int] | None":
    """Return the bridged prompt and routed-experts replay start.

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
        with looptime("render_bridge"):
            bridged = await _maybe_offload(renderer, bridge)
        with _bridge_metrics_lock:
            _bridge_metrics["attempts"] += 1
            _bridge_metrics["successes" if bridged is not None else "failures"] += 1
        if bridged is not None:
            start = max(len(previous_prompt_ids) + len(previous_completion_ids) - 1, 0)
            return bridged, start
        return None

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
        extra_kwargs: dict[str, Any] = {}
        if _image_offload_enabled():
            # Offload the synchronous image work (base64 decode + NFS writes/touch)
            # to a thread: on a shared NFS dir these fs ops can block for seconds,
            # and inline on the event loop they stall the env-worker heartbeat
            # (-> 30s timeout -> restart) under high inflight. See render-loop lag.
            extra_kwargs["_image_offload_stats"] = await asyncio.to_thread(
                _offload_prompt_images, messages
            )
        return (
            _attach_tool_call_names([_to_renderer_message(m) for m in messages]),
            extra_kwargs,
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
        offload_stats = cast(
            "dict[str, int] | None", kwargs.pop("_image_offload_stats", None)
        )

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
        log_context = {
            "session_id": session_id,
            "trajectory_id": _get_value(state, "trajectory_id")
            if state is not None
            else None,
            "prior_turns": prior_turns,
            "prompt_msgs": len(prompt),
            "model": model,
            "cache_salt": cache_salt,
            "client_idx": self._config.client_idx if self._config is not None else None,
        }

        # ``to_native_prompt`` offloads the original Verifiers prompt before
        # conversion, so the step later stored in ``state["trajectory"]`` keeps
        # the same cheap refs used by this native renderer prompt.
        if offload_stats is not None and offload_stats["prompt_rewritten"]:
            _mm_logger.info(
                "renderer_image_offload prompt_rewritten=%d bytes_mb=%.1f "
                "dir=%s ctx=%s",
                offload_stats["prompt_rewritten"],
                offload_stats["prompt_bytes"] / (1024.0 * 1024.0),
                _image_offload_dir(),
                _format_context(log_context),
            )

        bridged_with_start = await _get_incremental_prompt_ids(
            renderer=renderer,
            prompt=prompt,
            state=state,
            tools=tools,
        )
        # ``bridged_with_start`` is (RenderedTokens, replay_start) | None.
        # Unpack token_ids + mm_data (multimodal feature pass-through) and
        # prompt_attribution (per-token mask sidecar). On the first turn
        # (``bridged_with_start is None``), ``generate`` renders and emits the
        # attribution itself.
        if bridged_with_start is not None:
            bridged, routed_experts_prompt_start = bridged_with_start
            prompt_ids = bridged.token_ids
            multi_modal_data = bridged.multi_modal_data
            prompt_attribution = bridged
            sampling_params["routed_experts_prompt_start"] = routed_experts_prompt_start
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
            return await _generate_with_mm_fallback(
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
                mm_log_context=log_context,
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
        if not (has_content or has_tool_calls):
            if has_reasoning:
                raise EmptyModelResponseError(
                    "Model returned reasoning but no content and did not call any tools"
                )
            raise EmptyModelResponseError(
                "Model returned no content and did not call any tools"
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
