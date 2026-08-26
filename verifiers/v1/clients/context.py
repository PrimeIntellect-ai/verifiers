"""Model context-window discovery for OpenAI-compatible endpoints."""

from collections.abc import Mapping
from typing import Any, cast

from openai import APIError

from verifiers.v1.clients.base import build_async_openai
from verifiers.v1.clients.client import ModelContext

CONTEXT_WINDOW_FIELDS = (
    "max_model_len",
    "context_length",
    "context_window",
    "max_context_length",
)
_context_window_cache: dict[tuple[str, str], int | None] = {}


def model_context_window(payload: Mapping[str, Any], model: str) -> int | None:
    """Read a provider context-window extension from one model card."""
    for card in payload.get("data") or []:
        if not isinstance(card, Mapping) or card.get("id") != model:
            continue
        for field in CONTEXT_WINDOW_FIELDS:
            value = card.get(field)
            if isinstance(value, int) and not isinstance(value, bool) and value > 0:
                return value
        break
    return None


def compaction_threshold(context_window: int) -> int:
    """Reserve ten percent of the model context for checkpointing."""
    return max(1, context_window * 9 // 10)


async def resolve_compaction_threshold(ctx: ModelContext) -> int | None:
    """Discover a model's proactive compaction threshold when advertised."""
    key = (ctx.client.model_dump_json(), ctx.model)
    if key not in _context_window_cache:
        try:
            async with build_async_openai(ctx.client) as client:
                payload = await client.get("/models", cast_to=cast(Any, dict[str, Any]))
            _context_window_cache[key] = model_context_window(payload, ctx.model)
        except APIError:
            _context_window_cache[key] = None

    window = _context_window_cache[key]
    return compaction_threshold(window) if window is not None else None
