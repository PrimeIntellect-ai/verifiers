from __future__ import annotations

import asyncio
import json
import logging
from collections.abc import Mapping
from typing import cast

from verifiers.types import EvalCost, ModelPricing, TokenUsage

logger = logging.getLogger(__name__)

_PRIME_PRICING_CACHE: dict[str, ModelPricing] | None = None
_PRIME_PRICING_TASK: asyncio.Task[dict[str, ModelPricing]] | None = None


def _coerce_float(value: object) -> float | None:
    if not isinstance(value, str | int | float):
        return None

    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _coerce_model_pricing(value: object) -> ModelPricing | None:
    if not isinstance(value, Mapping):
        return None

    pricing = cast(Mapping[str, object], value)
    input_raw = pricing.get("input_usd_per_mtok")
    output_raw = pricing.get("output_usd_per_mtok")
    input_usd_per_mtok = _coerce_float(input_raw)
    output_usd_per_mtok = _coerce_float(output_raw)
    if input_usd_per_mtok is None or output_usd_per_mtok is None:
        return None

    return {
        "input_usd_per_mtok": input_usd_per_mtok,
        "output_usd_per_mtok": output_usd_per_mtok,
    }


def _parse_prime_pricing(payload: object) -> dict[str, ModelPricing]:
    if not isinstance(payload, Mapping):
        return {}

    payload_mapping = cast(Mapping[str, object], payload)
    data = payload_mapping.get("data")
    if not isinstance(data, list):
        return {}

    pricing_by_model: dict[str, ModelPricing] = {}
    for item in data:
        if not isinstance(item, Mapping):
            continue
        item_mapping = cast(Mapping[str, object], item)
        model_id = item_mapping.get("id")
        if not isinstance(model_id, str) or not model_id:
            continue
        pricing = _coerce_model_pricing(item_mapping.get("pricing"))
        if pricing is not None:
            pricing_by_model[model_id] = pricing

    return pricing_by_model


async def _fetch_prime_pricing_uncached() -> dict[str, ModelPricing]:
    try:
        process = await asyncio.create_subprocess_exec(
            "prime",
            "inference",
            "models",
            "--output",
            "json",
            "--plain",
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, stderr = await process.communicate()
    except (FileNotFoundError, OSError) as exc:
        logger.debug("Prime Inference pricing unavailable: %s", exc)
        return {}

    if process.returncode != 0:
        stderr_text = stderr.decode("utf-8", errors="replace").strip()
        logger.debug(
            "Prime Inference pricing command failed with exit code %s: %s",
            process.returncode,
            stderr_text,
        )
        return {}

    try:
        payload: object = json.loads(stdout.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        logger.debug("Prime Inference pricing response was not valid JSON: %s", exc)
        return {}

    return _parse_prime_pricing(payload)


async def fetch_prime_pricing() -> dict[str, ModelPricing]:
    """Fetch Prime Inference pricing once per process."""

    global _PRIME_PRICING_CACHE, _PRIME_PRICING_TASK

    if _PRIME_PRICING_CACHE is not None:
        return _PRIME_PRICING_CACHE

    if _PRIME_PRICING_TASK is None:
        _PRIME_PRICING_TASK = asyncio.create_task(_fetch_prime_pricing_uncached())

    _PRIME_PRICING_CACHE = await _PRIME_PRICING_TASK
    _PRIME_PRICING_TASK = None
    return _PRIME_PRICING_CACHE


def _clear_prime_pricing_cache() -> None:
    global _PRIME_PRICING_CACHE, _PRIME_PRICING_TASK

    _PRIME_PRICING_CACHE = None
    _PRIME_PRICING_TASK = None


def is_prime_inference_url(base_url: str) -> bool:
    return "pinference.ai" in base_url.lower()


def compute_cost_usd(
    usage: TokenUsage | None, pricing: ModelPricing | None
) -> float | None:
    if usage is None or pricing is None:
        return None

    input_tokens = _coerce_float(usage.get("input_tokens", 0.0))
    output_tokens = _coerce_float(usage.get("output_tokens", 0.0))
    if input_tokens is None or output_tokens is None:
        return None

    return (
        input_tokens * pricing["input_usd_per_mtok"] / 1_000_000
        + output_tokens * pricing["output_usd_per_mtok"] / 1_000_000
    )


def compute_eval_cost(
    usage: TokenUsage | None, pricing: ModelPricing | None
) -> EvalCost | None:
    total_usd = compute_cost_usd(usage, pricing)
    if usage is None or pricing is None or total_usd is None:
        return None

    input_tokens = _coerce_float(usage.get("input_tokens", 0.0))
    output_tokens = _coerce_float(usage.get("output_tokens", 0.0))
    if input_tokens is None or output_tokens is None:
        return None

    input_usd = input_tokens * pricing["input_usd_per_mtok"] / 1_000_000
    output_usd = output_tokens * pricing["output_usd_per_mtok"] / 1_000_000
    return {
        "input_usd": input_usd,
        "output_usd": output_usd,
        "total_usd": total_usd,
    }


def format_cost_usd(cost_usd: float) -> str:
    if cost_usd >= 1:
        return f"${cost_usd:.2f}"
    return f"${cost_usd:.4f}"
