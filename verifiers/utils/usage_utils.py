from collections.abc import Mapping
import math
from types import MappingProxyType

from typing import Any

from verifiers.types import TokenUsage


def _get_usage_value(usage_obj: object, key: str) -> int | float:
    if isinstance(usage_obj, Mapping):
        return usage_obj.get(key, 0)  # type: ignore[return-value]
    return getattr(usage_obj, key, 0)


def _get_optional_usage_value(usage_obj: object, key: str) -> object:
    if isinstance(usage_obj, Mapping):
        return usage_obj.get(key)
    return getattr(usage_obj, key, None)


def _get_nested_usage_value(usage_obj: object, key: str) -> object:
    value = _get_optional_usage_value(usage_obj, key)
    if value is not None:
        return value
    details = _get_optional_usage_value(usage_obj, "prompt_tokens_details")
    if isinstance(details, Mapping):
        return details.get(key)
    if details is not None:
        return getattr(details, key, None)
    return None


def _get_response_usage(response: object) -> object:
    if isinstance(response, Mapping):
        return response.get("usage")
    return getattr(response, "usage", None)


def _coerce_usage_int(value: object) -> int:
    """Best-effort usage coercion. Invalid values degrade to zero."""
    if value is None:
        return 0
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, int):
        return max(0, value)
    if isinstance(value, float):
        if math.isnan(value) or math.isinf(value):
            return 0
        return max(0, int(value))
    if isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            return 0
        try:
            return max(0, int(stripped))
        except (TypeError, ValueError):
            try:
                parsed = float(stripped)
                if math.isnan(parsed) or math.isinf(parsed):
                    return 0
                return max(0, int(parsed))
            except (TypeError, ValueError):
                return 0
    return 0


def extract_usage_token_details(response: object) -> dict[str, int] | None:
    usage = _get_response_usage(response)
    if usage is None:
        return None

    prompt_tokens = _get_usage_value(usage, "prompt_tokens")
    completion_tokens = _get_usage_value(usage, "completion_tokens")
    if not prompt_tokens and not completion_tokens:
        prompt_tokens = _get_usage_value(usage, "input_tokens")
        completion_tokens = _get_usage_value(usage, "output_tokens")
    details = {
        "input_tokens": _coerce_usage_int(prompt_tokens),
        "output_tokens": _coerce_usage_int(completion_tokens),
    }

    subtract_cached_from_input = False
    cached_input_tokens = _get_optional_usage_value(usage, "cached_input_tokens")
    if cached_input_tokens is None:
        cached_input_tokens = _get_optional_usage_value(usage, "cache_read_input_tokens")
    if cached_input_tokens is None:
        cached_input_tokens = _get_nested_usage_value(usage, "cached_tokens")
        subtract_cached_from_input = cached_input_tokens is not None
    if cached_input_tokens is not None:
        cached_int = _coerce_usage_int(cached_input_tokens)
        details["cached_input_tokens"] = cached_int
        if subtract_cached_from_input:
            details["input_tokens"] = max(0, details["input_tokens"] - cached_int)

    cache_creation_input_tokens = _get_optional_usage_value(
        usage, "cache_creation_input_tokens"
    )
    if cache_creation_input_tokens is not None:
        details["input_tokens"] += _coerce_usage_int(cache_creation_input_tokens)

    return details


def extract_usage_tokens(response: object) -> tuple[int, int]:
    details = extract_usage_token_details(response)
    if details is None:
        return 0, 0
    return details["input_tokens"], details["output_tokens"]


class StateUsageTracker:
    """Accumulates token usage and exposes a read-only live usage mapping."""

    __slots__ = ("_usage_seen", "_usage_totals", "_usage_view")

    def __init__(self) -> None:
        self._usage_seen = False
        self._usage_totals: dict[str, float] = {
            "input_tokens": 0.0,
            "output_tokens": 0.0,
        }
        self._usage_view = MappingProxyType(self._usage_totals)

    @property
    def usage(self) -> Mapping[str, float]:
        return self._usage_view

    def increment(
        self,
        input_tokens: int | float = 0,
        output_tokens: int | float = 0,
        *,
        cached_input_tokens: int | float | None = None,
        mark_seen: bool = True,
    ) -> None:
        deltas: dict[str, float] = {
            "input_tokens": float(input_tokens or 0.0),
            "output_tokens": float(output_tokens or 0.0),
        }
        if cached_input_tokens is not None:
            deltas["cached_input_tokens"] = float(cached_input_tokens or 0.0)
        if any(delta < 0 for delta in deltas.values()):
            raise ValueError("Token usage increments must be non-negative.")
        if mark_seen:
            self._usage_seen = True
        for key, delta in deltas.items():
            self._usage_totals[key] = self._usage_totals.get(key, 0.0) + delta

    def increment_from_response(self, response: object) -> None:
        if _get_response_usage(response) is None:
            return
        details = extract_usage_token_details(response)
        if details is None:
            return
        self.increment(
            details["input_tokens"],
            details["output_tokens"],
            cached_input_tokens=details.get("cached_input_tokens"),
            mark_seen=True,
        )

    def snapshot(self) -> TokenUsage | None:
        if not self._usage_seen:
            return None
        return cast_token_usage(self._usage_totals)


def cast_token_usage(usage: Mapping[str, Any]) -> TokenUsage:
    out: TokenUsage = {
        "input_tokens": float(usage.get("input_tokens", 0.0)),
        "output_tokens": float(usage.get("output_tokens", 0.0)),
    }
    for key in ("cached_input_tokens",):
        value = usage.get(key)
        if value is not None:
            out[key] = float(value)
    return out


def compute_context_token_metrics(
    trajectory: list,
) -> dict[str, float]:
    """Compute context token metrics from the trajectory.

    Assumes a linear rollout: uses the last trajectory step with a
    response as the reference point, and sums completion_tokens across
    all steps as the model-generated tokens in context.

    Returns a dict with:
        final_output_tokens: Model-generated tokens (sum of completion_tokens
            across all steps).
        final_input_tokens: Non-model tokens in context (last step's total
            context minus final_output_tokens).
    """
    _zero: dict[str, float] = {
        "final_output_tokens": 0,
        "final_input_tokens": 0,
    }
    if not trajectory:
        return _zero

    # Find the last step with usage data.
    last_step_total = 0
    found = False
    for step in reversed(trajectory):
        response = step.get("response")
        if response is None or _get_response_usage(response) is None:
            continue
        details = extract_usage_token_details(response)
        if details is None:
            continue
        prompt_tokens = details["input_tokens"] + details.get(
            "cached_input_tokens", 0
        )
        completion_tokens = details["output_tokens"]
        last_step_total = prompt_tokens + completion_tokens
        found = True
        break

    if not found:
        return _zero

    # Sum completion tokens across all steps with usage data.
    total_completion = 0
    for step in trajectory:
        response = step.get("response")
        if response is None or _get_response_usage(response) is None:
            continue
        details = extract_usage_token_details(response)
        if details is not None:
            total_completion += details["output_tokens"]

    return {
        "final_output_tokens": total_completion,
        "final_input_tokens": max(0, last_step_total - total_completion),
    }
