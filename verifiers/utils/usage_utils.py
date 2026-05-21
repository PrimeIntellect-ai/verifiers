from collections.abc import Mapping, Sequence
from types import MappingProxyType
from typing import Any, cast

from verifiers.types import Response, TokenUsage, Usage


def _get_field(obj: object, key: str) -> object:
    if isinstance(obj, Mapping):
        return cast(Mapping[str, object], obj).get(key)
    return getattr(obj, key, None)


def _as_token_count(value: object) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return max(0, value)
    if isinstance(value, float) and value.is_integer():
        return max(0, int(value))
    return None


def _response_usage(response: object) -> object | None:
    return _get_field(response, "usage")


def _nested_cached_tokens(usage: object) -> int | None:
    for details_key in ("prompt_tokens_details", "input_tokens_details"):
        details = _get_field(usage, details_key)
        if details is None:
            continue
        cached = _as_token_count(_get_field(details, "cached_tokens"))
        if cached is not None:
            return cached
    return None


def _cache_creation_tokens(usage: object) -> int:
    return _as_token_count(_get_field(usage, "cache_creation_input_tokens")) or 0


def _direct_cached_tokens(usage: object) -> int | None:
    cached = _as_token_count(_get_field(usage, "cached_input_tokens"))
    if cached is not None:
        return cached
    return _as_token_count(_get_field(usage, "cache_read_input_tokens"))


def extract_usage_token_details(response: object) -> dict[str, int] | None:
    usage = _response_usage(response)
    if usage is None:
        return None

    input_tokens = _as_token_count(_get_field(usage, "prompt_tokens"))
    output_tokens = _as_token_count(_get_field(usage, "completion_tokens"))
    if input_tokens is None and output_tokens is None:
        input_tokens = _as_token_count(_get_field(usage, "input_tokens"))
        output_tokens = _as_token_count(_get_field(usage, "output_tokens"))
    if input_tokens is None or output_tokens is None:
        return None

    input_tokens += _cache_creation_tokens(usage)
    details = {
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
    }

    cached_tokens = _direct_cached_tokens(usage)
    if cached_tokens is not None:
        details["cached_input_tokens"] = cached_tokens
        return details

    cached_tokens = _nested_cached_tokens(usage)
    if cached_tokens is not None:
        details["input_tokens"] = max(0, input_tokens - cached_tokens)
        details["cached_input_tokens"] = cached_tokens
    return details


def extract_usage_tokens(response: object) -> tuple[int, int]:
    details = extract_usage_token_details(response)
    if details is None:
        return 0, 0
    return details["input_tokens"], details["output_tokens"]


def usage_tokens(usage: Usage) -> tuple[int, int]:
    if usage.prompt_tokens < 0 or usage.completion_tokens < 0:
        raise ValueError("Response usage tokens must be non-negative.")
    return usage.prompt_tokens, usage.completion_tokens


def response_usage_tokens(response: Response) -> tuple[int, int]:
    usage = response.usage
    if usage is None:
        return 0, 0
    return usage_tokens(usage)


def cast_token_usage(usage: Mapping[str, Any]) -> TokenUsage:
    out: TokenUsage = {
        "input_tokens": float(usage.get("input_tokens", 0.0)),
        "output_tokens": float(usage.get("output_tokens", 0.0)),
    }
    cached = usage.get("cached_input_tokens")
    if cached is not None:
        out["cached_input_tokens"] = float(cached)
    return out


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


def compute_context_token_metrics(
    trajectory: Sequence[Mapping[str, object]],
) -> dict[str, float]:
    """Compute context token metrics from the trajectory.

    Assumes a linear rollout: uses the last trajectory step with a
    response as the reference point, and sums completion_tokens across
    all steps as the model-generated tokens in context.

    Returns a dict with:
        final_output_tokens: Model-generated tokens (sum of completion_tokens
            across all steps).
        final_input_tokens: Non-model tokens in context (system prompts, user
            messages, tool results, etc.).
    """
    zero = {"final_output_tokens": 0.0, "final_input_tokens": 0.0}
    if not trajectory:
        return zero

    last_step_total = 0
    found = False
    for step in reversed(trajectory):
        details = extract_usage_token_details(step.get("response"))
        if details is None:
            continue
        last_step_total = (
            details["input_tokens"]
            + details.get("cached_input_tokens", 0)
            + details["output_tokens"]
        )
        found = True
        break

    if not found:
        return zero

    total_completion = 0
    for step in trajectory:
        details = extract_usage_token_details(step.get("response"))
        if details is not None:
            total_completion += details["output_tokens"]

    return {
        "final_output_tokens": float(total_completion),
        "final_input_tokens": float(max(0, last_step_total - total_completion)),
    }
