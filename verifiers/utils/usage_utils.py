from collections.abc import Mapping
import math
from types import MappingProxyType

from verifiers.types import TokenUsage


def _get_usage_value(usage_obj: object, key: str) -> int | float:
    if isinstance(usage_obj, Mapping):
        return usage_obj.get(key, 0)  # type: ignore[return-value]
    return getattr(usage_obj, key, 0)


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


def extract_usage_tokens(response: object) -> tuple[int, int]:
    usage = getattr(response, "usage", None)
    if usage is None:
        return 0, 0

    prompt_tokens = _get_usage_value(usage, "prompt_tokens")
    completion_tokens = _get_usage_value(usage, "completion_tokens")
    if not prompt_tokens and not completion_tokens:
        prompt_tokens = _get_usage_value(usage, "input_tokens")
        completion_tokens = _get_usage_value(usage, "output_tokens")
    return _coerce_usage_int(prompt_tokens), _coerce_usage_int(completion_tokens)


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
        mark_seen: bool = True,
    ) -> None:
        input_delta = float(input_tokens or 0.0)
        output_delta = float(output_tokens or 0.0)
        if input_delta < 0 or output_delta < 0:
            raise ValueError("Token usage increments must be non-negative.")
        if mark_seen:
            self._usage_seen = True
        self._usage_totals["input_tokens"] += input_delta
        self._usage_totals["output_tokens"] += output_delta

    def increment_from_response(self, response: object) -> None:
        if getattr(response, "usage", None) is None:
            return
        input_tokens, output_tokens = extract_usage_tokens(response)
        self.increment(input_tokens, output_tokens, mark_seen=True)

    def snapshot(self) -> TokenUsage | None:
        if not self._usage_seen:
            return None
        return {
            "input_tokens": self._usage_totals["input_tokens"],
            "output_tokens": self._usage_totals["output_tokens"],
        }


def compute_context_token_metrics(
    trajectory: list,
) -> dict[str, float]:
    """Compute context token metrics from the trajectory.

    Counts assistant messages in the last trajectory step's prompt to
    determine how many prior completions are still in context. This
    handles summarization/context-dropping automatically — dropped turns
    are simply absent from the prompt.

    Returns a dict with:
        longest_context_completion_tokens: Model-generated tokens still
            in context (sum of completion_tokens for in-context turns).
        longest_context_non_completion_tokens: Non-model tokens in context
            (total context minus completion tokens).
    """
    if not trajectory:
        return {
            "longest_context_completion_tokens": 0,
            "longest_context_non_completion_tokens": 0,
        }

    last_step = trajectory[-1]
    last_response = last_step.get("response")
    if last_response is None:
        return {
            "longest_context_completion_tokens": 0,
            "longest_context_non_completion_tokens": 0,
        }

    # The last step's prompt contains all messages currently in context.
    # Count assistant messages in the prompt — each one is a prior turn's
    # completion that's still in context.
    prompt_messages = last_step.get("prompt", [])
    assistant_count_in_prompt = sum(
        1
        for msg in prompt_messages
        if (getattr(msg, "role", None) == "assistant")
        or (isinstance(msg, dict) and msg.get("role") == "assistant")
    )

    # Collect completion_tokens from the last N+1 steps (N prior turns
    # in the prompt + the last step's own completion).
    # The most recent steps are the ones in context.
    steps_in_context = assistant_count_in_prompt + 1
    context_steps = trajectory[-steps_in_context:]

    context_completion = 0
    for step in context_steps:
        response = step.get("response")
        if response is None:
            continue
        _, completion_tokens = extract_usage_tokens(response)
        context_completion += completion_tokens

    # Total context = last step's prompt_tokens + completion_tokens
    last_prompt_tokens, last_completion_tokens = extract_usage_tokens(last_response)
    context_total = last_prompt_tokens + last_completion_tokens

    return {
        "longest_context_completion_tokens": context_completion,
        "longest_context_non_completion_tokens": max(
            0, context_total - context_completion
        ),
    }
