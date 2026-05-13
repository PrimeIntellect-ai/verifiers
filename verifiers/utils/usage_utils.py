from collections.abc import Mapping, Sequence
from types import MappingProxyType

from verifiers.types import Response, TokenUsage, Usage


def extract_usage_tokens(response: Response) -> tuple[int, int]:
    usage = response.usage
    if usage is None:
        return 0, 0
    return usage_tokens(usage)


def usage_tokens(usage: Usage) -> tuple[int, int]:
    if usage.prompt_tokens < 0 or usage.completion_tokens < 0:
        raise ValueError("Response usage tokens must be non-negative.")
    return usage.prompt_tokens, usage.completion_tokens


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

    def increment_from_response(self, response: Response) -> None:
        if response.usage is None:
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
    trajectory: Sequence[Mapping[str, object]],
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
        if not isinstance(response, Response) or response.usage is None:
            continue
        prompt_tokens, completion_tokens = extract_usage_tokens(response)
        last_step_total = prompt_tokens + completion_tokens
        found = True
        break

    if not found:
        return _zero

    # Sum completion tokens across all steps with usage data.
    total_completion = 0
    for step in trajectory:
        response = step.get("response")
        if not isinstance(response, Response) or response.usage is None:
            continue
        _, completion_tokens = extract_usage_tokens(response)
        total_completion += completion_tokens

    return {
        "final_output_tokens": total_completion,
        "final_input_tokens": max(0, last_step_total - total_completion),
    }
