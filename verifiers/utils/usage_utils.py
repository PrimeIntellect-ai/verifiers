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


def _normalize_for_comparison(value: object) -> object:
    """Normalize messages for prefix comparison.

    Same idea as the TITO best-effort prefix matching: convert pydantic
    objects to dicts and recursively normalize so that structurally
    identical messages compare equal regardless of their Python type.
    """
    if hasattr(value, "model_dump"):
        return _normalize_for_comparison(value.model_dump())  # type: ignore[union-attr]
    if isinstance(value, Mapping):
        return {str(k): _normalize_for_comparison(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_normalize_for_comparison(item) for item in value]
    return value


def _find_parent_step(
    trajectory: list,
    step_idx: int,
    normalized_prompts: list,
    normalized_completions: list,
) -> int:
    """Find the ancestor of trajectory[step_idx] via message prefix matching.

    Walks backwards through prior steps.  A step *j* is a candidate parent
    of step *i* when ``normalize(step_j.prompt + step_j.completion)`` is a
    prefix of ``normalize(step_i.prompt)``.  Returns the index of the best
    (longest prefix) match, or -1 if none.
    """
    target_prompt = normalized_prompts[step_idx]
    best_idx = -1
    best_prefix_len = 0
    for j in range(step_idx - 1, -1, -1):
        candidate = normalized_prompts[j] + normalized_completions[j]
        prefix_len = len(candidate)
        if prefix_len <= 0 or prefix_len <= best_prefix_len:
            continue
        if prefix_len > len(target_prompt):
            continue
        if target_prompt[:prefix_len] == candidate:
            best_prefix_len = prefix_len
            best_idx = j
    return best_idx


def compute_context_token_metrics(
    trajectory: list,
) -> dict[str, float]:
    """Compute context token metrics from the trajectory.

    Uses the same message-prefix matching approach as the best-effort TITO
    mechanism to detect which prior completions are still in context.  For
    each step, ``step.prompt + step.completion`` is compared as a prefix
    against later steps' prompts.  This automatically handles context
    dropping, summarization, branching, and history rewriting — no
    trajectory_id filtering required.

    Finds the step with the largest total context (prompt_tokens +
    completion_tokens), then walks its ancestor chain to partition context
    into model-generated vs. non-model tokens.

    Returns a dict with:
        longest_context_completion_tokens: Model-generated tokens still
            in context (sum of completion_tokens for in-context turns).
        longest_context_non_completion_tokens: Non-model tokens in context
            (total context minus completion tokens).
    """
    _zero: dict[str, float] = {
        "longest_context_completion_tokens": 0,
        "longest_context_non_completion_tokens": 0,
    }
    if not trajectory:
        return _zero

    # Find the step with the largest total context.
    best_step_idx = -1
    best_total = -1
    for i, step in enumerate(trajectory):
        response = step.get("response")
        if response is None:
            continue
        prompt_tokens, completion_tokens = extract_usage_tokens(response)
        total = prompt_tokens + completion_tokens
        if total > best_total:
            best_total = total
            best_step_idx = i

    if best_step_idx < 0:
        return _zero

    # Pre-normalize all prompts and completions once.
    normalized_prompts = [
        _normalize_for_comparison(step.get("prompt") or []) for step in trajectory
    ]
    normalized_completions = [
        _normalize_for_comparison(step.get("completion") or []) for step in trajectory
    ]

    # Build the ancestor chain for the target step via prefix matching.
    chain_indices = [best_step_idx]
    current_idx = best_step_idx
    while current_idx > 0:
        parent_idx = _find_parent_step(
            trajectory, current_idx, normalized_prompts, normalized_completions
        )
        if parent_idx < 0:
            break
        chain_indices.append(parent_idx)
        current_idx = parent_idx

    # Sum completion tokens along the ancestor chain.
    context_completion = 0
    for idx in chain_indices:
        response = trajectory[idx].get("response")
        if response is None:
            continue
        _, completion_tokens = extract_usage_tokens(response)
        context_completion += completion_tokens

    return {
        "longest_context_completion_tokens": context_completion,
        "longest_context_non_completion_tokens": max(
            0, best_total - context_completion
        ),
    }
