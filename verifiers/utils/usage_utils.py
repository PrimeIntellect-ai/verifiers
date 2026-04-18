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
    """Accumulates token usage and exposes a read-only live usage mapping.

    ``fork()`` returns a zero-initialised child tracker; ``merge(other)``
    absorbs the child's accumulated deltas. Used by ``MultiAgentEnv``'s
    simultaneous slot to isolate per-agent accounting: if the slot fails,
    the forked trackers are dropped and the parent stays at its pre-slot
    snapshot (no orphan tokens billed to a rolled-back slot).
    """

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

    def fork(self) -> "StateUsageTracker":
        """Return a zero-initialised child tracker for branch-local accounting."""
        return StateUsageTracker()

    def merge(self, other: "StateUsageTracker") -> None:
        """Absorb another tracker's accumulated deltas into this one.

        Called in the success phase of a simultaneous-slot commit to fold
        per-agent usage into the shared tracker. If the slot fails, the
        child trackers are dropped and no merge happens — the shared
        tracker stays at its pre-slot snapshot.
        """
        if other._usage_seen:
            self._usage_seen = True
        self._usage_totals["input_tokens"] += other._usage_totals["input_tokens"]
        self._usage_totals["output_tokens"] += other._usage_totals["output_tokens"]
