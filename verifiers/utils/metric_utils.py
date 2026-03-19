import math
from abc import ABC, abstractmethod
from collections import defaultdict
from typing import Any, Protocol, runtime_checkable

from verifiers.types import RolloutOutput


@runtime_checkable
class Metric(Protocol):
    """Protocol for incremental eval metrics over rollout outputs."""

    def add_output(self, output: RolloutOutput) -> None: ...
    def add_outputs(self, outputs: list[RolloutOutput]) -> None: ...
    def compute(self) -> Any: ...
    def reset(self) -> None: ...


class MeanMetric(ABC):
    """Abstract running mean over a value extracted from each RolloutOutput.

    Subclasses override ``extract`` to select which value to average.
    Return ``None`` from ``extract`` to skip an output.
    """

    __slots__ = ("_sum", "_count")

    def __init__(self) -> None:
        self.reset()

    @abstractmethod
    def extract(self, output: RolloutOutput) -> float | None:
        """Extract the value to accumulate, or None to skip this output."""

    def add_output(self, output: RolloutOutput) -> None:
        value = self.extract(output)
        if value is not None:
            self._sum += value
            self._count += 1

    def add_outputs(self, outputs: list[RolloutOutput]) -> None:
        for output in outputs:
            self.add_output(output)

    def reset(self) -> None:
        self._sum = 0.0
        self._count = 0

    def compute(self) -> float:
        return self._sum / self._count if self._count else 0.0

    @property
    def count(self) -> int:
        return self._count


class RewardMetric(MeanMetric):
    """Mean reward across outputs."""

    def extract(self, output: RolloutOutput) -> float:
        return output.get("reward", 0.0)


class ErrorRateMetric(MeanMetric):
    """Fraction of outputs with a non-None error."""

    def extract(self, output: RolloutOutput) -> float:
        return 1.0 if output.get("error") is not None else 0.0


class InputTokensMetric(MeanMetric):
    """Mean input tokens per output (skips outputs without token_usage)."""

    def extract(self, output: RolloutOutput) -> float | None:
        usage = output.get("token_usage")
        if isinstance(usage, dict):
            return float(usage.get("input_tokens", 0.0))
        return None


class OutputTokensMetric(MeanMetric):
    """Mean output tokens per output (skips outputs without token_usage)."""

    def extract(self, output: RolloutOutput) -> float | None:
        usage = output.get("token_usage")
        if isinstance(usage, dict):
            return float(usage.get("output_tokens", 0.0))
        return None


class EnvMetrics:
    """Per-key running means for environment-defined metrics.

    Dynamically creates accumulators for each metric key seen in
    ``output["metrics"]``. Non-numeric values are skipped.
    """

    def __init__(self) -> None:
        self.reset()

    def add_output(self, output: RolloutOutput) -> None:
        output_metrics = output.get("metrics", {})
        if output_metrics:
            for name, value in output_metrics.items():
                if isinstance(value, (int, float)):
                    if name not in self._sums:
                        self._sums[name] = 0.0
                        self._counts[name] = 0
                    self._sums[name] += value
                    self._counts[name] += 1

    def add_outputs(self, outputs: list[RolloutOutput]) -> None:
        for output in outputs:
            self.add_output(output)

    def reset(self) -> None:
        self._sums: dict[str, float] = {}
        self._counts: dict[str, int] = {}

    def compute(self) -> dict[str, float]:
        return {k: self._sums[k] / self._counts[k] for k in self._sums}


class PassAtKMetric:
    """Incremental pass@k and pass^k using unbiased estimators.

    pass@k = 1 - C(n-c, k) / C(n, k)  (at least one correct in k samples)
    pass^k = C(c, k) / C(n, k)         (all k samples correct)

    Both averaged across complete examples.
    n = rollouts_per_example, c = correct rollouts (reward >= threshold).
    k values: powers of 2 in [1, n].
    """

    def __init__(self, rollouts_per_example: int, threshold: float = 0.5) -> None:
        self.rollouts_per_example = rollouts_per_example
        self.threshold = threshold

        self._k_values: list[int] = []
        if rollouts_per_example > 1:
            k = 1
            while k <= rollouts_per_example:
                self._k_values.append(k)
                k *= 2

        self.reset()

    def add_output(self, output: RolloutOutput) -> None:
        if not self._k_values:
            return

        example_id = output.get("example_id", 0)
        self._example_counts[example_id] += 1
        if output.get("reward", 0.0) >= self.threshold:
            self._example_correct[example_id] += 1

        if self._example_counts[example_id] == self.rollouts_per_example:
            self._num_complete += 1
            n = self.rollouts_per_example
            c = self._example_correct[example_id]
            for kv in self._k_values:
                kv_str = str(kv)
                n_choose_k = math.comb(n, kv)
                if n - c < kv:
                    self._pass_at_k_sums[kv_str] += 1.0
                else:
                    self._pass_at_k_sums[kv_str] += (
                        1.0 - math.comb(n - c, kv) / n_choose_k
                    )
                self._pass_all_k_sums[kv_str] += math.comb(c, kv) / n_choose_k

    def add_outputs(self, outputs: list[RolloutOutput]) -> None:
        for output in outputs:
            self.add_output(output)

    def reset(self) -> None:
        self._example_counts: dict[int, int] = defaultdict(int)
        self._example_correct: dict[int, int] = defaultdict(int)
        self._num_complete = 0
        self._pass_at_k_sums: dict[str, float] = defaultdict(float)
        self._pass_all_k_sums: dict[str, float] = defaultdict(float)

    def compute(self) -> tuple[dict[str, float], dict[str, float]]:
        """Return (pass_at_k, pass_all_k) dicts. Empty if no complete examples."""
        if self._num_complete == 0:
            return {}, {}
        pass_at_k: dict[str, float] = {}
        pass_all_k: dict[str, float] = {}
        for kv in self._k_values:
            kv_str = str(kv)
            pass_at_k[kv_str] = self._pass_at_k_sums[kv_str] / self._num_complete
            pass_all_k[kv_str] = self._pass_all_k_sums[kv_str] / self._num_complete
        return pass_at_k, pass_all_k
