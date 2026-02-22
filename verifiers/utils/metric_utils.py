import math
from collections import defaultdict

from verifiers.types import RolloutOutput


def compute_pass_at_k(
    outputs: list[RolloutOutput],
    rollouts_per_example: int,
    threshold: float = 0.5,
) -> tuple[dict[str, float], dict[str, float]]:
    """Compute pass@k and pass^k metrics using unbiased estimators.

    pass@k = 1 - C(n-c, k) / C(n, k)  (at least one correct in k samples)
    pass^k = C(c, k) / C(n, k)         (all k samples correct)

    Both averaged across examples.
    n = rollouts per example, c = correct rollouts (reward >= threshold).
    k values: all powers of 2 in [1, n].
    Returns (empty, empty) if rollouts_per_example <= 1.
    """
    if rollouts_per_example <= 1:
        return {}, {}

    # Determine k values: powers of 2 in [1, n]
    k_values: list[int] = []
    k = 1
    while k <= rollouts_per_example:
        k_values.append(k)
        k *= 2

    if not k_values:
        return {}, {}

    # Group outputs by example_id
    examples: dict[int, list[RolloutOutput]] = defaultdict(list)
    for output in outputs:
        examples[output.get("example_id", 0)].append(output)

    # Compute pass@k and pass^k for each example, then average
    pass_at_k_sums: dict[str, float] = {str(kv): 0.0 for kv in k_values}
    pass_all_k_sums: dict[str, float] = {str(kv): 0.0 for kv in k_values}
    num_examples = len(examples)

    for example_outputs in examples.values():
        n = len(example_outputs)
        c = sum(1 for o in example_outputs if o.get("reward", 0.0) >= threshold)
        for kv in k_values:
            if n < kv:
                continue
            kv_str = str(kv)
            n_choose_k = math.comb(n, kv)
            # pass@k: P(at least one correct)
            if n - c < kv:
                pass_at_k_sums[kv_str] += 1.0
            else:
                pass_at_k_sums[kv_str] += 1.0 - math.comb(n - c, kv) / n_choose_k
            # pass^k: P(all correct)
            pass_all_k_sums[kv_str] += math.comb(c, kv) / n_choose_k

    pass_at_k = {
        str(kv): pass_at_k_sums[str(kv)] / num_examples
        for kv in k_values
        if num_examples > 0
    }
    pass_all_k = {
        str(kv): pass_all_k_sums[str(kv)] / num_examples
        for kv in k_values
        if num_examples > 0
    }
    return pass_at_k, pass_all_k
