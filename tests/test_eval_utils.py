"""Tests for verifiers.utils.eval_utils.

Covers:
- print_results indexing with multiple rollouts per example
"""

from pathlib import Path

from verifiers.types import GenerateMetadata, GenerateOutputs


def _make_metadata(
    num_examples: int, rollouts_per_example: int = 1
) -> GenerateMetadata:
    return GenerateMetadata(
        env_id="test-env",
        env_args={},
        model="test-model",
        base_url="http://localhost",
        num_examples=num_examples,
        rollouts_per_example=rollouts_per_example,
        sampling_args={},
        date="1970-01-01",
        time_ms=0.0,
        avg_reward=0.0,
        avg_metrics={},
        state_columns=[],
        path_to_save=Path("test.jsonl"),
    )


def test_print_results_rollout_indexing(capsys):
    """Test that print_results correctly groups results by rollout when sorted by example_id.

    Results are sorted by example_id, giving order: [ex0_r0, ex0_r1, ex1_r0, ex1_r1, ...]
    The indexing should correctly extract:
    - r1: all first rollouts (indices 0, 2, 4, ...)
    - r2: all second rollouts (indices 1, 3, 5, ...)
    """
    from verifiers.utils.eval_utils import print_results

    num_examples = 3
    rollouts_per_example = 2

    # Simulate results sorted by example_id (as generate() now does)
    # Order: [ex0_r0, ex0_r1, ex1_r0, ex1_r1, ex2_r0, ex2_r1]
    # Rewards are designed so we can verify correct grouping:
    # - All r0 rewards: 0.1, 0.3, 0.5 (for examples 0, 1, 2)
    # - All r1 rewards: 0.2, 0.4, 0.6 (for examples 0, 1, 2)
    rewards = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]
    example_ids = [0, 0, 1, 1, 2, 2]

    # Metric follows same pattern
    metric_values = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]

    results = GenerateOutputs(
        prompt=[[{"role": "user", "content": f"q{i}"}] for i in range(6)],
        completion=[[{"role": "assistant", "content": f"a{i}"}] for i in range(6)],
        answer=[""] * 6,
        state=[{"timing": {"generation_ms": 0.0, "scoring_ms": 0.0, "total_ms": 0.0}}]
        * 6,
        task=["default"] * 6,
        info=[{}] * 6,
        example_id=example_ids,
        reward=rewards,
        metrics={"test_metric": metric_values},
        metadata=_make_metadata(num_examples, rollouts_per_example),
    )

    print_results(results)
    captured = capsys.readouterr()

    # Verify rollout groupings are correct
    # r1 should have rewards [0.1, 0.3, 0.5] (first rollout of each example)
    assert "r1: [0.1, 0.3, 0.5]" in captured.out
    # r2 should have rewards [0.2, 0.4, 0.6] (second rollout of each example)
    assert "r2: [0.2, 0.4, 0.6]" in captured.out

    # Same for metrics
    assert "r1: [1.0, 3.0, 5.0]" in captured.out
    assert "r2: [2.0, 4.0, 6.0]" in captured.out


def test_print_results_single_rollout(capsys):
    """Test print_results with single rollout per example (edge case)."""
    from verifiers.utils.eval_utils import print_results

    num_examples = 3
    rollouts_per_example = 1

    rewards = [0.1, 0.2, 0.3]
    example_ids = [0, 1, 2]

    results = GenerateOutputs(
        prompt=[[{"role": "user", "content": f"q{i}"}] for i in range(3)],
        completion=[[{"role": "assistant", "content": f"a{i}"}] for i in range(3)],
        answer=[""] * 3,
        state=[{"timing": {"generation_ms": 0.0, "scoring_ms": 0.0, "total_ms": 0.0}}]
        * 3,
        task=["default"] * 3,
        info=[{}] * 3,
        example_id=example_ids,
        reward=rewards,
        metrics={},
        metadata=_make_metadata(num_examples, rollouts_per_example),
    )

    print_results(results)
    captured = capsys.readouterr()

    # With single rollout, r1 should have all rewards
    assert "r1: [0.1, 0.2, 0.3]" in captured.out


def test_print_results_three_rollouts(capsys):
    """Test print_results with three rollouts per example."""
    from verifiers.utils.eval_utils import print_results

    num_examples = 2
    rollouts_per_example = 3

    # Order: [ex0_r0, ex0_r1, ex0_r2, ex1_r0, ex1_r1, ex1_r2]
    rewards = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]
    example_ids = [0, 0, 0, 1, 1, 1]

    results = GenerateOutputs(
        prompt=[[{"role": "user", "content": f"q{i}"}] for i in range(6)],
        completion=[[{"role": "assistant", "content": f"a{i}"}] for i in range(6)],
        answer=[""] * 6,
        state=[{"timing": {"generation_ms": 0.0, "scoring_ms": 0.0, "total_ms": 0.0}}]
        * 6,
        task=["default"] * 6,
        info=[{}] * 6,
        example_id=example_ids,
        reward=rewards,
        metrics={},
        metadata=_make_metadata(num_examples, rollouts_per_example),
    )

    print_results(results)
    captured = capsys.readouterr()

    # r1 should have [0.1, 0.4] (first rollout of each example)
    assert "r1: [0.1, 0.4]" in captured.out
    # r2 should have [0.2, 0.5] (second rollout of each example)
    assert "r2: [0.2, 0.5]" in captured.out
    # r3 should have [0.3, 0.6] (third rollout of each example)
    assert "r3: [0.3, 0.6]" in captured.out
