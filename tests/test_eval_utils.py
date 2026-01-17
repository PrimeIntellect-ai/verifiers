"""Tests for verifiers.utils.eval_utils.

Covers:
- print_results indexing with multiple rollouts per example
"""

from verifiers.types import GenerateMetadata, GenerateOutputs, RolloutResult


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
    )


def _make_rollout(
    example_id: int,
    reward: float,
    metrics: dict[str, float] | None = None,
) -> RolloutResult:
    return RolloutResult(
        prompt=[{"role": "user", "content": f"q{example_id}"}],
        completion=[{"role": "assistant", "content": f"a{example_id}"}],
        example_id=example_id,
        task="default",
        reward=reward,
        metrics=metrics or {},
        timing={"generation_ms": 0.0, "scoring_ms": 0.0, "total_ms": 0.0},
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
    rollouts = [
        _make_rollout(0, 0.1, {"test_metric": 1.0}),
        _make_rollout(0, 0.2, {"test_metric": 2.0}),
        _make_rollout(1, 0.3, {"test_metric": 3.0}),
        _make_rollout(1, 0.4, {"test_metric": 4.0}),
        _make_rollout(2, 0.5, {"test_metric": 5.0}),
        _make_rollout(2, 0.6, {"test_metric": 6.0}),
    ]

    output = GenerateOutputs(
        rollouts=rollouts,
        metadata=_make_metadata(num_examples, rollouts_per_example),
    )

    print_results(output)
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

    rollouts = [
        _make_rollout(0, 0.1),
        _make_rollout(1, 0.2),
        _make_rollout(2, 0.3),
    ]

    output = GenerateOutputs(
        rollouts=rollouts,
        metadata=_make_metadata(num_examples, rollouts_per_example),
    )

    print_results(output)
    captured = capsys.readouterr()

    # With single rollout, r1 should have all rewards
    assert "r1: [0.1, 0.2, 0.3]" in captured.out


def test_print_results_three_rollouts(capsys):
    """Test print_results with three rollouts per example."""
    from verifiers.utils.eval_utils import print_results

    num_examples = 2
    rollouts_per_example = 3

    # Order: [ex0_r0, ex0_r1, ex0_r2, ex1_r0, ex1_r1, ex1_r2]
    rollouts = [
        _make_rollout(0, 0.1),
        _make_rollout(0, 0.2),
        _make_rollout(0, 0.3),
        _make_rollout(1, 0.4),
        _make_rollout(1, 0.5),
        _make_rollout(1, 0.6),
    ]

    output = GenerateOutputs(
        rollouts=rollouts,
        metadata=_make_metadata(num_examples, rollouts_per_example),
    )

    print_results(output)
    captured = capsys.readouterr()

    # r1 should have [0.1, 0.4] (first rollout of each example)
    assert "r1: [0.1, 0.4]" in captured.out
    # r2 should have [0.2, 0.5] (second rollout of each example)
    assert "r2: [0.2, 0.5]" in captured.out
    # r3 should have [0.3, 0.6] (third rollout of each example)
    assert "r3: [0.3, 0.6]" in captured.out
