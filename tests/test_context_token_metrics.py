"""Tests for per-turn context token metrics.

Tests the StateUsageTracker per-turn tracking, branch detection,
and the branch-aware context token computation
(longest_context_completion_tokens, longest_context_non_completion_tokens).
"""

from unittest.mock import MagicMock

import pytest

from verifiers.utils.usage_utils import (
    StateUsageTracker,
    compute_context_token_metrics,
)


# =========================================================================
# Helpers
# =========================================================================


def _make_response(prompt_tokens: int, completion_tokens: int) -> MagicMock:
    """Create a mock response with usage data."""
    response = MagicMock()
    response.usage = MagicMock(
        prompt_tokens=prompt_tokens, completion_tokens=completion_tokens
    )
    return response


def _make_response_no_usage() -> MagicMock:
    """Create a mock response with no usage data."""
    response = MagicMock()
    response.usage = None
    return response


# =========================================================================
# StateUsageTracker per-turn tracking
# =========================================================================


class TestPerTurnTracking:
    def test_per_turn_empty(self):
        tracker = StateUsageTracker()
        assert tracker.per_turn == []

    def test_per_turn_appends_on_increment_from_response(self):
        tracker = StateUsageTracker()
        tracker.increment_from_response(_make_response(100, 20))
        tracker.increment_from_response(_make_response(130, 25))
        assert tracker.per_turn == [(100, 20), (130, 25)]

    def test_per_turn_skips_no_usage(self):
        tracker = StateUsageTracker()
        tracker.increment_from_response(_make_response(100, 20))
        tracker.increment_from_response(_make_response_no_usage())
        tracker.increment_from_response(_make_response(200, 30))
        # No-usage responses are skipped (no append)
        assert tracker.per_turn == [(100, 20), (200, 30)]

    def test_per_turn_does_not_affect_totals(self):
        tracker = StateUsageTracker()
        tracker.increment_from_response(_make_response(100, 20))
        tracker.increment_from_response(_make_response(130, 25))
        snapshot = tracker.snapshot()
        assert snapshot["input_tokens"] == 230
        assert snapshot["output_tokens"] == 45

    def test_manual_increment_does_not_append_per_turn(self):
        """increment() (without response) should not add to per_turn."""
        tracker = StateUsageTracker()
        tracker.increment(input_tokens=100, output_tokens=20)
        assert tracker.per_turn == []


# =========================================================================
# Branch tracking
# =========================================================================


class TestBranchTracking:
    def test_no_branches_by_default(self):
        tracker = StateUsageTracker()
        assert tracker.branch_points == []

    def test_mark_branch(self):
        tracker = StateUsageTracker()
        tracker.increment_from_response(_make_response(100, 20))
        tracker.increment_from_response(_make_response(130, 25))
        tracker.mark_branch()
        tracker.increment_from_response(_make_response(80, 15))
        assert tracker.branch_points == [2]

    def test_multiple_branches(self):
        tracker = StateUsageTracker()
        for i in range(5):
            tracker.increment_from_response(_make_response(100 + i * 10, 20))
            if i in (1, 3):
                tracker.mark_branch()
        assert tracker.branch_points == [2, 4]

    def test_mark_branch_at_start(self):
        tracker = StateUsageTracker()
        tracker.mark_branch()
        assert tracker.branch_points == [0]

    def test_mark_branch_preserves_per_turn(self):
        tracker = StateUsageTracker()
        tracker.increment_from_response(_make_response(100, 20))
        tracker.mark_branch()
        tracker.increment_from_response(_make_response(80, 15))
        assert tracker.per_turn == [(100, 20), (80, 15)]


# =========================================================================
# compute_context_token_metrics — single branch (no summarization)
# =========================================================================


class TestContextMetricsSingleBranch:
    def test_empty_tracker(self):
        tracker = StateUsageTracker()
        metrics = compute_context_token_metrics(tracker)
        assert metrics["longest_context_completion_tokens"] == 0
        assert metrics["longest_context_non_completion_tokens"] == 0

    def test_single_turn(self):
        tracker = StateUsageTracker()
        tracker.increment_from_response(_make_response(100, 20))
        metrics = compute_context_token_metrics(tracker)
        assert metrics["longest_context_completion_tokens"] == 20
        assert metrics["longest_context_non_completion_tokens"] == 100

    def test_multi_turn_growing_context(self):
        """Context grows as conversation progresses."""
        tracker = StateUsageTracker()
        tracker.increment_from_response(_make_response(100, 20))
        tracker.increment_from_response(_make_response(150, 25))
        tracker.increment_from_response(_make_response(215, 30))

        metrics = compute_context_token_metrics(tracker)
        assert metrics["longest_context_completion_tokens"] == 20 + 25 + 30
        assert metrics["longest_context_non_completion_tokens"] == (215 + 30) - (
            20 + 25 + 30
        )

    def test_invariant_total_equals_sum(self):
        """completion + non_completion = last turn's prompt + completion."""
        tracker = StateUsageTracker()
        tracker.increment_from_response(_make_response(100, 20))
        tracker.increment_from_response(_make_response(150, 25))
        tracker.increment_from_response(_make_response(200, 30))

        metrics = compute_context_token_metrics(tracker)
        total = (
            metrics["longest_context_completion_tokens"]
            + metrics["longest_context_non_completion_tokens"]
        )
        assert total == 200 + 30


# =========================================================================
# compute_context_token_metrics — multiple branches (summarization)
# =========================================================================


class TestContextMetricsMultipleBranches:
    def test_two_branches_takes_max(self):
        tracker = StateUsageTracker()
        # Branch 1: turns 0-1
        tracker.increment_from_response(_make_response(100, 20))
        tracker.increment_from_response(_make_response(150, 25))
        tracker.mark_branch()
        # Branch 2: turns 2-3 (context shrunk)
        tracker.increment_from_response(_make_response(80, 15))
        tracker.increment_from_response(_make_response(110, 20))

        metrics = compute_context_token_metrics(tracker)

        # Branch 1: total = 175, completion = 45
        # Branch 2: total = 130, completion = 35
        # Max by total → branch 1
        assert metrics["longest_context_completion_tokens"] == 45
        assert metrics["longest_context_non_completion_tokens"] == 175 - 45

    def test_second_branch_larger(self):
        tracker = StateUsageTracker()
        tracker.increment_from_response(_make_response(100, 20))
        tracker.mark_branch()
        tracker.increment_from_response(_make_response(80, 15))
        tracker.increment_from_response(_make_response(200, 50))
        tracker.increment_from_response(_make_response(300, 40))

        metrics = compute_context_token_metrics(tracker)

        # Branch 2: total = 340, completion = 105
        assert metrics["longest_context_completion_tokens"] == 105
        assert metrics["longest_context_non_completion_tokens"] == 340 - 105

    def test_three_branches(self):
        tracker = StateUsageTracker()
        tracker.increment_from_response(_make_response(100, 10))
        tracker.mark_branch()
        tracker.increment_from_response(_make_response(80, 20))
        tracker.increment_from_response(_make_response(120, 15))
        tracker.mark_branch()
        tracker.increment_from_response(_make_response(60, 10))

        metrics = compute_context_token_metrics(tracker)

        # Branch 2 is longest (total=135)
        assert metrics["longest_context_completion_tokens"] == 35
        assert metrics["longest_context_non_completion_tokens"] == 100

    def test_invariant_holds_with_branches(self):
        tracker = StateUsageTracker()
        tracker.increment_from_response(_make_response(100, 20))
        tracker.increment_from_response(_make_response(150, 25))
        tracker.mark_branch()
        tracker.increment_from_response(_make_response(80, 15))
        tracker.increment_from_response(_make_response(110, 20))

        metrics = compute_context_token_metrics(tracker)
        # Winning branch is branch 1 (total=175)
        total = (
            metrics["longest_context_completion_tokens"]
            + metrics["longest_context_non_completion_tokens"]
        )
        assert total == 175

    def test_empty_branch_at_end(self):
        tracker = StateUsageTracker()
        tracker.increment_from_response(_make_response(100, 20))
        tracker.increment_from_response(_make_response(150, 25))
        tracker.mark_branch()

        metrics = compute_context_token_metrics(tracker)
        assert metrics["longest_context_completion_tokens"] == 45

    def test_branch_at_start(self):
        tracker = StateUsageTracker()
        tracker.mark_branch()
        tracker.increment_from_response(_make_response(100, 20))
        tracker.increment_from_response(_make_response(150, 25))

        metrics = compute_context_token_metrics(tracker)
        assert metrics["longest_context_completion_tokens"] == 45


# =========================================================================
# Metric classes
# =========================================================================


class TestContextTokenMetricClasses:
    def test_longest_context_completion_metric(self):
        from verifiers.utils.metric_utils import LongestContextCompletionTokensMetric

        m = LongestContextCompletionTokensMetric()
        m.add_output({"token_usage": {"longest_context_completion_tokens": 50.0}})
        m.add_output({"token_usage": {"longest_context_completion_tokens": 100.0}})
        assert m.compute() == pytest.approx(75.0)

    def test_longest_context_non_completion_metric(self):
        from verifiers.utils.metric_utils import LongestContextNonCompletionTokensMetric

        m = LongestContextNonCompletionTokensMetric()
        m.add_output({"token_usage": {"longest_context_non_completion_tokens": 150.0}})
        m.add_output({"token_usage": {"longest_context_non_completion_tokens": 250.0}})
        assert m.compute() == pytest.approx(200.0)

    def test_skips_outputs_without_token_usage(self):
        from verifiers.utils.metric_utils import LongestContextCompletionTokensMetric

        m = LongestContextCompletionTokensMetric()
        m.add_output({})
        m.add_output({"token_usage": {}})
        assert m.count == 0
        assert m.compute() == 0.0
