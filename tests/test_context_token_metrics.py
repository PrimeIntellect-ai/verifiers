"""Tests for per-turn context token metrics.

Tests the trajectory-based context token computation
(longest_context_completion_tokens, longest_context_non_completion_tokens).
"""

from unittest.mock import MagicMock

import pytest

from verifiers.utils.usage_utils import compute_context_token_metrics


# =========================================================================
# Helpers
# =========================================================================


def _make_response(prompt_tokens: int, completion_tokens: int) -> MagicMock:
    response = MagicMock()
    response.usage = MagicMock(
        prompt_tokens=prompt_tokens, completion_tokens=completion_tokens
    )
    return response


def _make_response_no_usage() -> MagicMock:
    response = MagicMock()
    response.usage = None
    return response


def _make_step(
    prompt_tokens: int,
    completion_tokens: int,
    prompt_assistant_count: int = 0,
) -> dict:
    """Create a trajectory step with a response and prompt messages.

    prompt_assistant_count: number of assistant messages in the prompt
    (representing prior turns still in context).
    """
    prompt_messages = [{"role": "user", "content": "hi"}]
    for i in range(prompt_assistant_count):
        prompt_messages.append({"role": "assistant", "content": f"response {i}"})
        prompt_messages.append(
            {"role": "tool", "tool_call_id": f"t{i}", "content": "ok"}
        )
    return {
        "prompt": prompt_messages,
        "completion": [{"role": "assistant", "content": "reply"}],
        "response": _make_response(prompt_tokens, completion_tokens),
    }


# =========================================================================
# StateUsageTracker per-turn tracking
# =========================================================================
# compute_context_token_metrics — no context dropping
# =========================================================================


class TestContextMetricsNoBranching:
    def test_empty_trajectory(self):
        metrics = compute_context_token_metrics([])
        assert metrics["longest_context_completion_tokens"] == 0
        assert metrics["longest_context_non_completion_tokens"] == 0

    def test_single_turn(self):
        trajectory = [_make_step(100, 20, prompt_assistant_count=0)]
        metrics = compute_context_token_metrics(trajectory)
        # 1 step in context (the step itself), completion = 20
        assert metrics["longest_context_completion_tokens"] == 20
        # non-completion = (100 + 20) - 20 = 100
        assert metrics["longest_context_non_completion_tokens"] == 100

    def test_multi_turn_all_in_context(self):
        trajectory = [
            _make_step(100, 20, prompt_assistant_count=0),
            _make_step(150, 25, prompt_assistant_count=1),
            _make_step(215, 30, prompt_assistant_count=2),
        ]
        metrics = compute_context_token_metrics(trajectory)
        # All 3 steps in context (2 assistants in prompt + the step itself)
        assert metrics["longest_context_completion_tokens"] == 20 + 25 + 30
        assert metrics["longest_context_non_completion_tokens"] == (215 + 30) - (
            20 + 25 + 30
        )

    def test_completion_equals_cumulative_decode_without_branching(self):
        trajectory = [
            _make_step(100, 20, prompt_assistant_count=0),
            _make_step(150, 25, prompt_assistant_count=1),
            _make_step(200, 30, prompt_assistant_count=2),
        ]
        metrics = compute_context_token_metrics(trajectory)
        cumulative_decode = 20 + 25 + 30
        assert metrics["longest_context_completion_tokens"] == cumulative_decode

    def test_invariant_total_equals_sum(self):
        trajectory = [
            _make_step(100, 20, prompt_assistant_count=0),
            _make_step(150, 25, prompt_assistant_count=1),
            _make_step(200, 30, prompt_assistant_count=2),
        ]
        metrics = compute_context_token_metrics(trajectory)
        total = (
            metrics["longest_context_completion_tokens"]
            + metrics["longest_context_non_completion_tokens"]
        )
        assert total == 200 + 30


# =========================================================================
# compute_context_token_metrics — with context dropping (summarization)
# =========================================================================


class TestContextMetricsWithContextDropping:
    def test_dropped_turns_not_counted(self):
        """When turns are dropped, the last prompt has fewer assistant messages."""
        trajectory = [
            _make_step(100, 20, prompt_assistant_count=0),
            _make_step(150, 25, prompt_assistant_count=1),
            _make_step(200, 30, prompt_assistant_count=2),
            # After summarization: only 1 prior assistant in prompt (turns dropped)
            _make_step(120, 15, prompt_assistant_count=1),
        ]
        metrics = compute_context_token_metrics(trajectory)
        # Last step sees 1 assistant in prompt + itself = 2 steps in context
        # Those are the last 2 steps: (200, 30) and (120, 15)
        assert metrics["longest_context_completion_tokens"] == 30 + 15
        assert metrics["longest_context_non_completion_tokens"] == (120 + 15) - (
            30 + 15
        )

    def test_all_prior_turns_dropped(self):
        """Aggressive summarization: no prior assistants in prompt."""
        trajectory = [
            _make_step(100, 20, prompt_assistant_count=0),
            _make_step(150, 25, prompt_assistant_count=1),
            # After summarization: 0 prior assistants
            _make_step(80, 10, prompt_assistant_count=0),
        ]
        metrics = compute_context_token_metrics(trajectory)
        # Only the last step itself: completion = 10
        assert metrics["longest_context_completion_tokens"] == 10
        assert metrics["longest_context_non_completion_tokens"] == (80 + 10) - 10

    def test_no_response_on_last_step(self):
        trajectory = [{"prompt": [], "completion": [], "response": None}]
        metrics = compute_context_token_metrics(trajectory)
        assert metrics["longest_context_completion_tokens"] == 0
        assert metrics["longest_context_non_completion_tokens"] == 0


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
        from verifiers.utils.metric_utils import (
            LongestContextNonCompletionTokensMetric,
        )

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
