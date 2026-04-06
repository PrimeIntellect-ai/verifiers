"""Tests for per-turn context token metrics.

Tests the trajectory-based context token computation
(input_tokens, output_tokens) which assumes a linear rollout
using the last trajectory step.
"""

from unittest.mock import MagicMock

import pytest

from verifiers.utils.usage_utils import compute_context_token_metrics


# =========================================================================
# Helpers
# =========================================================================

SYS = {"role": "system", "content": "You are helpful"}
USER = {"role": "user", "content": "hi"}


def _make_response(prompt_tokens: int, completion_tokens: int) -> MagicMock:
    response = MagicMock()
    response.usage = MagicMock(
        prompt_tokens=prompt_tokens, completion_tokens=completion_tokens
    )
    return response


def _asst(i: int) -> dict:
    return {"role": "assistant", "content": f"response {i}"}


# =========================================================================
# compute_context_token_metrics
# =========================================================================


class TestContextMetrics:
    def test_empty_trajectory(self):
        metrics = compute_context_token_metrics([])
        assert metrics["output_tokens"] == 0
        assert metrics["input_tokens"] == 0

    def test_single_turn(self):
        trajectory = [
            {
                "prompt": [SYS, USER],
                "completion": [_asst(0)],
                "response": _make_response(100, 20),
            },
        ]
        metrics = compute_context_token_metrics(trajectory)
        assert metrics["output_tokens"] == 20
        assert metrics["input_tokens"] == 100

    def test_multi_turn(self):
        trajectory = [
            {
                "response": _make_response(100, 20),
            },
            {
                "response": _make_response(150, 25),
            },
            {
                "response": _make_response(200, 30),
            },
        ]
        metrics = compute_context_token_metrics(trajectory)
        # Last step total = 200 + 30 = 230
        # Sum of completion tokens = 20 + 25 + 30 = 75
        assert metrics["output_tokens"] == 75
        assert metrics["input_tokens"] == 230 - 75

    def test_invariant_total_equals_last_step(self):
        trajectory = [
            {"response": _make_response(100, 20)},
            {"response": _make_response(150, 25)},
            {"response": _make_response(200, 30)},
        ]
        metrics = compute_context_token_metrics(trajectory)
        total = metrics["output_tokens"] + metrics["input_tokens"]
        # Total should equal last step's prompt_tokens + completion_tokens
        assert total == 200 + 30

    def test_no_response_on_any_step(self):
        trajectory = [{"response": None}]
        metrics = compute_context_token_metrics(trajectory)
        assert metrics["output_tokens"] == 0
        assert metrics["input_tokens"] == 0

    def test_last_step_used_not_largest(self):
        """Even if an earlier step has a larger context, we use the last step."""
        trajectory = [
            {"response": _make_response(500, 100)},  # larger context
            {"response": _make_response(100, 20)},  # last step, smaller
        ]
        metrics = compute_context_token_metrics(trajectory)
        # Last step total = 120, sum completions = 100 + 20 = 120
        assert metrics["output_tokens"] == 120
        assert metrics["input_tokens"] == 0  # clamped to 0

    def test_skips_none_responses_for_last_step(self):
        """Last step with response=None is skipped; uses previous step."""
        trajectory = [
            {"response": _make_response(100, 20)},
            {"response": _make_response(200, 30)},
            {"response": None},
        ]
        metrics = compute_context_token_metrics(trajectory)
        # Last step with response is step 1: total = 230
        # Sum completions from all steps with responses: 20 + 30 = 50
        assert metrics["output_tokens"] == 50
        assert metrics["input_tokens"] == 230 - 50

    def test_input_tokens_clamped_to_zero(self):
        """If sum of completions exceeds last step total, input is clamped to 0."""
        trajectory = [
            {"response": _make_response(10, 500)},  # huge completion
            {"response": _make_response(50, 10)},
        ]
        metrics = compute_context_token_metrics(trajectory)
        # Last step total = 60, sum completions = 510
        assert metrics["output_tokens"] == 510
        assert metrics["input_tokens"] == 0


# =========================================================================
# Metric classes
# =========================================================================


class TestContextTokenMetricClasses:
    def test_input_tokens_metric(self):
        from verifiers.utils.metric_utils import InputTokensMetric

        m = InputTokensMetric()
        m.add_output({"token_usage": {"input_tokens": 50.0}})
        m.add_output({"token_usage": {"input_tokens": 100.0}})
        assert m.compute() == pytest.approx(75.0)

    def test_output_tokens_metric(self):
        from verifiers.utils.metric_utils import OutputTokensMetric

        m = OutputTokensMetric()
        m.add_output({"token_usage": {"output_tokens": 150.0}})
        m.add_output({"token_usage": {"output_tokens": 250.0}})
        assert m.compute() == pytest.approx(200.0)

    def test_prefill_tokens_metric(self):
        from verifiers.utils.metric_utils import PrefillTokensMetric

        m = PrefillTokensMetric()
        m.add_output({"token_usage": {"prefill_tokens": 100.0}})
        m.add_output({"token_usage": {"prefill_tokens": 200.0}})
        assert m.compute() == pytest.approx(150.0)

    def test_decode_tokens_metric(self):
        from verifiers.utils.metric_utils import DecodeTokensMetric

        m = DecodeTokensMetric()
        m.add_output({"token_usage": {"decode_tokens": 40.0}})
        m.add_output({"token_usage": {"decode_tokens": 60.0}})
        assert m.compute() == pytest.approx(50.0)

    def test_skips_outputs_without_token_usage(self):
        from verifiers.utils.metric_utils import InputTokensMetric

        m = InputTokensMetric()
        m.add_output({})
        m.add_output({"token_usage": {}})
        assert m.count == 0
        assert m.compute() == 0.0
