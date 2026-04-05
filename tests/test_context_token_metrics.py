"""Tests for per-turn context token metrics.

Tests the trajectory-based context token computation
(longest_context_completion_tokens, longest_context_non_completion_tokens).

The algorithm uses message-prefix matching (same approach as best-effort
TITO) to automatically detect which prior completions are still in context,
handling context dropping, branching, and history rewriting without
requiring trajectory_id filtering.
"""

from unittest.mock import MagicMock

import pytest

from verifiers.utils.usage_utils import compute_context_token_metrics


# =========================================================================
# Helpers
# =========================================================================

# Shared message building blocks
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


def _tool(i: int) -> dict:
    return {"role": "tool", "tool_call_id": f"t{i}", "content": "ok"}


def _user(i: int) -> dict:
    return {"role": "user", "content": f"follow-up {i}"}


# =========================================================================
# compute_context_token_metrics — no context dropping
# =========================================================================


class TestContextMetricsNoBranching:
    def test_empty_trajectory(self):
        metrics = compute_context_token_metrics([])
        assert metrics["longest_context_completion_tokens"] == 0
        assert metrics["longest_context_non_completion_tokens"] == 0

    def test_single_turn(self):
        trajectory = [
            {
                "prompt": [SYS, USER],
                "completion": [_asst(0)],
                "response": _make_response(100, 20),
            },
        ]
        metrics = compute_context_token_metrics(trajectory)
        assert metrics["longest_context_completion_tokens"] == 20
        assert metrics["longest_context_non_completion_tokens"] == 100

    def test_multi_turn_all_in_context(self):
        # Each step's prompt starts with the previous step's prompt + completion
        trajectory = [
            {
                "prompt": [SYS, USER],
                "completion": [_asst(0)],
                "response": _make_response(100, 20),
            },
            {
                "prompt": [SYS, USER, _asst(0), _tool(0)],
                "completion": [_asst(1)],
                "response": _make_response(150, 25),
            },
            {
                "prompt": [SYS, USER, _asst(0), _tool(0), _asst(1), _tool(1)],
                "completion": [_asst(2)],
                "response": _make_response(215, 30),
            },
        ]
        metrics = compute_context_token_metrics(trajectory)
        # All 3 steps form a chain: step0 → step1 → step2
        assert metrics["longest_context_completion_tokens"] == 20 + 25 + 30
        assert metrics["longest_context_non_completion_tokens"] == (215 + 30) - (
            20 + 25 + 30
        )

    def test_completion_equals_cumulative_decode_without_branching(self):
        trajectory = [
            {
                "prompt": [SYS, USER],
                "completion": [_asst(0)],
                "response": _make_response(100, 20),
            },
            {
                "prompt": [SYS, USER, _asst(0), _tool(0)],
                "completion": [_asst(1)],
                "response": _make_response(150, 25),
            },
            {
                "prompt": [SYS, USER, _asst(0), _tool(0), _asst(1), _tool(1)],
                "completion": [_asst(2)],
                "response": _make_response(200, 30),
            },
        ]
        metrics = compute_context_token_metrics(trajectory)
        cumulative_decode = 20 + 25 + 30
        assert metrics["longest_context_completion_tokens"] == cumulative_decode

    def test_invariant_total_equals_sum(self):
        trajectory = [
            {
                "prompt": [SYS, USER],
                "completion": [_asst(0)],
                "response": _make_response(100, 20),
            },
            {
                "prompt": [SYS, USER, _asst(0), _tool(0)],
                "completion": [_asst(1)],
                "response": _make_response(150, 25),
            },
            {
                "prompt": [SYS, USER, _asst(0), _tool(0), _asst(1), _tool(1)],
                "completion": [_asst(2)],
                "response": _make_response(200, 30),
            },
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
        """After summarization, the prompt no longer starts with the prior
        step's prompt+completion, so the prefix match breaks the chain."""
        summary = {"role": "system", "content": "Summary of prior turns"}
        trajectory = [
            {
                "prompt": [SYS, USER],
                "completion": [_asst(0)],
                "response": _make_response(100, 20),
            },
            {
                "prompt": [SYS, USER, _asst(0), _tool(0)],
                "completion": [_asst(1)],
                "response": _make_response(150, 25),
            },
            {
                "prompt": [SYS, USER, _asst(0), _tool(0), _asst(1), _tool(1)],
                "completion": [_asst(2)],
                "response": _make_response(200, 30),
            },
            # After summarization: prompt is completely different, no prefix match
            {
                "prompt": [SYS, summary, _asst(2), _tool(2)],
                "completion": [_asst(3)],
                "response": _make_response(120, 15),
            },
        ]
        metrics = compute_context_token_metrics(trajectory)
        # Step2 has the largest total context (230). Its chain: step2 → step1 → step0
        assert metrics["longest_context_completion_tokens"] == 20 + 25 + 30
        assert metrics["longest_context_non_completion_tokens"] == (200 + 30) - (
            20 + 25 + 30
        )

    def test_all_prior_turns_dropped(self):
        """Aggressive summarization: prompt has no prefix match to any prior step."""
        summary = {"role": "system", "content": "Everything summarized"}
        trajectory = [
            {
                "prompt": [SYS, USER],
                "completion": [_asst(0)],
                "response": _make_response(100, 20),
            },
            {
                "prompt": [SYS, USER, _asst(0), _tool(0)],
                "completion": [_asst(1)],
                "response": _make_response(150, 25),
            },
            # After summarization: completely new prompt
            {
                "prompt": [SYS, summary],
                "completion": [_asst(2)],
                "response": _make_response(80, 10),
            },
        ]
        metrics = compute_context_token_metrics(trajectory)
        # Step1 has the largest context (175). Chain: step1 → step0
        assert metrics["longest_context_completion_tokens"] == 20 + 25
        assert metrics["longest_context_non_completion_tokens"] == (150 + 25) - (
            20 + 25
        )

    def test_no_response_on_any_step(self):
        trajectory = [{"prompt": [], "completion": [], "response": None}]
        metrics = compute_context_token_metrics(trajectory)
        assert metrics["longest_context_completion_tokens"] == 0
        assert metrics["longest_context_non_completion_tokens"] == 0


# =========================================================================
# compute_context_token_metrics — with branching (e.g. multi-agent)
# =========================================================================


class TestContextMetricsWithBranching:
    def test_interleaved_branches_picks_longest(self):
        """Two agents interleaved in the same trajectory. The algorithm
        should find the branch with the largest context, not be confused
        by interleaving."""
        # Agent A: 2 turns
        agent_a_sys = {"role": "system", "content": "You are agent A"}
        agent_a_user = {"role": "user", "content": "task for A"}
        agent_a_asst0 = {"role": "assistant", "content": "A turn 0"}
        agent_a_tool0 = {"role": "tool", "tool_call_id": "a0", "content": "ok"}
        agent_a_asst1 = {"role": "assistant", "content": "A turn 1"}

        # Agent B: 1 turn (shorter context)
        agent_b_sys = {"role": "system", "content": "You are agent B"}
        agent_b_user = {"role": "user", "content": "task for B"}
        agent_b_asst0 = {"role": "assistant", "content": "B turn 0"}

        trajectory = [
            # Agent A turn 0
            {
                "prompt": [agent_a_sys, agent_a_user],
                "completion": [agent_a_asst0],
                "response": _make_response(100, 30),
            },
            # Agent B turn 0 (different prompt — no prefix match to A)
            {
                "prompt": [agent_b_sys, agent_b_user],
                "completion": [agent_b_asst0],
                "response": _make_response(80, 20),
            },
            # Agent A turn 1 (continues from A turn 0)
            {
                "prompt": [agent_a_sys, agent_a_user, agent_a_asst0, agent_a_tool0],
                "completion": [agent_a_asst1],
                "response": _make_response(160, 40),
            },
        ]
        metrics = compute_context_token_metrics(trajectory)
        # Agent A's chain: step2 → step0 (prefix match skips step1)
        # Total context: 160 + 40 = 200
        # Completion: 30 + 40 = 70
        assert metrics["longest_context_completion_tokens"] == 30 + 40
        assert metrics["longest_context_non_completion_tokens"] == (160 + 40) - (
            30 + 40
        )

    def test_sub_llm_steps_not_counted_in_main_context(self):
        """Sub-LLM steps with different prompts don't form a prefix match
        with main model steps — they're automatically excluded."""
        main_asst0 = {"role": "assistant", "content": "main turn 0"}
        main_tool0 = {"role": "tool", "tool_call_id": "m0", "content": "ok"}
        main_asst1 = {"role": "assistant", "content": "main turn 1"}

        sub_sys = {"role": "system", "content": "sub-LLM system"}
        sub_user = {"role": "user", "content": "sub task"}
        sub_asst = {"role": "assistant", "content": "sub response"}

        trajectory = [
            # Main turn 0
            {
                "prompt": [SYS, USER],
                "completion": [main_asst0],
                "response": _make_response(100, 20),
            },
            # Sub-LLM call (independent prompt)
            {
                "prompt": [sub_sys, sub_user],
                "completion": [sub_asst],
                "response": _make_response(50, 15),
            },
            # Main turn 1 (continues from main turn 0)
            {
                "prompt": [SYS, USER, main_asst0, main_tool0],
                "completion": [main_asst1],
                "response": _make_response(150, 25),
            },
        ]
        metrics = compute_context_token_metrics(trajectory)
        # Main chain: step2 → step0 (step1 is sub-LLM, no prefix match)
        # Total context: 150 + 25 = 175
        # Completion: 20 + 25 = 45
        assert metrics["longest_context_completion_tokens"] == 20 + 25
        assert metrics["longest_context_non_completion_tokens"] == (150 + 25) - (
            20 + 25
        )


# =========================================================================
# Edge cases
# =========================================================================


class TestContextMetricsEdgeCases:
    def test_pydantic_messages_normalized(self):
        """Pydantic message objects should be normalized to dicts for comparison."""

        class FakeMsg:
            def __init__(self, role, content):
                self.role = role
                self.content = content

            def model_dump(self):
                return {"role": self.role, "content": self.content}

        trajectory = [
            {
                "prompt": [FakeMsg("system", "You are helpful"), FakeMsg("user", "hi")],
                "completion": [FakeMsg("assistant", "hello")],
                "response": _make_response(100, 20),
            },
            {
                # Second step's prompt uses dicts — should still prefix-match
                "prompt": [
                    {"role": "system", "content": "You are helpful"},
                    {"role": "user", "content": "hi"},
                    {"role": "assistant", "content": "hello"},
                    {"role": "tool", "tool_call_id": "t0", "content": "ok"},
                ],
                "completion": [{"role": "assistant", "content": "world"}],
                "response": _make_response(150, 25),
            },
        ]
        metrics = compute_context_token_metrics(trajectory)
        # Chain: step1 → step0
        assert metrics["longest_context_completion_tokens"] == 20 + 25

    def test_single_step_no_response(self):
        """Step with no response should return zeros."""
        trajectory = [
            {"prompt": [SYS, USER], "completion": [_asst(0)], "response": None}
        ]
        metrics = compute_context_token_metrics(trajectory)
        assert metrics["longest_context_completion_tokens"] == 0
        assert metrics["longest_context_non_completion_tokens"] == 0

    def test_history_rewriting_breaks_chain(self):
        """If an env rewrites message history, the prefix won't match."""
        rewritten_user = {"role": "user", "content": "rewritten prompt"}
        trajectory = [
            {
                "prompt": [SYS, USER],
                "completion": [_asst(0)],
                "response": _make_response(100, 20),
            },
            # Env rewrote history — prompt doesn't start with step0's messages
            {
                "prompt": [SYS, rewritten_user, _asst(0), _tool(0)],
                "completion": [_asst(1)],
                "response": _make_response(150, 25),
            },
        ]
        metrics = compute_context_token_metrics(trajectory)
        # Step1 has largest context (175) but no parent (prefix mismatch)
        # Only step1's own completion counts
        assert metrics["longest_context_completion_tokens"] == 25
        assert metrics["longest_context_non_completion_tokens"] == (150 + 25) - 25


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
