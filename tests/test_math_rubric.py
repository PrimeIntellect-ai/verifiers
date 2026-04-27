"""Tests for the MathRubric class."""

import asyncio

import pytest

import verifiers as vf
from verifiers.rubrics import math_rubric


class TestMathRubric:
    """Test cases for the MathRubric class."""

    def test_math_rubric_initialization_empty(self):
        """Test MathRubric initialization with no parameters."""
        rubric = vf.MathRubric()

        assert rubric.funcs == [rubric.correct_answer]
        assert rubric.weights == [1.0]
        assert isinstance(rubric.parser, vf.MaybeThinkParser)

    def test_math_rubric_initialization_with_kwargs(self):
        """Test MathRubric initialization - kwargs not supported."""
        # MathRubric doesn't accept arbitrary kwargs
        with pytest.raises(TypeError):
            vf.MathRubric(custom_param="test_value", another_param=42)  # type: ignore

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "test_case",
        [
            {"completion": "\\boxed{1}", "answer": "1"},
            {"completion": "\\boxed{x + 1}", "answer": "1 + x"},
            {"completion": "\\boxed{\\frac{1}{2}}", "answer": "0.5"},
        ],
        ids=lambda x: f"{x['completion']} == {x['answer']}",
    )
    async def test_score_valid_answers(self, test_case, make_input):
        """Test scoring a single rollout."""

        rubric = vf.MathRubric()

        state = vf.State(
            input=make_input(
                prompt="test prompt",
                answer=test_case["answer"],
            )
        )
        state["completion"] = test_case["completion"]
        state["trajectory"] = []
        state["timing"] = {
            "generation_ms": 0.0,
            "scoring_ms": 0.0,
            "total_ms": 0.0,
            "start_time": 0.0,
        }

        await rubric.score_rollout(state)

        assert state["metrics"]["correct_answer"] == 1.0

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "test_case",
        [
            {"completion": "\\boxed{1}", "answer": "2"},
            {"completion": "\\boxed{\\frac{1}{3}}", "answer": "0.5"},
        ],
        ids=lambda x: f"{x['completion']} != {x['answer']}",
    )
    async def test_score_invalid_answers(self, test_case, make_input):
        """Test scoring a single rollout."""

        rubric = vf.MathRubric()

        state = vf.State(
            input=make_input(
                prompt="test prompt",
                answer=test_case["answer"],
            )
        )
        state["completion"] = test_case["completion"]
        state["trajectory"] = []
        state["timing"] = {
            "generation_ms": 0.0,
            "scoring_ms": 0.0,
            "total_ms": 0.0,
            "start_time": 0.0,
        }

        await rubric.score_rollout(state)

        assert state["metrics"]["correct_answer"] == 0.0

    @pytest.mark.asyncio
    @pytest.mark.parametrize("timeout_seconds", [0.1, 10])
    async def test_timeout(self, timeout_seconds, make_input):
        """Test scoring a single rollout."""

        # very large input triggers timeout, takes ~2s to parse and verify
        answer = "1" * int(1e5)
        completion = "\\boxed{" + "1" * int(1e5) + "}"

        rubric = vf.MathRubric(
            max_workers=1, timeout_seconds=timeout_seconds, max_verify_chars=int(2e5)
        )

        state = vf.State(
            input=make_input(
                prompt="test prompt",
                answer=answer,
            )
        )
        state["completion"] = completion
        state["trajectory"] = []
        state["timing"] = {
            "generation_ms": 0.0,
            "scoring_ms": 0.0,
            "total_ms": 0.0,
            "start_time": 0.0,
        }

        await rubric.score_rollout(state)

        # rollout should only pass for large timeout
        if timeout_seconds == 10:
            assert state["metrics"]["correct_answer"] == 1.0
        else:
            assert state["metrics"]["correct_answer"] == 0.0


class TestVerifyResponseExceptionHandling:
    """Regression tests for the exception handling in verify_response.

    See commit narrowing ``except BaseException`` to
    ``except (Exception, MathVerifyTimeout)`` so that ``CancelledError``,
    ``KeyboardInterrupt``, and ``SystemExit`` propagate instead of being
    silently reported as a 0.0 score.
    """

    def test_cancellederror_propagates(self, monkeypatch):
        """CancelledError raised during math_verify must propagate, not
        get swallowed and reported as a score of 0.0."""

        def raise_cancelled(*args, **kwargs):
            raise asyncio.CancelledError

        monkeypatch.setattr(math_rubric, "parse", raise_cancelled)

        with pytest.raises(asyncio.CancelledError):
            math_rubric.verify_response(
                response="\\boxed{1}",
                answer="1",
                max_verify_chars=50_000,
                timeout_seconds=5,
            )

    def test_keyboardinterrupt_propagates(self, monkeypatch):
        """KeyboardInterrupt must propagate so Ctrl-C still works during
        scoring."""

        def raise_kbd(*args, **kwargs):
            raise KeyboardInterrupt

        monkeypatch.setattr(math_rubric, "parse", raise_kbd)

        with pytest.raises(KeyboardInterrupt):
            math_rubric.verify_response(
                response="\\boxed{1}",
                answer="1",
                max_verify_chars=50_000,
                timeout_seconds=5,
            )

    def test_math_verify_timeout_returns_zero(self, monkeypatch):
        """A real math_verify.errors.TimeoutException (which inherits from
        BaseException, not Exception) must still be caught and reported as
        a 0.0 score — that's why the catch is wider than just Exception."""
        from math_verify.errors import TimeoutException

        def raise_timeout(*args, **kwargs):
            raise TimeoutException("simulated math_verify timeout")

        monkeypatch.setattr(math_rubric, "parse", raise_timeout)

        score, elapsed = math_rubric.verify_response(
            response="\\boxed{1}",
            answer="1",
            max_verify_chars=50_000,
            timeout_seconds=5,
        )
        assert score == 0.0
        assert elapsed >= 0.0

    def test_regular_exception_returns_zero(self, monkeypatch):
        """A regular Exception from math_verify should continue to be
        swallowed and reported as 0.0 (library-raised something weird)."""

        def raise_exc(*args, **kwargs):
            raise ValueError("simulated parse failure")

        monkeypatch.setattr(math_rubric, "parse", raise_exc)

        score, elapsed = math_rubric.verify_response(
            response="\\boxed{1}",
            answer="1",
            max_verify_chars=50_000,
            timeout_seconds=5,
        )
        assert score == 0.0
        assert elapsed >= 0.0
