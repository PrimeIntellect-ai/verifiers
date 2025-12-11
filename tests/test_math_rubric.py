"""Tests for the MathRubric class."""

import time

import pytest

import verifiers as vf
from verifiers.utils.async_utils import NullAsyncContext


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
            {"completion": "1", "answer": "1"},
            {"completion": "x + 1", "answer": "1 + x"},
            {"completion": "\\frac{1}{2}", "answer": "0.5"},
        ],
        ids=lambda x: f"{x['completion']} == {x['answer']}",
    )
    async def test_score_valid_answers(self, test_case):
        """Test scoring a single rollout."""

        rubric = vf.MathRubric()

        state = vf.State(
            input=vf.RolloutInput(
                prompt="test prompt",
                answer=test_case["answer"],
                task="test_task",
                example_id=0,
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
        score_sem = NullAsyncContext()

        await rubric.score_rollout(state, score_sem)

        assert state["metrics"]["correct_answer"] == 1.0

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "test_case",
        [
            {"completion": "1", "answer": "2"},
            {"completion": "\\frac{1}{3}", "answer": "0.5"},
        ],
        ids=lambda x: f"{x['completion']} != {x['answer']}",
    )
    async def test_score_invalid_answers(self, test_case):
        """Test scoring a single rollout."""

        rubric = vf.MathRubric()

        state = vf.State(
            input=vf.RolloutInput(
                prompt="test prompt",
                answer=test_case["answer"],
                task="test_task",
                example_id=0,
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
        score_sem = NullAsyncContext()

        await rubric.score_rollout(state, score_sem)

        assert state["metrics"]["correct_answer"] == 0.0

    @pytest.mark.asyncio
    @pytest.mark.parametrize("timeout_seconds", [0.1, 1, 10])
    async def test_timeout(self, timeout_seconds):
        """Test scoring a single rollout."""

        answer = "1"
        # very large input triggers timeout, takes ~2s to parse and verify
        completion = "1" * int(1e6)

        rubric = vf.MathRubric(timeout_seconds=timeout_seconds)

        state = vf.State(
            input=vf.RolloutInput(
                prompt="test prompt",
                answer=answer,
                task="test_task",
                example_id=0,
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
        score_sem = NullAsyncContext()

        start_time = time.time()
        await rubric.score_rollout(state, score_sem)
        end_time = time.time()
        elapsed_time = end_time - start_time
        assert state["metrics"]["correct_answer"] == 0.0

        # Entire function should timeout within timeout + small overhead
        print(f"Time taken: {elapsed_time:.2f}s")
        overhead_seconds = 0.5
        assert elapsed_time < timeout_seconds + overhead_seconds, (
            f"Time taken: {elapsed_time:.2f}s (expected < {timeout_seconds + overhead_seconds}s)"
        )
