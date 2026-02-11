"""Tests for the MathRubric class."""

import pytest

import verifiers as vf


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
    async def test_score_valid_answers(self, test_case, make_input):
        """Test scoring a single rollout."""

        rubric = vf.MathRubric()

        state = vf.State(
            input=make_input(
                prompt="test prompt",
                answer=test_case["answer"],
                task="test_task",
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
            {"completion": "1", "answer": "2"},
            {"completion": "\\frac{1}{3}", "answer": "0.5"},
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
                task="test_task",
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
        completion = "1" * int(1e5)

        rubric = vf.MathRubric(max_workers=1, timeout_seconds=timeout_seconds)

        state = vf.State(
            input=make_input(
                prompt="test prompt",
                answer=answer,
                task="test_task",
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
