"""Tests for the OracleRubric class."""

import pytest

import verifiers as vf


class TestOracleRubric:
    """Test cases for the OracleRubric class."""

    def test_oracle_rubric_initialization(self):
        """Test OracleRubric initialization with a simple callable oracle."""
        rubric = vf.OracleRubric(oracle=len)

        assert rubric.funcs == [rubric.oracle_property, rubric.correct_answer]
        assert rubric.weights == [0.0, 1.0]
        assert isinstance(rubric.parser, vf.Parser)

    @pytest.mark.asyncio
    async def test_threshold_only_scoring(self, make_input):
        """Test threshold-only comparisons against a numeric property."""
        rubric = vf.OracleRubric(oracle=len)

        state = vf.State(
            input=make_input(
                prompt="test prompt",
                answer={"threshold": 5},
                task="test_task",
            )
        )
        state["completion"] = "hello!"
        state["trajectory"] = []
        state["timing"] = {
            "generation_ms": 0.0,
            "scoring_ms": 0.0,
            "total_ms": 0.0,
            "start_time": 0.0,
        }

        await rubric.score_rollout(state)

        assert state["metrics"]["oracle_property"] == 6.0
        assert state["metrics"]["correct_answer"] == 1.0
        assert state["oracle_property_value"] == 6
        assert state["oracle_threshold"] == 5.0

    @pytest.mark.asyncio
    async def test_custom_oracle_pipeline(self, make_input):
        """Test custom oracle invocation and property extraction."""

        class MockAPIServer:
            async def evaluate(self, text: str) -> dict[str, float]:
                return {"score": len(text) / 10}

        rubric = vf.OracleRubric(
            oracle=MockAPIServer(),
            oracle_fn=lambda oracle, oracle_input, **kwargs: oracle.evaluate(
                oracle_input
            ),
            property_extractor=lambda oracle_output, **kwargs: oracle_output["score"],
        )

        state = vf.State(
            input=make_input(
                prompt="test prompt",
                answer={"target": 0.5, "threshold": 0.01},
                task="test_task",
            )
        )
        state["completion"] = "hello"
        state["trajectory"] = []
        state["timing"] = {
            "generation_ms": 0.0,
            "scoring_ms": 0.0,
            "total_ms": 0.0,
            "start_time": 0.0,
        }

        await rubric.score_rollout(state)

        assert state["metrics"]["oracle_property"] == pytest.approx(0.5)
        assert state["metrics"]["correct_answer"] == 1.0
        assert state["oracle_response"] == {"score": 0.5}
        assert state["oracle_match"] is True

    @pytest.mark.asyncio
    async def test_oracle_measurement_is_cached_within_rollout(self, make_input):
        """Test that property measurement is reused across metric and reward calls."""
        calls = 0

        def oracle(text: str) -> int:
            nonlocal calls
            calls += 1
            return len(text)

        rubric = vf.OracleRubric(oracle=oracle)

        state = vf.State(
            input=make_input(
                prompt="test prompt",
                answer={"threshold": 3},
                task="test_task",
            )
        )
        state["completion"] = "abcd"
        state["trajectory"] = []
        state["timing"] = {
            "generation_ms": 0.0,
            "scoring_ms": 0.0,
            "total_ms": 0.0,
            "start_time": 0.0,
        }

        await rubric.score_rollout(state)

        assert calls == 1