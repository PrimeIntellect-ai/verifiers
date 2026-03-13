"""Tests for the OracleRubric class."""

import pytest

import verifiers as vf


class TestOracleRubric:
    """Test cases for the OracleRubric class."""

    def test_oracle_rubric_initialization(self):
        """Test OracleRubric initialization with a simple callable oracle."""
        rubric = vf.OracleRubric(oracle=len)

        assert len(rubric.funcs) == 0
        assert rubric.weights == []
        assert isinstance(rubric.parser, vf.Parser)

    @pytest.mark.asyncio
    async def test_basic_oracle_scoring(self, make_input):
        """Reward function calls oracle directly and derives score from result."""

        async def score_fn(oracle, prompt, completion, answer, state, **kwargs):
            result = await oracle(prompt, completion, answer, state)
            threshold = answer.get("threshold", 0) if isinstance(answer, dict) else 0
            return 1.0 if result >= threshold else 0.0

        rubric = vf.OracleRubric(oracle=len)
        rubric.add_reward_func(score_fn)

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

        assert state["metrics"]["score_fn"] == 1.0

    @pytest.mark.asyncio
    async def test_oracle_fn_receives_backend(self, make_input):
        """oracle_fn receives oracle_backend as oracle kwarg and handles input prep."""

        class MockServer:
            async def evaluate(self, text: str) -> dict:
                return {"score": len(text) / 10.0}

        async def call_backend(oracle, response, **kwargs):
            return await oracle.evaluate(response)

        async def score_fn(oracle, prompt, completion, answer, state, **kwargs):
            result = await oracle(prompt, completion, answer, state)
            return 1.0 if result.get("score", 0.0) >= 0.5 else 0.0

        rubric = vf.OracleRubric(
            oracle=MockServer(),
            oracle_fn=call_backend,
        )
        rubric.add_reward_func(score_fn)

        state = vf.State(
            input=make_input(
                prompt="test prompt",
                answer={},
                task="test_task",
            )
        )
        state["completion"] = "hello world!"
        state["trajectory"] = []
        state["timing"] = {
            "generation_ms": 0.0,
            "scoring_ms": 0.0,
            "total_ms": 0.0,
            "start_time": 0.0,
        }

        await rubric.score_rollout(state)

        assert state["metrics"]["score_fn"] == 1.0

    @pytest.mark.asyncio
    async def test_score_using_answer_dict(self, make_input):
        """Reward function reads answer dict directly without property extractors."""

        async def score_fn(oracle, prompt, completion, answer, state, **kwargs):
            result = await oracle(prompt, completion, answer, state)
            threshold = float(answer.get("threshold", 0)) if isinstance(answer, dict) else 0
            return 1.0 if float(result) >= threshold else 0.0

        rubric = vf.OracleRubric(oracle=len)
        rubric.add_reward_func(score_fn)

        state = vf.State(
            input=make_input(
                prompt="test prompt",
                answer={"threshold": 4},
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

        assert state["metrics"]["score_fn"] == 1.0

    @pytest.mark.asyncio
    async def test_oracle_measurement_is_cached_within_rollout(self, make_input):
        """Oracle backend is called once even when multiple reward functions use oracle."""
        calls = 0

        def oracle_backend(text: str) -> int:
            nonlocal calls
            calls += 1
            return len(text)

        async def score_fn1(oracle, prompt, completion, answer, state, **kwargs):
            result = await oracle(prompt, completion, answer, state)
            return 1.0 if result >= 3 else 0.0

        async def score_fn2(oracle, prompt, completion, answer, state, **kwargs):
            result = await oracle(prompt, completion, answer, state)
            return 1.0 if result >= 3 else 0.0

        rubric = vf.OracleRubric(oracle=oracle_backend)
        rubric.add_reward_func(score_fn1)
        rubric.add_reward_func(score_fn2)

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
        assert state["metrics"]["score_fn1"] == 1.0
        assert state["metrics"]["score_fn2"] == 1.0
