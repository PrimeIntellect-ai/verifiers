"""Tests for per-turn StepTiming on TrajectoryStep."""

import pytest
from datasets import Dataset

from verifiers import Messages, MultiTurnEnv, Parser, Rubric, SingleTurnEnv, State


class TestSingleTurnStepTiming:
    @pytest.mark.asyncio
    async def test_single_turn_has_step_timing(self, mock_client, make_input):
        """SingleTurnEnv rollout produces a step with timing."""
        dataset = Dataset.from_dict({"question": ["q1"], "answer": ["a1"]})
        env = SingleTurnEnv(
            client=mock_client,
            model="test-model",
            dataset=dataset,
            rubric=Rubric(),
        )
        mock_client.set_default_response("hello")

        state = await env.rollout(
            input=make_input(prompt=[{"role": "user", "content": "q1"}], answer="a1"),
            client=mock_client,
            model="test-model",
        )

        assert len(state["trajectory"]) == 1
        step = state["trajectory"][0]
        assert "timing" in step
        t = step["timing"]
        assert t["model_s"] > 0
        assert t["env_s"] == 0.0
        assert t["turn_s"] == t["model_s"]

    @pytest.mark.asyncio
    async def test_timing_values_are_seconds(self, mock_client, make_input):
        """Assert timing values are small floats (seconds, not ms)."""
        dataset = Dataset.from_dict({"question": ["q1"], "answer": ["a1"]})
        env = SingleTurnEnv(
            client=mock_client,
            model="test-model",
            dataset=dataset,
            rubric=Rubric(),
        )
        mock_client.set_default_response("hello")

        state = await env.rollout(
            input=make_input(prompt=[{"role": "user", "content": "q1"}], answer="a1"),
            client=mock_client,
            model="test-model",
        )

        step = state["trajectory"][0]
        assert step["timing"]["model_s"] < 10
        assert step["timing"]["turn_s"] < 10


class TestMultiTurnStepTiming:
    @pytest.mark.asyncio
    async def test_multi_turn_backfills_env_timing(self, mock_client, make_input):
        """In a 2-turn env, step 0 gets env_s backfilled > 0, last step has env_s == 0."""

        class TwoTurnEnv(MultiTurnEnv):
            def __init__(self, **kwargs):
                super().__init__(max_turns=2, **kwargs)

            async def env_response(self, messages: Messages, state: State, **kwargs):
                return [{"role": "user", "content": "follow-up"}]

        dataset = Dataset.from_dict({"question": ["q1"], "answer": ["a1"]})
        env = TwoTurnEnv(
            client=mock_client,
            model="test-model",
            dataset=dataset,
            parser=Parser(),
            rubric=Rubric(),
        )
        mock_client.set_default_response("response")

        state = await env.rollout(
            input=make_input(prompt=[{"role": "user", "content": "q1"}], answer="a1"),
            client=mock_client,
            model="test-model",
        )

        assert len(state["trajectory"]) == 2

        step0 = state["trajectory"][0]
        step1 = state["trajectory"][1]

        # step 0 should have env_s backfilled from the get_prompt_messages call
        # that produced step 1's prompt
        assert "timing" in step0
        assert step0["timing"]["env_s"] > 0
        assert step0["timing"]["turn_s"] >= step0["timing"]["model_s"]

        # last step should have env_s == 0 (no subsequent get_prompt_messages)
        assert "timing" in step1
        assert step1["timing"]["env_s"] == 0.0
        assert step1["timing"]["turn_s"] == step1["timing"]["model_s"]

        # All values should be seconds (small floats)
        for step in state["trajectory"]:
            assert step["timing"]["model_s"] < 10
            assert step["timing"]["env_s"] < 10
            assert step["timing"]["turn_s"] < 10
