"""Tests for per-turn StepTiming on RolloutTiming.steps."""

import pytest
from datasets import Dataset

from verifiers import Messages, MultiTurnEnv, Parser, Rubric, SingleTurnEnv, State


class TestSingleTurnStepTiming:
    @pytest.mark.asyncio
    async def test_single_turn_has_step_timing(self, mock_client, make_input):
        """SingleTurnEnv rollout records one step in RolloutTiming.steps."""
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

        steps = state["timing"]["steps"]
        assert len(steps) == 1
        t = steps[0]
        assert t.model > 0
        assert t.env == 0.0
        assert t.turn == t.model

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

        step = state["timing"]["steps"][0]
        assert step.model < 10
        assert step.turn < 10


class TestMultiTurnStepTiming:
    @pytest.mark.asyncio
    async def test_multi_turn_backfills_env_timing(self, mock_client, make_input):
        """In a 2-turn env, step 0 gets env backfilled > 0, last step has env == 0."""

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

        steps = state["timing"]["steps"]
        assert len(steps) == 2

        # step 0 should have env backfilled from the get_prompt_messages call
        # that produced step 1's prompt
        assert steps[0].env > 0
        assert steps[0].turn >= steps[0].model

        # last step should have env == 0 (no subsequent get_prompt_messages)
        assert steps[1].env == 0.0
        assert steps[1].turn == steps[1].model

        # All values should be seconds (small floats)
        for s in steps:
            assert s.model < 10
            assert s.env < 10
            assert s.turn < 10
