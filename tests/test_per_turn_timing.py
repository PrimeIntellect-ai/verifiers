"""Tests for the flat RolloutTiming.steps timing list."""

import pytest
from datasets import Dataset

from verifiers import Messages, MultiTurnEnv, Parser, Rubric, SingleTurnEnv, State


class TestSingleTurnTiming:
    @pytest.mark.asyncio
    async def test_single_turn_records_one_model_entry(self, mock_client, make_input):
        """SingleTurnEnv rollout records exactly one model entry, no env entry."""
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
        assert steps[0].kind == "model"
        assert steps[0].duration > 0
        assert steps[0].duration < 10  # seconds, not ms


class TestMultiTurnTiming:
    @pytest.mark.asyncio
    async def test_multi_turn_alternates_model_and_env(self, mock_client, make_input):
        """A 2-turn env produces model, env, model entries in execution order."""

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
        kinds = [s.kind for s in steps]
        assert kinds == ["model", "env", "model"]

        # All durations are seconds (small floats)
        for s in steps:
            assert s.duration < 10
