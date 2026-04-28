"""Tests for the per-rollout TimedSpans timing structure."""

import pytest
from datasets import Dataset

from verifiers import Messages, MultiTurnEnv, Parser, Rubric, SingleTurnEnv, State


class TestSingleTurnTiming:
    @pytest.mark.asyncio
    async def test_single_turn_records_one_model_span(self, mock_client, make_input):
        """SingleTurnEnv rollout records exactly one model span, no env span."""
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

        timing = state["timing"]
        assert len(timing.model.spans) == 1
        assert len(timing.env.spans) == 0
        assert 0 < timing.model.spans[0].duration < 10  # seconds


class TestMultiTurnTiming:
    @pytest.mark.asyncio
    async def test_multi_turn_records_model_and_env_spans(
        self, mock_client, make_input
    ):
        """A 2-turn env produces 2 model spans and 1 env span."""

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

        timing = state["timing"]
        assert len(timing.model.spans) == 2
        assert len(timing.env.spans) == 1
        for s in [*timing.model.spans, *timing.env.spans]:
            assert s.duration < 10
