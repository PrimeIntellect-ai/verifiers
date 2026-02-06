#!/usr/bin/env python3
"""Tests for bug fixes in the unified event system.

Tests for two specific bugs:
1. Server mode bypass in grouped scoring path
2. Missing documentation (tested manually)

Also includes coverage tests for correct num_examples calculation.
"""

import asyncio
import pytest
from unittest.mock import Mock, AsyncMock, patch
from datasets import Dataset

import verifiers as vf
from verifiers.types import StartEvent, EvalEvent
from verifiers.envs.environment import ClientConfig


class EventCollector:
    """Simple event collector for testing."""

    def __init__(self):
        self.events: list[EvalEvent] = []

    async def __call__(self, event: EvalEvent) -> None:
        self.events.append(event)


class TestBugFix1ServerModeBypass:
    """Test for Bug 1: Server mode bypass in grouped scoring path.

    When independent_scoring=False, the code was calling _run_group_with_states()
    directly, bypassing the server mode check in run_group().
    """

    @pytest.mark.asyncio
    async def test_grouped_scoring_with_server_mode(self):
        """Test that grouped scoring respects server mode."""
        # Create a simple dataset
        dataset = Dataset.from_dict({
            "prompt": ["Q1?", "Q2?"]
        })

        # Create environment with rubric
        env = vf.SingleTurnEnv(dataset=dataset)

        # Mock env_client to simulate server mode
        mock_env_client = Mock()
        mock_env_client.run_group = AsyncMock(return_value=[
            {"prompt": "Q1?", "completion": "A1", "reward": 1.0},
            {"prompt": "Q1?", "completion": "A1b", "reward": 1.0},
        ])

        # Inject mock env_client
        env.env_client = mock_env_client

        # Create mock client config (for server mode)
        client_config = ClientConfig(
            model="test-model",
            base_url="http://test",
            api_key="test-key"
        )

        collector = EventCollector()

        # Run evaluation with grouped scoring (independent_scoring=False)
        results = await env.evaluate(
            client=client_config,
            model="test-model",
            num_examples=2,
            rollouts_per_example=2,
            independent_scoring=False,  # This should trigger grouped scoring
            on_event=collector,
        )

        # Verify that run_group was called (not _run_group_with_states)
        # This proves that the server mode check is not bypassed
        assert mock_env_client.run_group.called, "run_group should be called in server mode"
        assert mock_env_client.run_group.call_count == 2, "run_group should be called for each group"


    @pytest.mark.asyncio
    async def test_grouped_scoring_without_server_mode(self):
        """Test that grouped scoring works correctly in local mode."""
        # Create a simple dataset
        dataset = Dataset.from_dict({
            "prompt": ["Q1?", "Q2?"]
        })

        # Create environment
        env = vf.SingleTurnEnv(dataset=dataset)

        # Make sure env_client is None (local mode)
        assert env.env_client is None

        # Mock client
        class MockClient:
            async def chat(self, *args, **kwargs):
                class MockResponse:
                    choices = [
                        type(
                            "Choice",
                            (),
                            {
                                "message": type(
                                    "Message", (), {"content": "answer", "role": "assistant"}
                                )(),
                                "finish_reason": "stop",
                            },
                        )()
                    ]
                    usage = type("Usage", (), {"prompt_tokens": 10, "completion_tokens": 5})()

                return MockResponse()

        mock_client = type("MockClient", (), {"chat": MockClient().chat})()

        collector = EventCollector()

        # Run evaluation with grouped scoring
        results = await env.evaluate(
            client=mock_client,
            model="test-model",
            num_examples=2,
            rollouts_per_example=2,
            independent_scoring=False,
            on_event=collector,
        )

        # Verify GroupCompleteEvent was emitted (only happens with _run_group_with_states)
        group_complete_events = [e for e in collector.events if e["type"] == "group_complete"]
        assert len(group_complete_events) == 2, "Should have 2 GroupCompleteEvents"

        # Verify each group has states
        for ge in group_complete_events:
            assert "states" in ge, "GroupCompleteEvent should have states in local mode"
            assert len(ge["states"]) == 2, "Each group should have 2 states"


class TestNumExamplesCalculation:
    """Test coverage for correct num_examples calculation.

    Verifies that num_examples is correctly calculated using len(set([example_id]))
    which properly handles duplicate example_ids when rollouts_per_example > 1.
    """

    @pytest.mark.asyncio
    async def test_num_examples_with_independent_scoring_and_duplicates(self):
        """Test that num_examples is calculated correctly when rollouts_per_example > 1."""
        # Create a dataset with 3 examples
        dataset = Dataset.from_dict({
            "prompt": ["Q1?", "Q2?", "Q3?"]
        })

        # Simple reward function
        def always_correct(prompt, completion, **kwargs):
            return 1.0

        rubric = vf.Rubric(funcs=[always_correct])
        env = vf.SingleTurnEnv(dataset=dataset, rubric=rubric)

        # Mock client
        class MockClient:
            async def chat(self, *args, **kwargs):
                class MockResponse:
                    choices = [
                        type(
                            "Choice",
                            (),
                            {
                                "message": type(
                                    "Message", (), {"content": "answer", "role": "assistant"}
                                )(),
                                "finish_reason": "stop",
                            },
                        )()
                    ]
                    usage = type("Usage", (), {"prompt_tokens": 10, "completion_tokens": 5})()

                return MockResponse()

        mock_client = type("MockClient", (), {"chat": MockClient().chat})()

        collector = EventCollector()

        # Run evaluation with independent scoring and multiple rollouts per example
        results = await env.evaluate(
            client=mock_client,
            model="test-model",
            num_examples=3,
            rollouts_per_example=4,  # 4 rollouts per example = 12 total rollouts
            independent_scoring=True,
            on_event=collector,
        )

        # Get the StartEvent
        start_events = [e for e in collector.events if e["type"] == "start"]
        assert len(start_events) == 1, "Should have exactly one StartEvent"

        start_event = start_events[0]

        # Verify the counts are correct
        assert start_event["num_examples"] == 3, f"num_examples should be 3, got {start_event['num_examples']}"
        assert start_event["rollouts_per_example"] == 4, f"rollouts_per_example should be 4, got {start_event['rollouts_per_example']}"
        assert start_event["total_rollouts"] == 12, f"total_rollouts should be 12, got {start_event['total_rollouts']}"

        # Verify metadata in results
        assert results["metadata"]["num_examples"] == 3
        assert results["metadata"]["rollouts_per_example"] == 4
        assert len(results["outputs"]) == 12


    @pytest.mark.asyncio
    async def test_num_examples_with_independent_scoring_single_rollout(self):
        """Test that num_examples is correct when rollouts_per_example = 1."""
        # Create a dataset with 5 examples
        dataset = Dataset.from_dict({
            "prompt": [f"Q{i}?" for i in range(5)]
        })

        # Simple reward function
        def always_correct(prompt, completion, **kwargs):
            return 1.0

        rubric = vf.Rubric(funcs=[always_correct])
        env = vf.SingleTurnEnv(dataset=dataset, rubric=rubric)

        # Mock client
        class MockClient:
            async def chat(self, *args, **kwargs):
                class MockResponse:
                    choices = [
                        type(
                            "Choice",
                            (),
                            {
                                "message": type(
                                    "Message", (), {"content": "answer", "role": "assistant"}
                                )(),
                                "finish_reason": "stop",
                            },
                        )()
                    ]
                    usage = type("Usage", (), {"prompt_tokens": 10, "completion_tokens": 5})()

                return MockResponse()

        mock_client = type("MockClient", (), {"chat": MockClient().chat})()

        collector = EventCollector()

        # Run evaluation with independent scoring and single rollout per example
        results = await env.evaluate(
            client=mock_client,
            model="test-model",
            num_examples=5,
            rollouts_per_example=1,
            independent_scoring=True,
            on_event=collector,
        )

        # Get the StartEvent
        start_event = [e for e in collector.events if e["type"] == "start"][0]

        # Verify the counts are correct
        assert start_event["num_examples"] == 5
        assert start_event["rollouts_per_example"] == 1
        assert start_event["total_rollouts"] == 5


    @pytest.mark.asyncio
    async def test_num_examples_with_grouped_scoring(self):
        """Test that num_examples is correct with grouped scoring (baseline)."""
        # Create a dataset with 2 examples
        dataset = Dataset.from_dict({
            "prompt": ["Q1?", "Q2?"]
        })

        env = vf.SingleTurnEnv(dataset=dataset)

        # Mock client
        class MockClient:
            async def chat(self, *args, **kwargs):
                class MockResponse:
                    choices = [
                        type(
                            "Choice",
                            (),
                            {
                                "message": type(
                                    "Message", (), {"content": "answer", "role": "assistant"}
                                )(),
                                "finish_reason": "stop",
                            },
                        )()
                    ]
                    usage = type("Usage", (), {"prompt_tokens": 10, "completion_tokens": 5})()

                return MockResponse()

        mock_client = type("MockClient", (), {"chat": MockClient().chat})()

        collector = EventCollector()

        # Run evaluation with grouped scoring and multiple rollouts per example
        results = await env.evaluate(
            client=mock_client,
            model="test-model",
            num_examples=2,
            rollouts_per_example=3,
            independent_scoring=False,  # Grouped scoring
            on_event=collector,
        )

        # Get the StartEvent
        start_event = [e for e in collector.events if e["type"] == "start"][0]

        # Verify the counts are correct
        assert start_event["num_examples"] == 2
        assert start_event["rollouts_per_example"] == 3
        assert start_event["total_rollouts"] == 6


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
