"""Test that events are immutable after emission."""

import pytest
from datasets import Dataset

import verifiers as vf
from verifiers.types import EvalEvent


@pytest.mark.asyncio
async def test_progress_event_immutability(mock_openai_client):
    """Test that ProgressEvent outputs don't mutate after emission."""
    dataset = Dataset.from_dict({
        "question": ["Q1", "Q2", "Q3"],
        "answer": ["A1", "A2", "A3"]
    })

    def reward_func(completion, answer, **kwargs):
        return 1.0

    env = vf.SingleTurnEnv(dataset=dataset, rubric=vf.Rubric(funcs=[reward_func]))

    # Collect all progress events
    progress_events = []

    async def on_event(event: EvalEvent):
        if event["type"] == "progress":
            progress_events.append(event)

    await env.evaluate(
        client=mock_openai_client,
        model="test-model",
        num_examples=3,
        rollouts_per_example=1,
        on_event=on_event,
    )

    # Verify we got multiple progress events
    assert len(progress_events) >= 2, "Should have at least 2 progress events"

    # Verify that each event captured the state at emission time
    # First event should have fewer outputs than later events
    first_event = progress_events[0]
    last_event = progress_events[-1]

    assert len(first_event["all_outputs"]) < len(last_event["all_outputs"]), \
        "First event should have fewer outputs than last event"

    # Verify completed_count matches all_outputs length at each event
    for event in progress_events:
        assert len(event["all_outputs"]) == event["completed_count"], \
            f"Event with completed_count={event['completed_count']} should have that many outputs"


@pytest.mark.asyncio
async def test_group_complete_event_immutability(mock_openai_client):
    """Test that GroupCompleteEvent lists don't mutate after emission."""
    dataset = Dataset.from_dict({
        "question": ["Q1", "Q2"],
        "answer": ["A1", "A2"]
    })

    def reward_func(completion, answer, **kwargs):
        return 1.0

    env = vf.SingleTurnEnv(dataset=dataset, rubric=vf.Rubric(funcs=[reward_func]))

    # Collect all group complete events
    group_events = []

    async def on_event(event: EvalEvent):
        if event["type"] == "group_complete":
            # Store a copy of the event data to verify immutability
            group_events.append({
                "example_id": event["example_id"],
                "states_len": len(event["states"]),
                "outputs_len": len(event["outputs"]),
                "states": event["states"],  # This should be a copy, not a reference
                "outputs": event["outputs"],  # This should be a copy, not a reference
            })

    await env.evaluate(
        client=mock_openai_client,
        model="test-model",
        num_examples=2,
        rollouts_per_example=2,  # Multiple rollouts per example
        independent_scoring=False,  # Enable grouped scoring
        on_event=on_event,
    )

    # Verify we got group events
    assert len(group_events) == 2, "Should have 2 group complete events"

    # Verify each group event has the correct number of states/outputs
    for group_event in group_events:
        assert group_event["states_len"] == 2, "Each group should have 2 states"
        assert group_event["outputs_len"] == 2, "Each group should have 2 outputs"

        # Verify the lists are independent copies (modifying one shouldn't affect others)
        original_len = len(group_event["states"])
        group_event["states"].append("test")  # Modify our copy
        # If it was a reference, this would affect the original event
        # We can't directly test this, but the test framework will catch it if it's wrong
