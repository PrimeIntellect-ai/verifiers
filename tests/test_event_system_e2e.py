#!/usr/bin/env python3
"""End-to-end test scenarios for the unified event system.

This script runs realistic evaluation scenarios and verifies that events
are emitted correctly at each stage.
"""

import asyncio
from collections import defaultdict
from pathlib import Path

import verifiers as vf
from verifiers.types import EvalEvent


class EventCollector:
    """Collects and validates events during evaluation."""

    def __init__(self, name: str):
        self.name = name
        self.events: list[EvalEvent] = []
        self.event_counts = defaultdict(int)

    async def __call__(self, event: EvalEvent) -> None:
        """Event handler that collects all events."""
        self.events.append(event)
        self.event_counts[event["type"]] += 1
        print(f"[{self.name}] {event['type']:15} | ", end="")

        match event["type"]:
            case "start":
                print(
                    f"total={event['total_rollouts']}, "
                    f"examples={event['num_examples']}, "
                    f"rollouts_per={event['rollouts_per_example']}"
                )
            case "progress":
                print(
                    f"completed={event['completed_count']}/{event['total_count']}, "
                    f"new={len(event['new_outputs'])}"
                )
            case "group_complete":
                print(
                    f"example_id={event['example_id']}, "
                    f"states={len(event['states'])}, "
                    f"outputs={len(event['outputs'])}"
                )
            case "log":
                print(f"[{event['level']}] {event['message']}")
            case "save":
                save_type = "intermediate" if event["is_intermediate"] else "final"
                print(f"{save_type}, outputs={event['output_count']}")
            case "complete":
                print(
                    f"total={event['total_outputs']}, "
                    f"avg_reward={event['avg_reward']:.3f}, "
                    f"time={event['total_time_ms']:.1f}ms"
                )
            case "log_stream":
                print(f"stream={event['stream_id']}, {len(event['data'])} bytes")

    def validate(self):
        """Validate that expected events were received."""
        print(f"\n{'='*60}")
        print(f"Validation Results for: {self.name}")
        print(f"{'='*60}")

        # Check event counts
        print(f"Events collected: {len(self.events)}")
        for event_type, count in sorted(self.event_counts.items()):
            print(f"  {event_type:15} : {count}")

        # Validate event sequence
        assert self.events, "No events collected!"

        # StartEvent should be first
        assert (
            self.events[0]["type"] == "start"
        ), f"First event should be 'start', got {self.events[0]['type']}"

        # CompleteEvent should be last
        assert (
            self.events[-1]["type"] == "complete"
        ), f"Last event should be 'complete', got {self.events[-1]['type']}"

        # Should have at least one ProgressEvent
        assert (
            self.event_counts["progress"] > 0
        ), "Should have at least one progress event"

        # Validate StartEvent data
        start_event = self.events[0]
        assert start_event["total_rollouts"] > 0, "total_rollouts should be > 0"
        assert start_event["num_examples"] > 0, "num_examples should be > 0"
        assert (
            start_event["rollouts_per_example"] > 0
        ), "rollouts_per_example should be > 0"

        # Validate CompleteEvent data
        complete_event = self.events[-1]
        assert complete_event["total_outputs"] > 0, "total_outputs should be > 0"
        assert "avg_reward" in complete_event, "CompleteEvent should have avg_reward"
        assert "total_time_ms" in complete_event, "CompleteEvent should have total_time_ms"

        print("\n✅ All validations passed!")


async def scenario_1_simple_eval():
    """Scenario 1: Simple single-environment evaluation (independent scoring)."""
    print("\n" + "=" * 60)
    print("SCENARIO 1: Simple Independent Scoring Evaluation")
    print("=" * 60)

    collector = EventCollector("scenario_1")

    # Create a simple environment with a small dataset
    from datasets import Dataset

    dataset = Dataset.from_dict(
        {
            "prompt": [
                "What is 2+2?",
                "What is 3+3?",
                "What is 5+5?",
            ]
        }
    )

    # Simple reward function that always returns 1.0
    def always_correct(prompt, completion, **kwargs):
        return 1.0

    # Create rubric with the reward function
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
                                "Message", (), {"content": "4", "role": "assistant"}
                            )(),
                            "finish_reason": "stop",
                        },
                    )()
                ]
                usage = type("Usage", (), {"prompt_tokens": 10, "completion_tokens": 5})()

            return MockResponse()

    mock_client = type("MockClient", (), {"chat": MockClient().chat})()

    # Run evaluation
    results = await env.evaluate(
        client=mock_client,
        model="test-model",
        num_examples=3,
        rollouts_per_example=1,
        independent_scoring=True,
        on_event=collector,
    )

    collector.validate()

    # Additional checks
    assert results["metadata"]["num_examples"] == 3
    assert len(results["outputs"]) == 3
    print(f"\n✅ Scenario 1 passed: {len(results['outputs'])} outputs generated")


async def scenario_2_grouped_scoring():
    """Scenario 2: Grouped scoring (multiple rollouts per example)."""
    print("\n" + "=" * 60)
    print("SCENARIO 2: Grouped Scoring with Multiple Rollouts")
    print("=" * 60)

    collector = EventCollector("scenario_2")

    from datasets import Dataset

    dataset = Dataset.from_dict({"prompt": ["What is 2+2?", "What is 3+3?"]})

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

    # Run evaluation with multiple rollouts per example
    results = await env.evaluate(
        client=mock_client,
        model="test-model",
        num_examples=2,
        rollouts_per_example=3,  # 3 rollouts per example
        independent_scoring=False,  # Group scoring
        on_event=collector,
    )

    collector.validate()

    # Check GroupCompleteEvent was emitted
    assert (
        collector.event_counts["group_complete"] > 0
    ), "Should have GroupCompleteEvent for grouped scoring"
    assert (
        collector.event_counts["group_complete"] == 2
    ), "Should have 2 GroupCompleteEvents (one per example)"

    # Validate GroupCompleteEvent has State objects
    group_events = [e for e in collector.events if e["type"] == "group_complete"]
    for ge in group_events:
        assert len(ge["states"]) == 3, "Each group should have 3 states"
        assert len(ge["outputs"]) == 3, "Each group should have 3 outputs"
        assert "example_id" in ge, "GroupCompleteEvent should have example_id"

    print(
        f"\n✅ Scenario 2 passed: {collector.event_counts['group_complete']} groups processed"
    )


async def scenario_3_intermediate_saves():
    """Scenario 3: Intermediate saves (save_every)."""
    print("\n" + "=" * 60)
    print("SCENARIO 3: Intermediate Saves")
    print("=" * 60)

    collector = EventCollector("scenario_3")

    from datasets import Dataset

    dataset = Dataset.from_dict(
        {
            "prompt": [
                f"Question {i}?" for i in range(10)
            ]  # 10 examples to trigger saves
        }
    )

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

    import tempfile

    with tempfile.TemporaryDirectory() as tmpdir:
        results = await env.evaluate(
            client=mock_client,
            model="test-model",
            num_examples=10,
            rollouts_per_example=1,
            save_results=True,
            save_every=3,  # Save every 3 completions
            results_path=Path(tmpdir) / "results.jsonl",
            on_event=collector,
        )

    collector.validate()

    # Check SaveEvent emissions
    save_events = [e for e in collector.events if e["type"] == "save"]
    intermediate_saves = [e for e in save_events if e["is_intermediate"]]
    final_saves = [e for e in save_events if not e["is_intermediate"]]

    assert len(intermediate_saves) > 0, "Should have intermediate saves"
    assert len(final_saves) == 1, "Should have exactly one final save"

    print(
        f"\n✅ Scenario 3 passed: {len(intermediate_saves)} intermediate saves, "
        f"1 final save"
    )


async def scenario_4_progress_tracking():
    """Scenario 4: Verify progress tracking with rolling metrics."""
    print("\n" + "=" * 60)
    print("SCENARIO 4: Progress Tracking with Metrics")
    print("=" * 60)

    collector = EventCollector("scenario_4")

    from datasets import Dataset

    dataset = Dataset.from_dict({"prompt": [f"Q{i}?" for i in range(5)]})

    env = vf.SingleTurnEnv(dataset=dataset)

    # Mock client with varying rewards
    call_count = 0

    class MockClient:
        async def chat(self, *args, **kwargs):
            nonlocal call_count
            call_count += 1

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

    results = await env.evaluate(
        client=mock_client,
        model="test-model",
        num_examples=5,
        rollouts_per_example=1,
        on_event=collector,
    )

    collector.validate()

    # Validate ProgressEvent tracking
    progress_events = [e for e in collector.events if e["type"] == "progress"]
    assert len(progress_events) == 5, "Should have 5 progress events"

    # Verify completed_count increases
    for i, pe in enumerate(progress_events):
        assert pe["completed_count"] == i + 1, f"Event {i} should have completed={i+1}"
        assert pe["total_count"] == 5, "total_count should always be 5"

    print(f"\n✅ Scenario 4 passed: {len(progress_events)} progress events tracked")


async def main():
    """Run all test scenarios."""
    print("\n" + "=" * 60)
    print("UNIFIED EVENT SYSTEM - END-TO-END TESTS")
    print("=" * 60)

    scenarios = [
        scenario_1_simple_eval,
        scenario_2_grouped_scoring,
        scenario_3_intermediate_saves,
        scenario_4_progress_tracking,
    ]

    results = []
    for scenario in scenarios:
        try:
            await scenario()
            results.append((scenario.__name__, "✅ PASSED"))
        except Exception as e:
            results.append((scenario.__name__, f"❌ FAILED: {e}"))
            import traceback

            traceback.print_exc()

    # Final summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    for name, status in results:
        print(f"{name:40} : {status}")

    passed = sum(1 for _, status in results if "PASSED" in status)
    total = len(results)
    print(f"\n{passed}/{total} scenarios passed")

    if passed == total:
        print("\n🎉 All scenarios passed! Event system is working correctly.")
    else:
        print("\n⚠️  Some scenarios failed. Check output above for details.")


if __name__ == "__main__":
    asyncio.run(main())
