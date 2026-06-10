"""End-to-end eval runs on trivial tasksets — each scores reward 1.0, with no errors.

Every task is configured for the least work that still exercises the path: one task, one
greedy rollout (`temperature=0`, set in the `run_v1` helper), with turn/timeout caps. All
four tasks (single-turn, multi-turn, multi-turn + tools, agentic) fan out across the built-in
runtimes (subprocess/docker/prime, modal excluded) via the `runtime` fixture.
"""

import pytest

pytestmark = pytest.mark.e2e


async def test_single_turn(run_v1, runtime, tmp_path):
    """Trivial single-turn task (echo a short phrase back)."""
    (trace,) = await run_v1(
        "echo-v1",
        runtime=runtime,
        output_dir=tmp_path,
        max_turns=2,
    )
    assert trace.errors == []
    assert not trace.is_truncated
    assert trace.num_turns == 1
    assert trace.reward == 1.0


async def test_multi_turn(run_v1, runtime, tmp_path):
    """Trivial multi-turn task (sort one name per turn, two turns)."""
    (trace,) = await run_v1(
        "alphabet-sort-v1",
        runtime=runtime,
        output_dir=tmp_path,
        max_turns=4,
        taskset_overrides={
            "min_turns": 2,
            "max_turns": 2,
            "min_names_per_turn": 1,
            "max_names_per_turn": 1,
        },
    )
    assert trace.errors == []
    assert not trace.is_truncated
    assert trace.num_turns >= 2  # genuinely multi-turn
    assert trace.reward == 1.0


async def test_multi_turn_with_tools(run_v1, runtime, tmp_path):
    """Trivial multi-turn task with a colocated tool (look a fact up, then answer)."""
    (trace,) = await run_v1(
        "glossary-v1",
        runtime=runtime,
        output_dir=tmp_path,
        max_turns=6,
    )
    assert trace.errors == []
    assert not trace.is_truncated
    assert trace.num_turns >= 2  # tool call + answer
    assert trace.reward == 1.0


async def test_agentic(run_v1, runtime, tmp_path):
    """Trivial agentic task: write a phrase to a file with the bash tool, checked in the runtime."""
    (trace,) = await run_v1(
        "agentic-echo-v1",
        runtime=runtime,
        output_dir=tmp_path,
        harness="default",
        enable_bash=True,
        max_turns=10,
    )
    assert trace.errors == []
    assert trace.num_turns >= 1  # ran a command, then finished
    assert trace.reward == 1.0
