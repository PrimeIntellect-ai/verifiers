"""End-to-end eval runs on trivial tasksets — each scores reward 1.0, with no errors.

Every task is one greedy rollout (`temperature=0`, set in `run_v1`) on a single task with
turn/timeout caps. The reward tests fan out across the full **harness x runtime** matrix (the
`harness` x `runtime` fixtures): the built-in harnesses (default, rlm) x the runtimes
(subprocess/docker/prime, modal excluded).

The tools test reads the harness's `SUPPORTS_TASK_TOOLS` capability to decide its expectation:
an harness that can't drive a taskset's MCP tools (rlm) is refused up front, so the pairing
must raise rather than run. The agentic task is the one pinned combination — it needs the bash
tool, which only the default harness exposes — so it varies runtime but not harness.
"""

import pytest


@pytest.mark.e2e
async def test_single_turn(run_v1, harness, runtime, tmp_path):
    """Single-turn (echo a short phrase back)."""
    (trace,) = await run_v1(
        "echo-v1", harness=harness, runtime=runtime, output_dir=tmp_path, max_turns=2
    )
    assert trace.errors == []
    assert not trace.is_truncated
    assert trace.num_turns == 1
    assert trace.reward == 1.0


@pytest.mark.e2e
async def test_multi_turn(run_v1, harness, runtime, tmp_path):
    """Multi-turn (sort one name per turn, two turns)."""
    (trace,) = await run_v1(
        "alphabet-sort-v1",
        harness=harness,
        runtime=runtime,
        output_dir=tmp_path,
        max_turns=4,
        taskset_overrides={
            "min_turns": 2,
            "max_turns": 2,
            "min_names_per_turn": 1,
            "max_names_per_turn": 1,
            "similarity_power": 1,  # linear similarity (no power scaling), so a near-perfect sort isn't sharply penalized
        },
    )
    assert trace.errors == []
    assert not trace.is_truncated
    assert trace.num_turns >= 2  # genuinely multi-turn
    assert trace.reward == 1.0


@pytest.mark.e2e
async def test_multi_turn_with_tools(
    run_v1, harness, runtime, supports_task_tools, tmp_path
):
    """Multi-turn with a colocated tool. An harness whose `SUPPORTS_TASK_TOOLS` is False can't
    drive the task's tools, so the pairing is refused at build time instead of running."""
    if not supports_task_tools(harness):
        with pytest.raises(ValueError, match="task tools"):
            await run_v1(
                "glossary-v1", harness=harness, runtime=runtime, output_dir=tmp_path
            )
        return
    (trace,) = await run_v1(
        "glossary-v1",
        harness=harness,
        runtime=runtime,
        output_dir=tmp_path,
        max_turns=6,
    )
    assert trace.errors == []
    assert not trace.is_truncated
    assert trace.num_turns >= 2  # tool call + answer
    assert trace.reward == 1.0


@pytest.mark.e2e
async def test_agentic(run_v1, runtime, tmp_path):
    """Agentic: write a phrase to a file with the bash tool, checked in the runtime. Only the
    default harness exposes the bash tool, so this varies runtime but pins the harness."""
    (trace,) = await run_v1(
        "agentic-echo-v1",
        harness="default",
        enable_bash=True,
        runtime=runtime,
        output_dir=tmp_path,
        max_turns=10,
    )
    assert trace.errors == []
    assert trace.num_turns >= 1  # ran a command, then finished
    assert trace.reward == 1.0
