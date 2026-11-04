"""An eval-wide shared writable tool (scratchpad) isolates state across concurrent
rollouts, across the runtime x runtime grid: on the evaluator, or on a Prime VM that
every rollout's workspace reaches over the eval's LAN."""

import pytest
from grid import placements


@pytest.mark.parametrize(
    "harness_runtime,tool_runtime",
    placements(tool_runtimes=("subprocess", "docker", "prime")),
    indirect=True,
)
async def test_shared_tool(run, harness_runtime, tool_runtime):
    episodes = await run(
        "scratchpad",
        harness="null",
        runtime=harness_runtime,
        num_tasks=2,
        n=1,
        max_turns=4,
        taskset_overrides={"task": {"tools": tool_runtime}},
    )
    traces = [trace for episode in episodes for trace in episode.traces]
    assert all(trace.num_branches == 1 for trace in traces)
    assert len(traces) == 2
    for trace in traces:
        assert trace.ok, [error.message for error in trace.errors]
        assert trace.num_turns >= 2  # tool call + answer
        assert trace.reward == 1.0
