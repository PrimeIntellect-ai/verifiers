"""The same writable tool isolates concurrent rollouts in either lifetime mode."""

import pytest
from grid import placements


@pytest.mark.parametrize(
    "harness_runtime,tool_runtime",
    placements(),
    indirect=True,
)
@pytest.mark.parametrize("shared", [False, True])
async def test_shared_tool(run, harness_runtime, tool_runtime, shared):
    episodes = await run(
        "scratchpad",
        harness="null",
        runtime=harness_runtime,
        num_tasks=2,
        n=1,
        max_turns=4,
        taskset_overrides={"task": {"tools": {**tool_runtime, "shared": shared}}},
    )
    traces = [trace for episode in episodes for trace in episode.traces]
    assert all(trace.num_branches == 1 for trace in traces)
    assert len(traces) == 2
    for trace in traces:
        assert trace.ok, [error.message for error in trace.errors]
        assert trace.num_turns >= 2  # tool call + answer
        assert trace.reward == 1.0
