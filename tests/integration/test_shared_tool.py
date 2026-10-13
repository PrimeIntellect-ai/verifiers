"""An eval-wide shared writable tool (scratchpad) isolates state across concurrent
rollouts, through the env-server pool, across the runtime x runtime grid. Shared servers
always run in their own runtime (colocation is per-rollout, shared is eval-level)."""

import pytest
from grid import HARNESS_RUNTIMES, placements


@pytest.mark.parametrize(
    "harness_runtime,tool_runtime",
    placements(tool_runtimes=tuple(HARNESS_RUNTIMES)),
    indirect=True,
)
async def test_shared_tool(run_v1_server, harness_runtime, tool_runtime, tmp_path):
    traces = await run_v1_server(
        "scratchpad",
        harness="null",
        runtime=harness_runtime,
        output_dir=tmp_path,
        num_tasks=2,
        n=1,
        max_turns=4,
        taskset_overrides={"tools": tool_runtime},
    )
    assert len(traces) == 2
    for trace in traces:
        assert trace.ok, [error.message for error in trace.errors]
        assert trace.num_turns >= 2  # tool call + answer
        assert trace.reward == 1.0
