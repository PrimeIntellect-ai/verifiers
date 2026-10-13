"""A tool that writes the rollout's state: a `@vf.tool` increments the typed `trace.state`
each call (counter-tool-v1) and the reward reads it back, across the tool server's
placements on the runtime x runtime grid."""

import pytest
from grid import placements


@pytest.mark.parametrize("harness_runtime,tool_runtime", placements(), indirect=True)
async def test_tool_state(run_v1, harness_runtime, tool_runtime, tmp_path):
    """The shared-state round-trip: a `@vf.tool` increments the typed `trace.state` each
    call (synced over the interception server) and the `@reward` reads it back — reward
    1.0 proves tool writes reach the host's `trace.state`, colocated and own-runtime (a
    SHARED server's per-rollout state channel is covered by `test_shared_tool`)."""
    (trace,) = await run_v1(
        "counter-tool-v1",
        harness="null",
        runtime=harness_runtime,
        output_dir=tmp_path,
        max_turns=8,
        taskset_overrides={"task": {"tools": tool_runtime}},
    )
    assert trace.ok, [error.message for error in trace.errors]
    assert trace.num_turns >= 2  # at least two tool calls accumulated
    assert trace.reward == 1.0
