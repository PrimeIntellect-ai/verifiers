"""One task, every wire format: each harness whose config selects a transport runs a
shell task (echo-agentic-v1) over each one. A turn the harness re-sends differently over
that format, beyond what its `replay_view` declares, forks the trace."""

import pytest
from grid import dialect_grid


@pytest.mark.parametrize(
    "harness,harness_runtime", dialect_grid(needs=("shell",)), indirect=True
)
async def test_dialect(run, harness, harness_runtime):
    (episode,) = await run(
        "echo-agentic-v1",
        harness=harness,
        runtime=harness_runtime,
        max_turns=10,
        max_tokens=8192,
    )
    (trace,) = episode.traces
    assert trace.num_branches == 1
    assert trace.ok, [error.message for error in trace.errors]
    assert trace.reward == 1.0
