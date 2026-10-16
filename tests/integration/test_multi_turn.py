"""One task, many turns: the model completes a shell task (echo-agentic-v1: write a phrase
to a file, checked in the runtime) across the harness x runtime grid."""

import pytest
from grid import RUNTIMES, grid


@pytest.mark.parametrize(
    "harness,harness_runtime", grid(runtimes=RUNTIMES, needs=("shell",)), indirect=True
)
async def test_multi_turn(run, harness, harness_runtime):
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
    assert trace.num_turns >= 1  # ran a command, then finished
    assert trace.reward == 1.0
