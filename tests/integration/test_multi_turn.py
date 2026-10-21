"""One task, many turns: the model completes a shell task (echo-agentic: write a phrase
to a file, checked in the runtime) across the harness x runtime grid, and on prime over
every other `dialect` a harness config can select. A turn the harness re-sends
differently, beyond what its `replay_view` declares, forks the trace."""

import pytest
from grid import RUNTIMES, assert_one_branch, dialect_grid, grid, mark

MULTI_TURN = [
    *grid(runtimes=RUNTIMES, needs=("shell",)),
    *dialect_grid(needs=("shell",)),
    pytest.param(
        "bash",
        {"type": "prime", "allow": []},
        marks=[mark("bash"), mark("prime")],
        id="bash-in-prime-framework-only",
    ),
]


@pytest.mark.parametrize("harness,harness_runtime", MULTI_TURN, indirect=True)
async def test_multi_turn(run, harness, harness_runtime):
    (episode,) = await run(
        "echo-agentic",
        harness=harness,
        runtime=harness_runtime,
        max_turns=10,
        max_tokens=8192,
    )
    (trace,) = episode.traces
    assert_one_branch(trace, harness, forks=("openclaw",))
    assert trace.ok, [error.message for error in trace.errors]
    assert trace.num_turns >= 1  # ran a command, then finished
    assert trace.reward == 1.0
