"""One task, many turns: the model completes a shell task (echo-agentic-v1: write a phrase
to a file, checked in the runtime) across the harness x runtime grid."""

import pytest
from grid import grid, mark

MULTI_TURN = [
    *grid(needs=("shell",)),
    pytest.param(
        "bash",
        {"type": "modal", "allow": []},
        marks=[mark("bash"), mark("modal")],
        id="bash-in-modal-framework-only",
    ),
]


@pytest.mark.parametrize("harness,harness_runtime", MULTI_TURN, indirect=True)
async def test_multi_turn(run_v1, harness, harness_runtime, tmp_path):
    (trace,) = await run_v1(
        "echo-agentic-v1",
        harness=harness,
        runtime=harness_runtime,
        output_dir=tmp_path,
        max_turns=10,
        max_tokens=8192,
    )
    assert trace.ok, [error.message for error in trace.errors]
    assert trace.num_turns >= 1  # ran a command, then finished
    assert trace.reward == 1.0
