"""A task's skill reaches the program: the answer lives only in a SKILL.md the task
installs (skill-codeword-v1), across the harness x runtime grid."""

import pytest
from grid import grid, harness_id


def skip_rlm(cells: list) -> list:
    # TODO: fix skill discovery in the upstream nano-rlm harness, then drop this skip.
    # The model often fails to find a SKILL.md installed for rlm within the turn cap.
    return [
        pytest.param(
            *cell.values,
            marks=[
                *cell.marks,
                pytest.mark.skip(reason="nano-rlm skill discovery is flaky"),
            ],
            id=cell.id,
        )
        if harness_id(cell.values[0]) == "rlm"
        else cell
        for cell in cells
    ]


@pytest.mark.parametrize(
    "harness,harness_runtime",
    skip_rlm(grid(needs=("skills",))),
    indirect=True,
)
async def test_skills(run_v1, harness, harness_runtime, tmp_path):
    (trace,) = await run_v1(
        "skill-codeword-v1",
        harness=harness,
        runtime=harness_runtime,
        output_dir=tmp_path,
        max_turns=8,
        max_tokens=8192,
    )
    assert trace.ok, [error.message for error in trace.errors]
    assert trace.reward == 1.0, trace.last_reply
