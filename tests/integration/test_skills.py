"""A task's skill reaches the program: the answer lives only in a SKILL.md the task
installs (skill-codeword-v1), across the harness x runtime grid."""

import pytest
from grid import grid


@pytest.mark.parametrize(
    "harness,harness_runtime",
    grid(needs=("skills",)),
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
