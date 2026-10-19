"""A task's skill reaches the program: the answer lives only in a SKILL.md the task
installs (skill-codeword), across the harness x runtime grid."""

import pytest
from grid import grid


@pytest.mark.parametrize(
    "harness,harness_runtime", grid(needs=("skills",)), indirect=True
)
async def test_skills(run, harness, harness_runtime):
    (episode,) = await run(
        "skill-codeword",
        harness=harness,
        runtime=harness_runtime,
        max_turns=8,
        max_tokens=8192,
    )
    (trace,) = episode.traces
    assert trace.num_branches == 1
    assert trace.ok, [error.message for error in trace.errors]
    assert trace.reward == 1.0, trace.last_reply
