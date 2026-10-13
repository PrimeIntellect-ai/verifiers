"""A task's skill reaches the program: the answer lives only in a SKILL.md the task
installs (skill-codeword-v1), across the harness x runtime grid."""

import pytest
from grid import grid, skip_harness

# TODO: fix skill discovery in the upstream nano-rlm harness, then drop this skip.
SKILLS = skip_harness(
    grid(needs=("skills",)),
    "rlm",
    "nano-rlm often fails to discover an installed SKILL.md within the turn cap",
)


@pytest.mark.parametrize(
    "harness,harness_runtime",
    SKILLS,
    indirect=True,
)
async def test_skills(run, harness, harness_runtime):
    (episode,) = await run(
        "skill-codeword-v1",
        harness=harness,
        runtime=harness_runtime,
        max_turns=8,
        max_tokens=8192,
    )
    (trace,) = episode.traces
    assert trace.ok, [error.message for error in trace.errors]
    assert trace.reward == 1.0, trace.last_reply
