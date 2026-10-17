"""A task's skill reaches the program: the answer lives only in a SKILL.md the task
installs (skill-codeword-v1), across the harness x runtime grid."""

import pytest
from grid import grid


@pytest.mark.parametrize(
    "harness,harness_runtime", grid(needs=("skills",)), indirect=True
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
    # OpenClaw strips the runtime context it injects before each turn from the
    # replayed history, so the trace forks by design. Recheck on version bumps:
    # https://github.com/openclaw/openclaw/blob/v2026.9.5/src/agents/embedded-agent-runner/run/attempt-llm-boundary.ts#L145
    if harness.id != "openclaw":
        assert trace.num_branches == 1
    assert trace.ok, [error.message for error in trace.errors]
    assert trace.reward == 1.0, trace.last_reply
