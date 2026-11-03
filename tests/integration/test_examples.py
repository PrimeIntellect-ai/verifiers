"""Smoke-run every example taskset in `environments/` in-process.

Each taskset runs with its required harness for capped rollouts, so a broken
example taskset fails CI. Interception exercises all four of its behaviors. `compact` is excluded (it's a harness, not a taskset);
SWE/container tasksets need a docker/prime runtime and are covered by the agentic tests.
"""

from pathlib import Path

import pytest
from conftest import CI_DECISION_JUDGE

EVAL_TIMEOUT = 600  # 10 minutes for a capped run (-n 1 -r 1)

ENVIRONMENTS = Path(__file__).parent.parent.parent / "environments"

SKIP_EVAL = {
    "nemo_gym_weather": "not part of the default CI install",
    # TODO: drop once the CI model's proposer reliably finishes within the caps.
    "proposer_solver": "the CI model's proposer reasons past the token cap",
}

# CI explicitly selects the existing decision judge for grading and interception.
JUDGE_OVERRIDES = {
    "interception": {"task": {"judge": {"id": "decision", **CI_DECISION_JUDGE}}},
    "wiki_search": {
        "task": {
            # Replace the reference judge with the existing decision judge in CI.
            "judges": [
                {
                    "id": "decision",
                    "question_field": "question",
                    "answer_field": "answer",
                    **CI_DECISION_JUDGE,
                }
            ]
        }
    },
}


def v1_tasksets() -> list[str]:
    if not ENVIRONMENTS.is_dir():
        return []
    return sorted(
        d.name
        for d in ENVIRONMENTS.iterdir()
        if d.is_dir() and d.name != "compact" and (d / "pyproject.toml").exists()
    )


@pytest.mark.parametrize("taskset", v1_tasksets())
async def test_eval(taskset: str, run):
    """Smoke-run each example, including all four interception behaviors."""
    if reason := SKIP_EVAL.get(taskset):
        pytest.skip(reason)
    episodes = await run(
        taskset,
        harness=None,
        num_tasks=4 if taskset == "interception" else 1,
        max_tokens=4096,
        max_turns=16,
        rollout_timeout=EVAL_TIMEOUT,
        taskset_overrides=JUDGE_OVERRIDES.get(taskset),
        # Recipe envs run their seats one at a time by default; in parallel, a
        # multi-seat episode (proposer + n solvers) fits the timeout.
        env={"max_concurrent_agents": 8},
    )
    assert episodes and all(episode.ok for episode in episodes), [
        error.message[:500]
        for episode in episodes
        for holder in (episode, *episode.traces)
        for error in holder.errors or []
    ]
    assert all(
        trace.num_branches == 1 for episode in episodes for trace in episode.traces
    )
    if taskset == "interception":
        assert len(episodes) == 4
        judged = [
            trace
            for episode in episodes
            for trace in episode.traces
            if trace.info.get("judge_calls")
        ]
        assert len(judged) == 1
        assert (
            judged[0].info["judge_calls"][0]["request"]["model"]
            == CI_DECISION_JUDGE["model"]
        )
        assert judged[0].response_rewrites
