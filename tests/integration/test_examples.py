"""Smoke-eval every example taskset in `environments/` through the `eval` CLI.

Each taskset runs with its required harness for capped rollouts, so a broken
example taskset fails CI. Interception exercises all four of its behaviors. `compact` is excluded (it's a harness, not a taskset);
SWE/container tasksets need a docker/prime runtime and are covered by the agentic tests.
"""

import json
import os
import subprocess
from pathlib import Path

import pytest
from conftest import CI_DECISION_JUDGE, CI_MODEL, CI_REASONING_EFFORT
from grid import agent_seats

import verifiers.v1 as vf

EVAL_TIMEOUT = 600  # 10 minutes for a capped eval (-n 1 -r 1)

ENVIRONMENTS = Path(__file__).parent.parent.parent / "environments"

SKIP_EVAL = {
    "nemo_gym_weather": "not part of the default CI install",
    # TODO: drop once the CI model's proposer reliably finishes within the caps.
    "proposer_solver": "the CI model's proposer reasons past the token cap",
}

# CI explicitly selects the existing decision judge for grading and interception.
JUDGE_OVERRIDES = {
    "interception": [
        "--env.taskset.task.judge.id",
        "decision",
        "--env.taskset.task.judge.model",
        CI_DECISION_JUDGE["model"],
    ],
    "wiki_search": [
        "--env.taskset.task.judges",
        # Replace the reference judge with the existing decision judge in CI.
        json.dumps(
            [
                {
                    "id": "decision",
                    "question_field": "question",
                    "answer_field": "answer",
                    **CI_DECISION_JUDGE,
                }
            ]
        ),
    ],
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
def test_eval(taskset: str, tmp_path: Path):
    """Smoke-eval each example, including all four interception behaviors."""
    if reason := SKIP_EVAL.get(taskset):
        pytest.skip(reason)
    if os.getenv("PRIME_API_KEY"):
        model = [
            "-m", CI_MODEL,
            "--sampling.reasoning-effort", CI_REASONING_EFFORT,
            "--client.base-url", "https://api.pinference.ai/api/v1",
            "--client.api-key-var", "PRIME_API_KEY",
            *JUDGE_OVERRIDES.get(taskset, []),
        ]  # fmt: skip
    elif os.getenv("OPENAI_API_KEY"):
        model = [
            "-m", "gpt-4.1-mini",
            "--client.base-url", "https://api.openai.com/v1",
            "--client.api-key-var", "OPENAI_API_KEY",
        ]  # fmt: skip
    else:
        pytest.skip("no model API key configured")

    caps = [
        flag
        for seat in agent_seats(taskset)
        for flag in (f"--env.{seat}.max-turns", "16")
    ]
    num_tasks = "4" if taskset == "interception" else "1"
    cmd = [
        "uv", "run", "--no-sync", "vf-eval", taskset,
        *model,
        "-n", num_tasks, "-r", "1", *caps,
        "--sampling.max-tokens", "4096", "--no-rich", "--no-push",
        "--output-dir", str(tmp_path), "--run.dir", "run",
        # Recipe envs run their seats one at a time by default; in parallel, a
        # multi-seat episode (proposer + n solvers) fits the timeout.
        "--env.max-concurrent-agents", "8",
    ]  # fmt: skip
    try:
        proc = subprocess.run(
            cmd, capture_output=True, text=True, timeout=EVAL_TIMEOUT, check=False
        )
    except subprocess.TimeoutExpired:
        pytest.fail(f"Timed out after {EVAL_TIMEOUT}s evaluating {taskset}")
    assert proc.returncode == 0, (
        f"eval {taskset} failed: {(proc.stderr or proc.stdout)[-2000:]}"
    )
    episodes = [
        vf.WireEpisode.model_validate_json(line)
        for line in (tmp_path / "run" / "traces.jsonl").read_text().splitlines()
    ]
    assert episodes and all(episode.ok for episode in episodes), [
        error.message[:500]
        for episode in episodes
        for holder in (episode, *episode.traces)
        for error in holder.errors or []
    ]
    assert all(
        trace.num_branches == 1 for episode in episodes for trace in episode.traces
    )
    if taskset == "interception" and os.getenv("PRIME_API_KEY"):
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
