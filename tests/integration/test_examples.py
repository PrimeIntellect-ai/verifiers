"""Smoke-eval every example taskset in `environments/` through the `eval` CLI.

Each taskset runs with its required harness for one short, capped rollout, so a broken
example taskset fails CI. `compact` is excluded (it's a harness, not a taskset);
SWE/container tasksets need a docker/prime runtime and are covered by the agentic tests.
"""

import json
import os
import subprocess
from pathlib import Path

import pytest
from conftest import CI_MODEL, CI_REASONING_EFFORT, CI_VISION_MODEL
from grid import agent_seats

EVAL_TIMEOUT = 600  # 10 minutes for a capped eval (-n 1 -r 1)

ENVIRONMENTS = Path(__file__).parent.parent.parent / "environments"

# V1 tasksets that aren't part of the default CI install.
SKIP_EVAL = {"nemo_gym_weather"}

# Tasksets that send images, which the text-only CI model rejects.
VISION_TASKSETS = {"color_codeword", "grayscale_interception"}

# Point each bundled judge at the CI model.
JUDGE_OVERRIDES = {
    "interception": ["--env.taskset.task.judge.model", CI_MODEL],
    "wiki_search": [
        "--env.taskset.task.judges",
        # The list replaces the default entry, so restate its `question_field`.
        json.dumps(
            [{"id": "wiki-search", "model": CI_MODEL, "question_field": "question"}]
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
    """Run one capped rollout of `taskset` and require every episode to finish ok; a
    taskset that bundles a harness uses it by default."""
    if taskset in SKIP_EVAL:
        pytest.skip(f"{taskset} can't run a plain-CI smoke eval")
    if os.getenv("PRIME_API_KEY"):
        model = [
            "-m", CI_VISION_MODEL if taskset in VISION_TASKSETS else CI_MODEL,
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
    cmd = [
        "uv", "run", "--no-sync", "vf-eval", taskset,
        *model,
        "-n", "1", "-r", "1", *caps,
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
        json.loads(line)
        for line in (tmp_path / "run" / "traces.jsonl").read_text().splitlines()
    ]
    assert episodes and all(episode["ok"] for episode in episodes), [
        error["message"][:500]
        for episode in episodes
        for holder in (episode, *episode["traces"])
        for error in holder.get("errors") or []
    ]
