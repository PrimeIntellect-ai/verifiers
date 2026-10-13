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

EVAL_TIMEOUT = 600  # 10 minutes for a capped eval (-n 1 -r 1)

ENVIRONMENTS = Path(__file__).parent.parent.parent / "environments"

# V1 tasksets that aren't part of the default CI install.
SKIP_EVAL = {"nemo_gym_weather"}

# Tasksets a capped single rollout can legitimately score 0 on, and why.
NO_REWARD_FLOOR = {
    "proposer_solver": "learnability is 4 * rate * (1 - rate): 0 when the solver always succeeds",
    "web_search_interception": "needs a native web search harness (codex); the smoke runs bash",
    "wordle": "the game outcome is the reward; a game lost within the 4-turn cap scores 0",
}

# Per-run caps are seat fields; recipe envs name their own seats.
SEATS: dict[str, tuple[str, ...]] = {
    "code_golf": ("golfer",),
    "kuhn_poker": ("player0", "player1"),
    "openenv_wordle": ("player",),
    "proposer_solver": ("proposer", "solver"),
    "wordle": ("player",),
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
    """Run one capped rollout of `taskset` and require a positive reward; a taskset that
    bundles a harness uses it by default."""
    if taskset in SKIP_EVAL:
        pytest.skip(f"{taskset} can't run a plain-CI smoke eval")
    if os.getenv("PRIME_API_KEY"):
        model = [
            "-m", "openai/gpt-5.6-luna",
            "--client.base-url", "https://api.pinference.ai/api/v1",
            "--client.api-key-var", "PRIME_API_KEY",
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
        for seat in SEATS.get(taskset, ("agent",))
        for flag in (f"--env.{seat}.max-turns", "4")
    ]
    cmd = [
        "uv", "run", "--no-sync", "vf-eval", taskset,
        *model,
        "-n", "1", "-r", "1", *caps,
        "--sampling.max-tokens", "4096", "--no-rich", "--no-push",
        "--output-dir", str(tmp_path), "--run.dir", "run",
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
    traces = [trace for episode in episodes for trace in episode["traces"]]
    rewards = [
        sum(
            (r["score"] or 0.0) * r.get("weight", 1.0)
            for r in trace.get("rewards", {}).values()
            if r is not None
        )
        for trace in traces
    ]
    if taskset in NO_REWARD_FLOOR:
        return
    assert traces and max(rewards) > 0, (
        f"eval {taskset} scored {rewards}: "
        f"{[t.get('errors') for t in traces if t.get('errors')]}"
    )
