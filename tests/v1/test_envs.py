"""Smoke-eval the example tasksets in `environments/` through the `eval` CLI.

Each taskset runs with its required harness for one short, capped rollout, so a broken
example taskset fails CI. `compact` is excluded (it's a harness, not a taskset), and so
are the tasksets tests/v1/test_e2e.py already runs end to end (`E2E_COVERED`).
"""

import os
import subprocess
from pathlib import Path

import pytest

pytestmark = pytest.mark.e2e

EVAL_TIMEOUT = 600  # 10 minutes for a capped eval (-n 1 -r 1)

ENVIRONMENTS = Path(__file__).parent.parent.parent / "environments"

# V1 tasksets that aren't part of the default CI install.
SKIP_EVAL = {"nemo_gym_weather"}

# Tasksets tests/v1/test_e2e.py already runs end to end and asserts on: kuhn_poker
# (test_kuhn_poker_self_play), reverse_text (test_replay_round_trip), scratchpad
# (test_shared_tool_isolation, through the env-server pool). Their rewards read the
# trace alone, so the CLI's default harness path adds nothing the remaining smokes
# don't already cover.
E2E_COVERED = {"kuhn_poker", "reverse_text", "scratchpad"}

# Smoke-sized env knobs: the fewest turns/attempts that still score an episode.
SMOKE_FLAGS: dict[str, tuple[str, ...]] = {
    "alphabet_sort": ("--env.taskset.max-turns", "1"),
    "code_golf": ("--env.attempts", "1"),
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
        if d.is_dir()
        and d.name != "compact"
        and d.name not in E2E_COVERED
        and (d / "pyproject.toml").exists()
    )


@pytest.mark.parametrize("taskset", v1_tasksets())
def test_eval(taskset: str):
    """Run one capped rollout of `taskset`; a taskset that bundles a harness uses it by default."""
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
        "uv", "run", "--no-sync", "eval", taskset,
        *model,
        "-n", "1", "-r", "1", *caps, *SMOKE_FLAGS.get(taskset, ()),
        "--sampling.max-tokens", "512", "--no-rich", "--no-push",
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
