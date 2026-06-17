"""Smoke-eval every v1 example taskset in `environments/` through the `eval` CLI.

The v0 counterpart (`tests/test_envs.py`) covers v0 envs (`vf.load_environment` + `vf-eval`);
here we run each `_v1` taskset with the default harness for one short, capped rollout and
require it to succeed — so a broken example taskset fails CI. `compact` is excluded (it's a
harness, not a taskset); SWE/container tasksets need a docker/prime runtime and are covered by
the dedicated v1 e2e tests instead.
"""

import os
import subprocess
from pathlib import Path

import pytest

pytestmark = pytest.mark.e2e

EVAL_TIMEOUT = 600  # 10 minutes for a capped eval (-n 1 -r 2)

ENVIRONMENTS = Path(__file__).parent.parent.parent / "environments"

# v1 tasksets that need a docker/prime runtime + image-backed sandboxes, so they can't run a
# smoke eval in plain CI — covered by the dedicated v1 e2e tests.
NEEDS_CONTAINER = {
    "openenv_echo_v1",
    "openenv_textarena_v1",
    "r2e_gym_v1",
    "scaleswe_v1",
    "swelego_v1",
    "swebench_verified_v1",
    "terminal_bench_2_v1",
}


def v1_tasksets() -> list[str]:
    if not ENVIRONMENTS.is_dir():
        return []
    return sorted(
        d.name for d in ENVIRONMENTS.iterdir() if d.is_dir() and d.name.endswith("_v1")
    )


@pytest.mark.parametrize("taskset", v1_tasksets())
def test_eval(taskset: str):
    """Run one capped rollout of `taskset` with the default harness and require success."""
    if taskset in NEEDS_CONTAINER:
        pytest.skip(f"{taskset} needs a docker/prime runtime (covered by v1 e2e tests)")
    if os.getenv("PRIME_API_KEY"):
        model = [
            "-m", "openai/gpt-4.1-mini",
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

    cmd = [
        "uv", "run", "--no-sync", "eval",
        "--taskset.id", taskset, "--harness.id", "default",
        *model,
        # -r 2: a taskset with @group_reward(s) needs >=2 rollouts to compare.
        "-n", "1", "-r", "2", "--max-turns", "4",
        "--sampling.max-tokens", "512", "--rich", "false",
    ]  # fmt: skip
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=EVAL_TIMEOUT)
    except subprocess.TimeoutExpired:
        pytest.fail(f"Timed out after {EVAL_TIMEOUT}s evaluating {taskset}")
    assert proc.returncode == 0, (
        f"eval {taskset} failed: {(proc.stderr or proc.stdout)[-2000:]}"
    )
