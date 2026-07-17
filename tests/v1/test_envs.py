"""Smoke-eval every V1 example taskset in `environments/` through the `eval` CLI.

Each `_v1` taskset runs with its required harness for one short, capped rollout, so a broken
example taskset fails CI. `compact` is excluded (it's a harness, not a taskset);
SWE/container tasksets need a docker/prime runtime and are covered by dedicated V1 e2e tests.
"""

import os
import subprocess
from pathlib import Path

import pytest

pytestmark = pytest.mark.e2e

EVAL_TIMEOUT = 600  # 10 minutes for a capped eval (-n 1 -r 1)

ENVIRONMENTS = Path(__file__).parent.parent.parent / "environments"

# v1 tasksets that can't run a plain-CI smoke eval — e.g. they need a docker/prime runtime or
# clone a corpus CI can't read. Empty: the SWE/container and corpus tasksets live in
# research-environments now.
SKIP_EVAL: set[str] = set()


def v1_tasksets() -> list[str]:
    if not ENVIRONMENTS.is_dir():
        return []
    return sorted(
        d.name for d in ENVIRONMENTS.iterdir() if d.is_dir() and d.name.endswith("_v1")
    )


@pytest.mark.parametrize("taskset", v1_tasksets())
def test_eval(taskset: str):
    """Run one capped rollout of `taskset`; a taskset that bundles a harness uses it by default."""
    if taskset in SKIP_EVAL:
        pytest.skip(f"{taskset} can't run a plain-CI smoke eval")
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
        "--taskset.id", taskset,
        *model,
        "-n", "1", "-r", "1", "--max-turns", "4",
        "--sampling.max-tokens", "512", "--rich", "false",
    ]  # fmt: skip
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=EVAL_TIMEOUT)
    except subprocess.TimeoutExpired:
        pytest.fail(f"Timed out after {EVAL_TIMEOUT}s evaluating {taskset}")
    assert proc.returncode == 0, (
        f"eval {taskset} failed: {(proc.stderr or proc.stdout)[-2000:]}"
    )
