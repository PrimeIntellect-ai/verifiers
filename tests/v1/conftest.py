"""Shared fixtures + helpers for the v1 end-to-end eval tests.

These tests run REAL eval runs (a live model endpoint, real runtimes) with the smallest
settings that still exercise the path, then assert on the resulting `Trace`(s) — they are
not unit tests of individual components. They need a model API key (`PRIME_API_KEY`);
without one the `e2e`-marked tests skip (config parsing still runs).

`run_v1` / `run_v0` mirror the eval CLI's two paths (`run_eval` for a v1 taskset,
`run_legacy_eval` for a v0 env). The `runtime` / `container_runtime` fixtures fan a test
out across the built-in runtimes (modal excluded for now); docker/prime are marked so they
deselect cleanly (`-m "not slow"`).
"""

import os
import subprocess
from importlib import invalidate_caches
from importlib.util import find_spec
from pathlib import Path

import pytest

from verifiers.v1.configs.eval import EvalConfig
from verifiers.v1.env import Environment
from verifiers.v1.cli.runner import run_eval
from verifiers.v1.trace import Trace

# The fixture tasksets (echo-v1, agentic-echo-v1) live in tests/v1/fixtures, added to the
# path via `pythonpath` in pyproject so the eval loader resolves them by id.
REPO_ROOT = Path(__file__).resolve().parents[2]

# Built-in runtimes, modal excluded. docker needs the daemon; prime provisions real
# sandboxes + tunnels (network + PRIME credentials), so both are marked to deselect easily.
RUNTIMES = [
    "subprocess",
    pytest.param("docker", marks=pytest.mark.slow),
    pytest.param("prime", marks=[pytest.mark.slow, pytest.mark.prime]),
]


@pytest.fixture(params=RUNTIMES)
def runtime(request) -> str:
    return request.param


def pytest_collection_modifyitems(config, items) -> None:
    """Skip the live-model tests (marked `e2e`) when no model endpoint is configured, so the
    rest of the suite (e.g. config parsing) still runs in a keyless environment."""
    if os.environ.get("PRIME_API_KEY") or os.environ.get("OPENAI_API_KEY"):
        return
    skip = pytest.mark.skip(reason="needs a model API key (PRIME_API_KEY / OPENAI_API_KEY)")
    for item in items:
        if "e2e" in item.keywords:
            item.add_marker(skip)


async def _run_v1(
    taskset: str,
    *,
    runtime: str,
    output_dir: Path,
    harness: str = "default",
    n: int = 1,
    max_tokens: int = 2048,
    max_turns: int | None = 4,
    rollout_timeout: float = 180,
    enable_bash: bool = False,
    taskset_overrides: dict | None = None,
) -> list[Trace]:
    """Run a v1 taskset end-to-end (the eval CLI's native path) and return its traces.

    `temperature=0` (greedy) makes the run reproducible; `max_tokens` is generous headroom,
    not a target — these trivial tasks finish in a few hundred tokens, so capping tighter only
    risks truncating the reasoning before the answer (which tanks the reward)."""
    harness_config: dict = {"id": harness, "runtime": {"type": runtime}}
    if enable_bash:
        harness_config["enable_bash"] = True
    config = EvalConfig(
        taskset={"id": taskset, **(taskset_overrides or {})},
        harness=harness_config,
        num_tasks=1,
        num_rollouts=n,
        max_turns=max_turns,
        max_output_tokens=max_tokens,
        sampling={"max_tokens": max_tokens, "temperature": 0},
        timeout={"rollout": rollout_timeout, "scoring": 60},
        rich=False,
        output_dir=output_dir,
    )
    return await run_eval(Environment(config), config)


async def _run_v0(
    env_id: str,
    *,
    output_dir: Path,
    n: int = 1,
    max_tokens: int = 2048,
    args: dict | None = None,
) -> list[Trace]:
    """Run a legacy v0 env through the v1 bridge (the eval CLI's `--id` path)."""
    from verifiers.v1.legacy import run_legacy_eval

    config = EvalConfig(
        id=env_id,
        args=args or {},
        num_tasks=1,
        num_rollouts=n,
        sampling={"max_tokens": max_tokens, "temperature": 0},
        rich=False,
        output_dir=output_dir,
    )
    return await run_legacy_eval(config)


@pytest.fixture
def run_v1():
    return _run_v1


@pytest.fixture
def run_v0():
    return _run_v0


@pytest.fixture
def ensure_v0():
    """Make a local v0 env importable. The legacy bridge only auto-installs *hub* ids, so an
    in-repo env (`environments/<name>`) must be installed first; do it without deps (verifiers
    + datasets are already present) and skip the test if that fails."""

    def _ensure(module: str, rel_path: str) -> None:
        if find_spec(module) is not None:
            return
        result = subprocess.run(
            ["uv", "pip", "install", "--no-deps", str(REPO_ROOT / rel_path)],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
        )
        invalidate_caches()
        if result.returncode != 0 or find_spec(module) is None:
            pytest.skip(f"could not install v0 env {module!r}:\n{result.stderr[-600:]}")

    return _ensure
