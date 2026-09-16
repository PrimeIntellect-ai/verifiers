"""Smoke-eval every example taskset in `environments/` in-process.

Each taskset runs with its required harness for one short, capped rollout through the
shared e2e runner, so a broken example taskset fails CI. `compact` is excluded (it's a
harness, not a taskset); SWE/container tasksets need a docker/prime runtime and are
covered by dedicated V1 e2e tests.
"""

from pathlib import Path

import pytest

pytestmark = pytest.mark.e2e

ENVIRONMENTS = Path(__file__).parent.parent.parent / "environments"

# V1 tasksets that aren't part of the default CI install.
SKIP_EVAL = {"nemo_gym_weather"}


def v1_tasksets() -> list[str]:
    if not ENVIRONMENTS.is_dir():
        return []
    return sorted(
        d.name
        for d in ENVIRONMENTS.iterdir()
        if d.is_dir() and d.name != "compact" and (d / "pyproject.toml").exists()
    )


@pytest.mark.parametrize("taskset", v1_tasksets())
async def test_eval(run_v1, taskset: str, tmp_path: Path):
    """Run one capped rollout of `taskset`; a taskset that bundles a harness uses it by default."""
    if taskset in SKIP_EVAL:
        pytest.skip(f"{taskset} can't run a plain-CI smoke eval")
    # `harness=None`: every seat keeps the taskset's own harness; the runner caps each
    # seat's turns.
    traces = await run_v1(
        taskset,
        output_dir=tmp_path / taskset,
        harness=None,
        max_tokens=512,
        rollout_timeout=600,
    )
    assert traces, f"{taskset} produced no trace"
    for trace in traces:
        assert trace.ok, f"{taskset}: {trace.errors}"
