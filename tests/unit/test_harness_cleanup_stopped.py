"""A harness's cleanup skips a runtime already torn down: a rollout whose setup failed
stopped its box first, and cleaning up then ran on a backend that had released its client
("'NoneType' object has no attribute 'start_background_job'")."""

from types import SimpleNamespace

from verifiers.v1.rollout import Rollout


async def cleanup(stopped: bool) -> list[str]:
    calls: list[str] = []

    async def harness_cleanup(trace, runtime) -> None:
        calls.append(trace.id)

    rollout = SimpleNamespace(
        _harness_cleaned=False,
        harness=SimpleNamespace(cleanup=harness_cleanup),
        trace=SimpleNamespace(id="t"),
    )
    runtime = SimpleNamespace(backend=SimpleNamespace(stopped=stopped))
    await Rollout._cleanup_harness(rollout, runtime)  # type: ignore[arg-type]
    await Rollout._cleanup_harness(rollout, runtime)  # type: ignore[arg-type]
    assert rollout._harness_cleaned
    return calls


async def test_cleanup_runs_once_on_a_live_runtime():
    assert await cleanup(stopped=False) == ["t"]


async def test_cleanup_skips_a_stopped_runtime():
    assert await cleanup(stopped=True) == []
