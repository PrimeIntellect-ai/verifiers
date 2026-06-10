"""v0 backwards-compat: a legacy env runs through the bridge and yields a v1-shaped Trace.

`run_legacy_eval` loads a classic `verifiers.load_environment` env and bridges its output to
the same `Trace` type a native v1 run produces. Here we run the v0 echo fixtures (single-turn
+ multi-turn) and assert the bridged Trace has the same output shape (Trace / Turn / Response
fields) as a native v1 run (echo-v1).
"""

import pytest

pytestmark = pytest.mark.e2e


def _shape(trace) -> dict:
    """Structural shape of a trace: the Trace, first-Turn, and Response field names."""
    dump = trace.model_dump()
    turn = dump["trajectory"][0]
    return {"trace": set(dump), "turn": set(turn), "response": set(turn["response"])}


async def test_v0_single_turn_matches_v1_shape(run_v0, run_v1, tmp_path):
    (v0,) = await run_v0("echo-v0", output_dir=tmp_path / "v0")
    (v1,) = await run_v1(
        "echo-v1", runtime="subprocess", output_dir=tmp_path / "v1", max_turns=2
    )
    assert v0.trajectory  # the bridge populated a trajectory
    assert v0.num_turns == 1
    assert isinstance(v0.reward, float)
    assert _shape(v0) == _shape(v1)


async def test_v0_multi_turn_matches_v1_shape(run_v0, run_v1, tmp_path):
    (v0,) = await run_v0("echo-multi-v0", output_dir=tmp_path / "v0")
    (v1,) = await run_v1(
        "echo-v1", runtime="subprocess", output_dir=tmp_path / "v1", max_turns=2
    )
    assert v0.num_turns >= 2  # genuinely multi-turn
    assert _shape(v0) == _shape(v1)
