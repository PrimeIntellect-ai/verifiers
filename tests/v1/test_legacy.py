"""v0 backwards-compat: a legacy env runs through the bridge and yields a v1-shaped Trace.

`run_legacy_eval` loads a classic `verifiers.load_environment` env and bridges its output to
the same `Trace` type a native v1 run produces. Here we run a trivial v0 single-turn
(reverse_text) and multi-turn (alphabet_sort) env and assert the bridged Trace has the same
output shape (Trace / Turn / Response fields) as the matching native v1 run.
"""

import pytest

pytestmark = pytest.mark.e2e


def _shape(trace) -> dict:
    """Structural shape of a trace: the Trace, first-Turn, and Response field names."""
    dump = trace.model_dump()
    turn = dump["trajectory"][0]
    return {"trace": set(dump), "turn": set(turn), "response": set(turn["response"])}


async def test_v0_single_turn_matches_v1_shape(run_v0, run_v1, ensure_v0, tmp_path):
    ensure_v0("reverse_text", "environments/reverse_text")
    (v0,) = await run_v0("reverse-text", output_dir=tmp_path / "v0")
    (v1,) = await run_v1(
        "reverse-text-v1", runtime="subprocess", output_dir=tmp_path / "v1", max_turns=2,
    )
    assert v0.trajectory  # the bridge populated a trajectory
    assert v0.num_turns == 1
    assert isinstance(v0.reward, float)
    assert _shape(v0) == _shape(v1)


async def test_v0_multi_turn_matches_v1_shape(run_v0, run_v1, ensure_v0, tmp_path):
    ensure_v0("alphabet_sort", "environments/alphabet_sort")
    (v0,) = await run_v0("alphabet-sort", output_dir=tmp_path / "v0", max_tokens=256)
    (v1,) = await run_v1(
        "alphabet-sort-v1", runtime="subprocess", output_dir=tmp_path / "v1",
        max_turns=4, max_tokens=256,
        taskset_overrides={
            "min_turns": 2, "max_turns": 2,
            "min_names_per_turn": 1, "max_names_per_turn": 1,
        },
    )
    assert v0.num_turns >= 2  # genuinely multi-turn
    assert _shape(v0) == _shape(v1)
