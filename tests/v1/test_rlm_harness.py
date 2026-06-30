"""`summarize_at_tokens`: a `(lo, hi)` pair draws a per-group threshold seeded by task index;
an int is fixed; `None` disables (empty env value); bad ranges are rejected at config time."""

import pytest

from verifiers.v1.harnesses.rlm.harness import RLMHarness, RLMHarnessConfig


def make_harness(summarize_at_tokens) -> RLMHarness:
    return RLMHarness(
        RLMHarnessConfig(id="rlm", summarize_at_tokens=summarize_at_tokens)
    )


def test_range_draw_is_per_group_and_in_bounds() -> None:
    harness = make_harness((100, 200))
    for idx in range(8):
        first = harness.summarize_threshold(idx)
        assert first == harness.summarize_threshold(idx)
        assert 100 <= int(first) <= 200
    draws = {harness.summarize_threshold(idx) for idx in range(8)}
    assert len(draws) > 1


def test_fixed_and_disabled() -> None:
    assert make_harness(1500).summarize_threshold(3) == "1500"
    assert make_harness(None).summarize_threshold(3) == ""


def test_invalid_range_rejected_at_config_time() -> None:
    for bad in [(0, 100), (100, 50), (-1, 5)]:  # non-positive bound, lo > hi
        with pytest.raises(ValueError):
            RLMHarnessConfig(id="rlm", summarize_at_tokens=bad)
