"""Resume seeds the dashboard with the kept on-disk rollouts.

`resume.plan` returns a `Baseline` over the rows it keeps (so a resume's progress counter and
reward cover the whole run, not just the resumed rollouts), and `format_mean` folds that baseline
into both the error-corrected and the global mean.
"""

import json
from pathlib import Path

from verifiers.v1.cli.eval import resume
from verifiers.v1.utils.format import format_mean


class _FakeTrace:
    """Minimal stand-in for the two attributes `format_mean` reads."""

    def __init__(self, reward: float, errored: bool = False) -> None:
        self._reward = reward
        self._errored = errored

    @property
    def reward(self) -> float:
        return self._reward

    @property
    def has_error(self) -> bool:
        return self._errored


def _value(t: _FakeTrace) -> float:
    return t.reward


def test_format_mean_without_base_is_unchanged() -> None:
    assert format_mean([], _value) == "—"
    assert format_mean([_FakeTrace(1.0), _FakeTrace(0.0)], _value) == "0.50"
    # one of three errored → error-corrected mean, then global (errored as 0) in parens
    traces = [_FakeTrace(1.0), _FakeTrace(1.0), _FakeTrace(0.0, errored=True)]
    assert format_mean(traces, _value) == "1.00 (0.67)"


def test_format_mean_folds_baseline_into_both_means() -> None:
    # 10 kept rows summing to 9.0, no live rows → 0.90
    assert format_mean([], _value, base_sum=9.0, base_n=10) == "0.90"
    # + one clean live row → (9+1)/11
    assert format_mean([_FakeTrace(1.0)], _value, base_sum=9.0, base_n=10) == "0.91"
    # + a clean and an errored live row: clean (9+1)/11=0.909; global (9+1+0)/12=0.833
    mixed = [_FakeTrace(1.0), _FakeTrace(0.0, errored=True)]
    assert format_mean(mixed, _value, base_sum=9.0, base_n=10) == "0.91 (0.83)"


def _write_results(path: Path, rows: list[dict]) -> None:
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))


def test_plan_aggregates_kept_rows_into_baseline(tmp_path: Path) -> None:
    _write_results(
        tmp_path / "results.jsonl",
        [
            {"task": {"idx": 0}, "rewards": {"solved": 1.0}, "metrics": {}},
            {"task": {"idx": 0}, "rewards": {"solved": 0.0}, "metrics": {}},
            {"task": {"idx": 1}, "rewards": {"solved": 1.0}, "metrics": {}},
            {"task": {"idx": 2}, "errors": [{"type": "SandboxError"}], "rewards": {}},
        ],
    )
    keep, owed, baseline = resume.plan(tmp_path, [0, 1, 2], num_rollouts=2, group=False)

    # task 0: 2 good (kept); task 1: 1 good (kept) + 1 owed; task 2: errored → dropped, 2 owed
    assert len(keep) == 3
    assert owed == {1: 1, 2: 2}
    # the baseline covers exactly the kept rows (all non-errored)
    assert baseline.n == 3
    assert baseline.reward_sum == 2.0
    assert baseline.comp_sum["rewards"]["solved"] == 2.0


def test_plan_group_drops_incomplete_groups(tmp_path: Path) -> None:
    # group-scored: a task is kept only if fully complete (>= num_rollouts), else redone whole
    _write_results(
        tmp_path / "results.jsonl",
        [
            {"task": {"idx": 0}, "rewards": {"solved": 1.0}, "metrics": {}},
            {"task": {"idx": 0}, "rewards": {"solved": 1.0}, "metrics": {}},
            {"task": {"idx": 1}, "rewards": {"solved": 1.0}, "metrics": {}},
        ],
    )
    keep, owed, baseline = resume.plan(tmp_path, [0, 1], num_rollouts=2, group=True)
    # task 0 complete (kept, both rows); task 1 has only 1/2 → whole group redone, kept none
    assert len(keep) == 2
    assert owed == {1: 2}
    assert baseline.n == 2
    assert baseline.reward_sum == 2.0
