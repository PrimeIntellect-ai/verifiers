"""Unit tests for module-level debate_rubric helpers.

Covers the reward-math primitives independently of pack / judge setup:
  - ``zero_sum_reward``: including tie (→ 0 for all) and judge neutrality
  - ``winning_member``: reverse-walk to last judge step; None on absent /
    decision-missing / wrong-ordering
"""

from __future__ import annotations

from typing import Any

from verifiers.envs.debate_rubric import winning_member, zero_sum_reward


def _step(member_id: str, **extras: Any) -> dict[str, Any]:
    """Minimal trajectory step for ``winning_member`` testing."""
    return {"extras": {"member_id": member_id, **extras}}


# -----------------------------------------------------------------------
# zero_sum_reward
# -----------------------------------------------------------------------


def test_zero_sum_winner_gets_plus_one() -> None:
    assert zero_sum_reward("debater_a", "debater_a") == 1.0


def test_zero_sum_loser_gets_minus_one() -> None:
    assert zero_sum_reward("debater_b", "debater_a") == -1.0


def test_zero_sum_judge_always_zero() -> None:
    assert zero_sum_reward("judge", "debater_a") == 0.0
    assert zero_sum_reward("judge", None) == 0.0
    assert zero_sum_reward("judge", "tie") == 0.0


def test_zero_sum_none_winner_zero_reward_for_all() -> None:
    # "no decision" — errored or missing judge → no reward for anyone.
    assert zero_sum_reward("debater_a", None) == 0.0
    assert zero_sum_reward("debater_b", None) == 0.0


def test_zero_sum_tie_zero_reward_for_all() -> None:
    # "tie" — judge declared a draw. Both debaters get 0; zero-sum
    # invariant holds trivially (0 + 0 = 0) and RAE baseline update is
    # neutral. Without this handling both debaters would get −1 each
    # (member_id == "tie" fails for both).
    assert zero_sum_reward("debater_a", "tie") == 0.0
    assert zero_sum_reward("debater_b", "tie") == 0.0


# -----------------------------------------------------------------------
# winning_member
# -----------------------------------------------------------------------


def test_winning_member_empty_trajectory() -> None:
    assert winning_member([]) is None


def test_winning_member_no_judge_step() -> None:
    # Only debaters committed; no judge verdict.
    traj = [_step("debater_a"), _step("debater_b")]
    assert winning_member(traj) is None


def test_winning_member_reads_last_judge_decision() -> None:
    traj = [
        _step("debater_a"),
        _step("debater_b"),
        _step("judge", fields={"decision": "debater_a"}),
    ]
    assert winning_member(traj) == "debater_a"


def test_winning_member_respects_reverse_walk() -> None:
    # If (pathologically) two judge steps appear, the LAST one wins.
    # Earlier verdicts are stale.
    traj = [
        _step("judge", fields={"decision": "debater_a"}),
        _step("debater_b"),
        _step("judge", fields={"decision": "debater_b"}),
    ]
    assert winning_member(traj) == "debater_b"


def test_winning_member_breaks_on_first_judge_without_decision() -> None:
    # Reverse-walks and breaks on the first judge step — whether or not
    # it carries a decision. Falling back to an earlier judge would
    # silently install a stale verdict.
    traj = [
        _step("judge", fields={"decision": "debater_a"}),
        _step("debater_b"),
        _step("judge", fields={}),  # malformed / empty
    ]
    assert winning_member(traj) is None


def test_winning_member_accepts_tie_decision() -> None:
    traj = [_step("judge", fields={"decision": "tie"})]
    assert winning_member(traj) == "tie"
