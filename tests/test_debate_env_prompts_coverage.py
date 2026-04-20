"""Tests for DebateEnv's init-time schedule × prompts coverage check.

The cross-check rejects packs that don't supply templates for every
(member_id, phase) pair the schedule will hit. Failure modes it catches:

  * system missing for a scheduled member  → would KeyError on first turn
  * question missing for a scheduled member → would silently render no Q
  * user[member][phase] missing AND no    → would silently render no
    'default' fallback                       instruction

These are caught at env.__init__ so the operator sees the error before
any rollout completes (~minutes saved, vs hours-of-GPU-then-confusion).
Dynamic SlotProgram implementations are exempt — their (member, phase)
set isn't enumerable at init.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import jinja2
import jinja2.sandbox
import pytest

from verifiers.envs.debate.prompts import DebatePrompts
from verifiers.envs.debate_env import DebateEnv
from verifiers.envs.multi_agent_kernel import StaticSchedule, TurnSlot


def _stub_rubric(members: list[str]) -> MagicMock:
    """A rubric stub with an awaitable teardown — Environment's atexit
    handler calls ``await self.rubric.teardown()`` and a bare MagicMock
    raises TypeError there (cosmetic, but noisy in pytest output)."""
    rubric = MagicMock()
    rubric.members = members
    rubric.teardown = AsyncMock(return_value=None)
    return rubric


_je = jinja2.sandbox.SandboxedEnvironment(undefined=jinja2.StrictUndefined)


def _pack(
    *,
    system: dict[str, str] | None = None,
    question: dict[str, str] | None = None,
    user: dict[str, dict[str, str]] | None = None,
) -> DebatePrompts:
    """Build a minimal DebatePrompts from raw template strings.

    Each kwarg defaults to a permissive everywhere-rendered map for the
    canonical members ``debater_a / debater_b / judge``; pass an explicit
    dict to override (e.g. omit a key to simulate a missing template).
    """
    if system is None:
        system = {m: "system" for m in ("debater_a", "debater_b", "judge")}
    if question is None:
        question = {m: "{{ task_prompt }}" for m in ("debater_a", "debater_b", "judge")}
    if user is None:
        user = {
            "debater_a": {"propose": "argue", "critique": "rebut"},
            "debater_b": {"propose": "argue", "critique": "rebut"},
            "judge": {"final": "decide"},
        }
    return DebatePrompts(
        system={k: _je.from_string(v) for k, v in system.items()},
        user={
            m: {p: _je.from_string(t) for p, t in phases.items()}
            for m, phases in user.items()
        },
        question={k: _je.from_string(v) for k, v in question.items()},
        fields={},
        think_visibility={},
        think_tag="thinking",
        prefill={},
        opponent_wrap=None,
        judges={},
        source_ref="<test>",
    )


_FULL_SCHEDULE = StaticSchedule(
    (
        TurnSlot(slot_id=0, agents=("debater_a",), phase="propose"),
        TurnSlot(slot_id=1, agents=("debater_b",), phase="propose"),
        TurnSlot(slot_id=2, agents=("debater_a",), phase="critique"),
        TurnSlot(slot_id=3, agents=("debater_b",), phase="critique"),
        TurnSlot(slot_id=4, agents=("judge",), phase="final"),
    )
)


def _build(
    prompts: DebatePrompts, schedule: StaticSchedule = _FULL_SCHEDULE
) -> DebateEnv:
    """Construct DebateEnv with a stub rubric. The cross-check 1
    (members ≡ rubric.members) is satisfied because the stub exposes
    matching .members; cross-check 2 is satisfied because the schedule
    agents match the members list."""
    return DebateEnv(
        schedule=schedule,
        prompts=prompts,
        members=["debater_a", "debater_b", "judge"],
        rubric=_stub_rubric(["debater_a", "debater_b", "judge"]),
        dataset=lambda: None,
    )


# ---------------------------------------------------------------------------
# Happy path
# ---------------------------------------------------------------------------


def test_full_pack_passes():
    """Pack with templates for every (member, phase) → init succeeds."""
    _build(_pack())  # no exception


def test_user_default_fallback_passes():
    """Missing user[member][phase] is OK when user[member]['default']
    exists — DebatePrompts.render_instruction's documented fallback."""
    user = {
        "debater_a": {"default": "do something"},  # NO propose/critique
        "debater_b": {"default": "do something"},
        "judge": {"final": "decide"},
    }
    _build(_pack(user=user))  # no exception — default covers propose+critique


# ---------------------------------------------------------------------------
# Loud-failure mode: missing system
# ---------------------------------------------------------------------------


def test_missing_system_for_scheduled_member_raises():
    """system missing → would KeyError on first render. Catch at init."""
    sys = {"debater_a": "x", "debater_b": "x"}  # judge missing
    with pytest.raises(ValueError) as ei:
        _build(_pack(system=sys))
    msg = str(ei.value)
    assert "system: missing" in msg
    assert "'judge'" in msg
    assert "KeyError" in msg


def test_missing_system_lists_all_offenders():
    sys = {"debater_a": "x"}  # debater_b AND judge missing
    with pytest.raises(ValueError) as ei:
        _build(_pack(system=sys))
    msg = str(ei.value)
    assert "'debater_b'" in msg
    assert "'judge'" in msg


# ---------------------------------------------------------------------------
# Silent-failure modes: missing question / user
# ---------------------------------------------------------------------------


def test_missing_question_raises_with_silent_label():
    q = {
        "debater_a": "{{ task_prompt }}",
        "debater_b": "{{ task_prompt }}",
    }  # judge missing
    with pytest.raises(ValueError) as ei:
        _build(_pack(question=q))
    msg = str(ei.value)
    assert "question: missing" in msg
    assert "'judge'" in msg
    assert "silent" in msg.lower()


def test_missing_user_phase_no_default_raises():
    user = {
        "debater_a": {"propose": "argue"},  # NO critique, NO default
        "debater_b": {"propose": "argue", "critique": "rebut"},
        "judge": {"final": "decide"},
    }
    with pytest.raises(ValueError) as ei:
        _build(_pack(user=user))
    msg = str(ei.value)
    assert "user['debater_a']" in msg
    assert "'critique'" in msg
    assert "no 'default' fallback" in msg
    assert "silent" in msg.lower()


def test_missing_user_groups_phases_per_member():
    """Multiple missing phases on the same member are grouped in one line."""
    user = {
        "debater_a": {},  # missing BOTH propose AND critique, no default
        "debater_b": {"propose": "argue", "critique": "rebut"},
        "judge": {"final": "decide"},
    }
    with pytest.raises(ValueError) as ei:
        _build(_pack(user=user))
    msg = str(ei.value)
    # Same member, both phases listed together.
    assert "user['debater_a']" in msg
    assert "'critique'" in msg
    assert "'propose'" in msg


# ---------------------------------------------------------------------------
# Combined-failure: error message lists ALL categories
# ---------------------------------------------------------------------------


def test_combined_failures_all_reported():
    """When system + question + user are all incomplete, the error
    enumerates every category (no eager-exit on first failure)."""
    sys = {"debater_a": "x", "debater_b": "x"}  # judge missing
    q = {"debater_a": "{{ task_prompt }}"}  # b + judge missing
    user = {
        "debater_a": {"propose": "argue"},  # critique missing, no default
        "debater_b": {"propose": "argue", "critique": "rebut"},
        "judge": {"final": "decide"},
    }
    with pytest.raises(ValueError) as ei:
        _build(_pack(system=sys, question=q, user=user))
    msg = str(ei.value)
    assert "system: missing" in msg
    assert "question: missing" in msg
    assert "user['debater_a']" in msg
    # Sanity: the pack-source-ref is mentioned for debugging.
    assert "<test>" in msg
    # Sanity: lists what the pack DOES have to help the operator.
    assert "pack has system keys" in msg
    assert "pack has user keys" in msg


# ---------------------------------------------------------------------------
# Dynamic schedules are exempt
# ---------------------------------------------------------------------------


def test_dynamic_schedule_skips_check():
    """A SlotProgram that isn't a StaticSchedule has no enumerable
    (member, phase) set — check is silently skipped (its violations
    surface at first render, which is the best we can do)."""

    class _DynamicProgram:
        def current_slot(self, state):
            return None  # episode finishes immediately

    # An intentionally-broken pack that would fail the check if static —
    # but with a dynamic program, init succeeds.
    bad_pack = _pack(system={"debater_a": "x"})  # b + judge missing
    DebateEnv(
        schedule=_DynamicProgram(),
        prompts=bad_pack,
        members=["debater_a", "debater_b", "judge"],
        rubric=_stub_rubric(["debater_a", "debater_b", "judge"]),
        dataset=lambda: None,
    )  # no exception


# ---------------------------------------------------------------------------
# Nontrivial schedule — variant protocols
# ---------------------------------------------------------------------------


def test_consultancy_shaped_schedule_pack_passes():
    """A 2-member consultancy-shaped MA env: consultant + judge,
    one argue phase, one final phase. Exact prompt coverage required."""
    schedule = StaticSchedule(
        (
            TurnSlot(slot_id=0, agents=("consultant",), phase="argue"),
            TurnSlot(slot_id=1, agents=("judge",), phase="final"),
        )
    )
    pack = DebatePrompts(
        system={k: _je.from_string("system") for k in ("consultant", "judge")},
        user={
            "consultant": {"argue": _je.from_string("defend the assigned answer")},
            "judge": {"final": _je.from_string("decide")},
        },
        question={
            k: _je.from_string("{{ task_prompt }}") for k in ("consultant", "judge")
        },
        fields={},
        think_visibility={},
        think_tag="thinking",
        prefill={},
        opponent_wrap=None,
        judges={},
        source_ref="<test>",
    )
    DebateEnv(
        schedule=schedule,
        prompts=pack,
        members=["consultant", "judge"],
        rubric=_stub_rubric(["consultant", "judge"]),
        dataset=lambda: None,
    )  # no exception


def test_consultancy_shaped_schedule_missing_judge_final_raises():
    """Same shape, but pack omits judge's 'final' user template AND no
    default — the silent-no-instruction failure mode for the judge turn."""
    schedule = StaticSchedule(
        (
            TurnSlot(slot_id=0, agents=("consultant",), phase="argue"),
            TurnSlot(slot_id=1, agents=("judge",), phase="final"),
        )
    )
    pack = DebatePrompts(
        system={k: _je.from_string("system") for k in ("consultant", "judge")},
        user={
            "consultant": {"argue": _je.from_string("defend")},
            "judge": {},  # no final, no default
        },
        question={
            k: _je.from_string("{{ task_prompt }}") for k in ("consultant", "judge")
        },
        fields={},
        think_visibility={},
        think_tag="thinking",
        prefill={},
        opponent_wrap=None,
        judges={},
        source_ref="<test>",
    )
    with pytest.raises(ValueError) as ei:
        DebateEnv(
            schedule=schedule,
            prompts=pack,
            members=["consultant", "judge"],
            rubric=_stub_rubric(["consultant", "judge"]),
            dataset=lambda: None,
        )
    msg = str(ei.value)
    assert "user['judge']" in msg
    assert "'final'" in msg
