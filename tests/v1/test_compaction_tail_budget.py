"""The retained compaction tail must be bounded in tokens, not just message count.

`keep_recent_messages` bounds the tail by COUNT, and a message is not a bounded
quantity: with 8192-token completions and multi-thousand-token eval observations,
six messages measured 22,555 tokens on a real rollout -- 69% of a 32,768 window,
from a 16,384 trigger. With a 3-turn compaction cooldown behind that, growth
continues into a hard vLLM `max_model_len` failure rather than a truncation.

The subtle half is ordering: trimming to fit and *then* walking back to a
resumable boundary re-adds the message the trim just dropped. That regression
returned a 9,316-token tail against an 8,192 budget, i.e. the bound silently did
nothing. `test_trim_is_not_undone_by_the_resumability_walk_back` pins it.
"""

import random

import pytest

from verifiers.v1.harnesses.skx.program import (
    _compaction_ready,
    _message_tokens,
    _split_for_compaction,
)

BUDGET = 8192


def _m(role, tokens, **kw):
    return {"role": role, "content": "x" * (tokens * 4), **kw}


def _tail(messages, keep_recent=6, budget=BUDGET):
    return _split_for_compaction(messages, keep_recent, budget)[2]


def _tokens(messages):
    return sum(_message_tokens(m) for m in messages)


def test_trim_is_not_undone_by_the_resumability_walk_back():
    """The regression: trim lands on a `tool`, walk-back restores what it dropped."""

    messages = (
        [_m("system", 200), _m("user", 400)]
        + [_m("assistant", 500), _m("tool", 500)] * 4
        + [_m("assistant", 9000), _m("tool", 100), _m("assistant", 100), _m("tool", 100)]
    )
    tail = _tail(messages, keep_recent=4)
    assert _tokens(tail) <= BUDGET, (_tokens(tail), [m["role"] for m in tail])


def test_a_fat_completion_in_the_tail_is_trimmed_away():
    """The measured shape: one 8192-token completion dominating six messages."""

    messages = (
        [_m("system", 400), _m("user", 900)]
        + [_m("assistant", 700), _m("tool", 2400)] * 6
        + [
            _m("assistant", 8192), _m("tool", 2400),
            _m("assistant", 900), _m("tool", 2400),
            _m("assistant", 600), _m("tool", 2400),
        ]
    )
    assert _tokens(_tail(messages)) <= BUDGET


def test_the_tail_never_starts_on_an_orphaned_tool_result():
    """A tail beginning at a `tool` has no assistant call to explain it."""

    messages = [
        _m("system", 100), _m("user", 100), _m("assistant", 9000),
        _m("tool", 50), _m("tool", 50), _m("tool", 50), _m("assistant", 50), _m("tool", 50),
    ]
    tail = _tail(messages)
    assert tail and tail[0]["role"] != "tool"


def test_the_final_exchange_survives_even_when_oversized():
    """A tail of nothing cannot be resumed from, so an oversized last turn stays."""

    messages = [_m("system", 100), _m("user", 100)] + [_m("assistant", 9000), _m("tool", 9000)] * 4
    tail = _tail(messages)
    assert len(tail) >= 1
    assert _tokens(tail) > BUDGET  # accepted: nothing smaller is resumable


def test_zero_budget_restores_count_only_behaviour():
    """The bound is opt-in; 0 must not change how existing runs compact."""

    messages = [_m("system", 100), _m("user", 100)] + [_m("assistant", 9000), _m("tool", 100)] * 3
    assert _tokens(_tail(messages, keep_recent=4, budget=0)) > BUDGET


def test_the_split_stays_lossless():
    """head + middle + tail must reconstruct the input exactly."""

    messages = [_m("system", 100), _m("user", 200)] + [_m("assistant", 4000), _m("tool", 2000)] * 5
    head, middle, tail = _split_for_compaction(messages, 6, BUDGET)
    assert head + middle + tail == messages


def test_compaction_is_not_starved_by_the_bound():
    """Trimming the tail grows the middle, so readiness must still hold."""

    messages = [_m("system", 400), _m("user", 900)] + [_m("assistant", 700), _m("tool", 2400)] * 6
    assert _compaction_ready(messages, 6, BUDGET)


@pytest.mark.parametrize("seed", range(12))
def test_bound_holds_across_randomized_histories(seed):
    """Fuzz: the invariant must not depend on the shape I happened to imagine."""

    rng = random.Random(seed)
    for _ in range(40):
        messages = [_m("system", rng.randint(50, 400)), _m("user", rng.randint(50, 900))]
        for _ in range(rng.randint(4, 40)):
            messages.append(_m("assistant", rng.choice([50, 300, 900, 4000, 8192])))
            if rng.random() < 0.75:
                messages.append(_m("tool", rng.choice([50, 400, 2400, 6000])))
        tail = _tail(messages, keep_recent=rng.randint(2, 10))
        assert tail, "tail must never be empty"
        assert tail[0]["role"] != "tool", "tail must be resumable"
        if len(tail) > 2:
            assert _tokens(tail) <= BUDGET, _tokens(tail)
