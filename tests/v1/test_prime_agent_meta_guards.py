"""Deterministic tests for the `_meta`-dependent Prime Agent reward guards.

These run offline: they feed synthetic `trace.info["acp_meta"]` histories, so a
weakened guard fails in CI instead of only showing up in a live rollout.
"""

from types import SimpleNamespace

import pytest
from prime_agent_meta_guards import (
    NAMESPACE,
    MissingAcpMeta,
    autonomous_continued,
    child_tokens_attributed,
    gate_attempted,
    gate_failure_reported,
    no_outstanding_subagents,
    observed_child_statuses,
    refinement_applied,
    spawned_and_finished,
)
from prime_agent_negatives_v1 import child_error_reported, quiescence_blocked_scoring


def trace_with(*events: dict) -> SimpleNamespace:
    return SimpleNamespace(info={"acp_meta": {NAMESPACE: list(events)}})


def test_missing_metadata_raises_instead_of_scoring_zero():
    """A broken harness must not look like a failing model."""
    for empty in (SimpleNamespace(info={}), SimpleNamespace(info={"acp_meta": {}})):
        with pytest.raises(MissingAcpMeta):
            spawned_and_finished(empty)
    # Present envelope but no subagents key is still missing evidence.
    with pytest.raises(MissingAcpMeta):
        spawned_and_finished(trace_with({"autonomous": {"enabled": True}}))
    # An empty history must raise too: a guard that quietly degrades to "no
    # evidence found" would score a broken harness as a failing model.
    with pytest.raises(MissingAcpMeta):
        spawned_and_finished(trace_with())
    with pytest.raises(MissingAcpMeta):
        autonomous_continued(trace_with())


def test_subagent_lifecycle_needs_a_real_transition():
    running = {"subagents": [{"id": "c1", "status": "running", "tokenCount": 0}]}
    done = {"subagents": [{"id": "c1", "status": "completed", "tokenCount": 120}]}
    assert spawned_and_finished(trace_with(running, done))
    assert observed_child_statuses(trace_with(running, done)) == {
        "c1": ["running", "completed"]
    }
    # Terminal-only history cannot prove the child ever ran, so ordering matters.
    assert not spawned_and_finished(trace_with(done))
    # Never finishing must not score.
    assert not spawned_and_finished(trace_with(running))


def test_child_token_attribution_requires_nonzero_usage():
    assert child_tokens_attributed(
        trace_with(
            {"subagents": [{"id": "c1", "status": "completed", "tokenCount": 5}]}
        )
    )
    assert not child_tokens_attributed(
        trace_with(
            {"subagents": [{"id": "c1", "status": "completed", "tokenCount": 0}]}
        )
    )


def test_quiescence_guards_track_the_final_snapshot():
    busy = {
        "quiescence": {"outstandingSubagents": 2, "remainingAutonomousContinuations": 1}
    }
    idle = {
        "quiescence": {"outstandingSubagents": 0, "remainingAutonomousContinuations": 0}
    }
    assert no_outstanding_subagents(trace_with(busy, idle))
    assert not no_outstanding_subagents(trace_with(idle, busy))
    assert quiescence_blocked_scoring(trace_with(busy, idle))
    assert not quiescence_blocked_scoring(trace_with(idle))


def test_autonomous_guards_reject_inert_configuration():
    inert = {"autonomous": {"enabled": True, "continuationsUsed": 0}}
    engaged = {
        "autonomous": {"enabled": True, "continuationsUsed": 2, "gateAttempt": 1}
    }
    assert autonomous_continued(trace_with(engaged))
    assert gate_attempted(trace_with(engaged))
    # Configured but never continued is exactly the silent-inert failure.
    assert not autonomous_continued(trace_with(inert))
    assert not gate_attempted(trace_with(inert))


def test_failure_guards_require_reported_failure():
    assert gate_failure_reported(
        trace_with({"autonomous": {"enabled": True, "gateFailure": "exit 1"}})
    )
    assert not gate_failure_reported(trace_with({"autonomous": {"enabled": True}}))
    assert child_error_reported(
        trace_with(
            {"subagents": [{"id": "c1", "status": "running"}]},
            {"subagents": [{"id": "c1", "status": "error"}]},
        )
    )
    assert not child_error_reported(
        trace_with({"subagents": [{"id": "c1", "status": "completed"}]})
    )


def test_refinement_requires_enumerated_changes():
    assert refinement_applied(
        trace_with(
            {"refinement": {"status": "complete", "changes": ["create memory:x"]}}
        )
    )
    # "complete" with no enumerated edits proves nothing changed.
    assert not refinement_applied(
        trace_with({"refinement": {"status": "complete", "changes": []}})
    )


def test_harness_state_reward_falls_through_to_goal():
    """A missing `refinement` envelope must not hide a present `goal`.

    The guards raise on absent metadata, so a naive `refinement or goal` never
    evaluated the second surface. Only a completely empty envelope is unscoreable.
    """
    import asyncio

    from prime_agent_harness_state_v1 import PrimeAgentHarnessStateTask

    task = PrimeAgentHarnessStateTask.__new__(PrimeAgentHarnessStateTask)
    goal_only = trace_with({"goal": {"status": "active", "objective": "x"}})
    assert asyncio.run(PrimeAgentHarnessStateTask.harness_state(task, goal_only)) == 1.0

    refine_only = trace_with(
        {"refinement": {"status": "complete", "changes": ["create memory:x"]}}
    )
    assert (
        asyncio.run(PrimeAgentHarnessStateTask.harness_state(task, refine_only)) == 1.0
    )

    # Envelope present but neither surface: a real zero, not missing evidence.
    other = trace_with({"autonomous": {"enabled": True}})
    assert asyncio.run(PrimeAgentHarnessStateTask.harness_state(task, other)) == 0.0

    # Nothing preserved at all: unscoreable, so raise rather than score zero.
    empty = SimpleNamespace(info={})
    try:
        asyncio.run(PrimeAgentHarnessStateTask.harness_state(task, empty))
        raise AssertionError("expected MissingAcpMeta for an empty envelope")
    except MissingAcpMeta:
        pass
