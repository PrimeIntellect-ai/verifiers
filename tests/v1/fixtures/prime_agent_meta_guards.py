"""Reward guards over preserved ACP `_meta`, shared by the Prime Agent fixtures.

Every guard reads `trace.info["acp_meta"][NAMESPACE]`, the ordered event history
recorded by `verifiers/v1/acp`. Ordering matters: a subagent's `running -> done`
transition only exists in the sequence, so a guard that inspected the last event
alone could not tell a finished child from one that never started.

These deliberately raise `MissingAcpMeta` when the metadata a fixture depends on
is absent. Returning 0.0 would report "the agent failed" for what is actually a
broken harness or an agent build that never emitted the field.
"""

NAMESPACE = "ai.primeintellect.prime-agent"


class MissingAcpMeta(RuntimeError):
    """Required Prime Agent metadata never arrived, so the run cannot be scored."""


def events(trace) -> list[dict]:
    recorded = (trace.info or {}).get("acp_meta")
    if not recorded or NAMESPACE not in recorded:
        raise MissingAcpMeta(
            f"no {NAMESPACE!r} metadata in trace.info['acp_meta']; the harness did "
            "not preserve inbound ACP _meta, or this agent build does not emit it"
        )
    return [event for event in recorded[NAMESPACE] if isinstance(event, dict)]


def field_history(trace, key: str) -> list[dict]:
    """Every value this metadata key took, in arrival order."""
    history = [
        event[key] for event in events(trace) if isinstance(event.get(key), dict)
    ]
    if not history:
        raise MissingAcpMeta(f"{NAMESPACE}.{key} never appeared in the ACP metadata")
    return history


def subagent_history(trace) -> list[list[dict]]:
    """Each `subagents` roster snapshot, in arrival order."""
    seen = events(trace)
    if not seen:
        raise MissingAcpMeta(f"{NAMESPACE} metadata history is empty; nothing to score")
    snapshots = [
        event["subagents"] for event in seen if isinstance(event.get("subagents"), list)
    ]
    if not snapshots:
        raise MissingAcpMeta(
            f"{NAMESPACE}.subagents never appeared in the ACP metadata"
        )
    return snapshots


def observed_child_statuses(trace) -> dict[str, list[str]]:
    """Status sequence per child id, so a lifecycle transition is checkable."""
    statuses: dict[str, list[str]] = {}
    for snapshot in subagent_history(trace):
        for child in snapshot:
            if not isinstance(child, dict):
                continue
            child_id = child.get("id")
            status = child.get("status")
            if not isinstance(child_id, str) or not isinstance(status, str):
                continue
            seen = statuses.setdefault(child_id, [])
            if not seen or seen[-1] != status:
                seen.append(status)
    if not statuses:
        raise MissingAcpMeta("no identifiable subagents appeared in the ACP metadata")
    return statuses


def spawned_and_finished(trace) -> bool:
    """A child ran and reached a terminal state before scoring.

    Interception cannot establish this: `ModelCall` carries no parent or agent
    field, so a child's model calls are indistinguishable from its parent's. The
    `subagents` roster is the only place this is observable.
    """
    statuses = observed_child_statuses(trace)
    terminal = {"completed", "done", "error", "cancelled"}
    return any(
        any(status not in terminal for status in seen) and seen[-1] in terminal
        for seen in statuses.values()
    )


def child_tokens_attributed(trace) -> bool:
    """Some child reported nonzero token usage."""
    return any(
        isinstance(child, dict)
        and isinstance(child.get("tokenCount"), int)
        and child["tokenCount"] > 0
        for snapshot in subagent_history(trace)
        for child in snapshot
    )


def quiescent_at_end(trace) -> bool:
    """The final quiescence snapshot shows no outstanding work.

    Scoring a turn that is still working races the agent, so a fixture that cares
    about completeness must assert this rather than trusting `end_turn`.
    """
    final = field_history(trace, "quiescence")[-1]
    return final.get("outstandingSubagents") == 0 and (
        final.get("remainingAutonomousContinuations") in (0, None)
        or isinstance(final.get("remainingAutonomousContinuations"), int)
    )


def no_outstanding_subagents(trace) -> bool:
    return field_history(trace, "quiescence")[-1].get("outstandingSubagents") == 0


def autonomous_continued(trace) -> bool:
    """Autonomous mode actually engaged, rather than being silently inert."""
    history = field_history(trace, "autonomous")
    return any(
        event.get("enabled")
        and isinstance(event.get("continuationsUsed"), int)
        and event["continuationsUsed"] > 0
        for event in history
    )


def gate_attempted(trace) -> bool:
    history = field_history(trace, "autonomous")
    return any(
        isinstance(event.get("gateAttempt"), int) and event["gateAttempt"] >= 1
        for event in history
    )


def gate_failure_reported(trace) -> bool:
    """A failing gate surfaced its failure instead of passing silently."""
    return any(event.get("gateFailure") for event in field_history(trace, "autonomous"))


def goal_progressed(trace) -> bool:
    statuses = [event.get("status") for event in field_history(trace, "goal")]
    return any(isinstance(status, str) and status for status in statuses)


def refinement_applied(trace) -> bool:
    """Continual-harness state changed, with the applied edits enumerated."""
    return any(
        event.get("status") == "complete"
        and isinstance(event.get("changes"), list)
        and event["changes"]
        for event in field_history(trace, "refinement")
    )
