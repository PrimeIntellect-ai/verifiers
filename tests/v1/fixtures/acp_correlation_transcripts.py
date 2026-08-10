"""Exact producer-shaped ACP correlation envelopes shared by consumer tests."""

NAMESPACE = "ai.primeintellect.prime-agent"


def success() -> list[dict]:
    return [
        {"promptTurnId": 1, "eventSequence": 11, "phase": "event", "kind": "progress"},
        {
            "promptTurnId": 1,
            "eventSequence": 12,
            "phase": "responseBoundary",
            "outcome": "result",
        },
        {
            "promptTurnId": 1,
            "eventSequence": 13,
            "phase": "terminalQuiescence",
            "outcome": "result",
            "quiescence": {
                "outstandingSubagents": 0,
                "remainingAutonomousContinuations": 0,
            },
        },
    ]


def error_terminal() -> list[dict]:
    return [
        {
            "promptTurnId": 1,
            "eventSequence": 21,
            "phase": "responseBoundary",
            "outcome": "error",
        },
        {
            "promptTurnId": 1,
            "eventSequence": 22,
            "phase": "terminalQuiescence",
            "outcome": "error",
            "quiescence": {
                "outstandingSubagents": 0,
                "remainingAutonomousContinuations": 0,
            },
        },
    ]


def error_incomplete() -> list[dict]:
    return [
        {
            "promptTurnId": 1,
            "eventSequence": 31,
            "phase": "responseBoundary",
            "outcome": "error",
        }
    ]


def cancelled() -> list[dict]:
    return []


def late_child() -> list[dict]:
    return [
        {
            "promptTurnId": 1,
            "eventSequence": 41,
            "phase": "event",
            "child": {"id": "late", "status": "done"},
        }
    ]


def global_sequence_turn_two() -> list[dict]:
    return [
        {
            "promptTurnId": 2,
            "eventSequence": 51,
            "phase": "responseBoundary",
            "outcome": "result",
        },
        {
            "promptTurnId": 2,
            "eventSequence": 52,
            "phase": "terminalQuiescence",
            "outcome": "result",
            "quiescence": {
                "outstandingSubagents": 0,
                "remainingAutonomousContinuations": 0,
            },
        },
    ]
