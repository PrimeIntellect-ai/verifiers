"""ACP metadata accumulation preserves ordered, namespaced extension events."""

from verifiers.v1.acp import _record_acp_meta


def test_acp_meta_accumulates_history_without_flattening() -> None:
    trace = type("TraceStub", (), {"info": {}})()
    namespace = "ai.primeintellect.prime-agent"
    _record_acp_meta(
        trace,
        {
            namespace: [
                {"autonomous": {"continuationsUsed": 0}},
                {
                    "quiescence": {
                        "outstandingSubagents": 1,
                        "remainingAutonomousContinuations": 2,
                    }
                },
            ],
            "other.namespace": [{"value": 1}],
        },
    )
    _record_acp_meta(
        trace,
        {
            namespace: [
                {
                    "quiescence": {
                        "outstandingSubagents": 0,
                        "remainingAutonomousContinuations": 0,
                    }
                }
            ]
        },
    )

    assert trace.info["acp_meta"][namespace] == [
        {"autonomous": {"continuationsUsed": 0}},
        {
            "quiescence": {
                "outstandingSubagents": 1,
                "remainingAutonomousContinuations": 2,
            }
        },
        {
            "quiescence": {
                "outstandingSubagents": 0,
                "remainingAutonomousContinuations": 0,
            }
        },
    ]
    assert trace.info["acp_meta"]["other.namespace"] == [{"value": 1}]
    assert trace.info["acp_meta"][namespace][-1]["quiescence"] == {
        "outstandingSubagents": 0,
        "remainingAutonomousContinuations": 0,
    }


def test_acp_meta_without_events_is_additive() -> None:
    trace = type("TraceStub", (), {"info": {"existing": "value"}})()

    _record_acp_meta(trace, {})

    assert trace.info == {"existing": "value"}
