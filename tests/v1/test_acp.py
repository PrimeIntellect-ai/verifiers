"""ACP metadata accumulation preserves ordered, namespaced extension events."""

from pathlib import Path

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


def test_acp_run_forwards_trace_to_the_recording_path() -> None:
    """`ACP.run` must hand `trace` to `_run`, or every caller's opt-in is a no-op.

    This was a silent hole: `run()` accepted `trace` and dropped it, so the
    one-shot path recorded nothing while appearing wired up. A signature-level
    check is enough and stays honest without a live runtime.
    """
    import inspect

    from verifiers.v1.acp import ACP

    assert "trace" in inspect.signature(ACP.run).parameters
    assert "trace" in inspect.signature(ACP._run).parameters
    source = inspect.getsource(ACP.run)
    # The forwarding argument itself, not merely the parameter.
    assert "trace=trace" in source, (
        "ACP.run accepts trace but never forwards it to _run"
    )


def test_acp_runner_preserves_late_turn_metadata_and_drops_failed_turn_metadata() -> (
    None
):
    """Guard the standalone runner's metadata ownership across live turns."""
    runner = (Path(__file__).parents[2] / "verifiers/v1/acp/runner.py").read_text()
    reset_start = runner.index("    def reset(self) -> None:")
    reset_end = runner.index("    async def session_update", reset_start)
    assert "turn_acp_meta = {}" not in runner[reset_start:reset_end]
    prompt_end = runner.index(
        "\n\nasync def run_once", runner.index("async def prompt(")
    )
    prompt_source = runner[runner.index("async def prompt(") : prompt_end]
    assert "LATE_UPDATE_GRACE_SECONDS" in prompt_source
    assert "wait_for" in prompt_source
    assert (
        "client.output_changed.wait_for(lambda: bool(client.turn_acp_meta))"
        in prompt_source
    )
    error_start = runner.index("            except Exception as error:")
    error_end = runner.index("            write_packet", error_start)
    error_source = runner[error_start:error_end]
    assert 'response["meta"]' not in error_source
    assert "session.client.turn_acp_meta = {}" in error_source
