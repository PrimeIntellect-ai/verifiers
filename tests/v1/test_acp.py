"""ACP metadata accumulation preserves ordered, namespaced extension events."""

import importlib.util
import sys
import types
from pathlib import Path

import pytest

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


def load_runner_without_acp_dependency(monkeypatch: pytest.MonkeyPatch):
    """Load the standalone script with only the ACP names this unit path needs."""
    acp = types.ModuleType("acp")
    acp.PROTOCOL_VERSION = "0.11"
    acp.Client = object
    acp.RequestError = RuntimeError
    acp.image_block = lambda data, media_type: (data, media_type)
    acp.spawn_agent_process = None
    acp.text_block = lambda text: text
    schema = types.ModuleType("acp.schema")
    for name in (
        "AgentMessageChunk",
        "AllowedOutcome",
        "ClientCapabilities",
        "DeniedOutcome",
        "HttpMcpServer",
        "PermissionOption",
        "RequestPermissionResponse",
        "SessionInfoUpdate",
        "TextContentBlock",
        "ToolCall",
        "ToolCallUpdate",
    ):
        setattr(schema, name, type(name, (), {}))
    monkeypatch.setitem(sys.modules, "acp", acp)
    monkeypatch.setitem(sys.modules, "acp.schema", schema)
    spec = importlib.util.spec_from_file_location(
        "test_acp_runner", Path(__file__).parents[2] / "verifiers/v1/acp/runner.py"
    )
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_wait_for_late_metadata_waits_for_quiet_not_first_event():
    """Stopping at the FIRST metadata event drops the trailing terminal update.

    ACP can dispatch several SessionInfoUpdates around the response, and the
    bucket is cleared once the response is built, so an event arriving after the
    first is lost or attributed to the next turn. runner.py executes under its own
    standalone ACP dependencies and cannot be imported here, so this asserts the
    settle contract at the source level.
    """
    source = Path("verifiers/v1/acp/runner.py").read_text()
    helper = source[source.index("async def wait_for_late_metadata") :]
    helper = helper[: helper.index("\ndef ")]
    # Loops until the event COUNT stops changing, rather than returning on the
    # first truthy bucket.
    assert "_meta_event_count(client) != seen" in helper
    assert "LATE_METADATA_SETTLE_SECONDS" in helper
    # Still bounded by the overall grace period.
    assert "LATE_UPDATE_GRACE_SECONDS" in helper
    # The old early-return on any existing metadata must be gone.
    assert "if client.turn_acp_meta:\n        return" not in helper
