"""ACP metadata accumulation preserves ordered, namespaced extension events."""

import asyncio
import hashlib
import importlib.util
import json
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


CANONICAL_TRANSCRIPT = (
    Path(__file__).parent / "fixtures" / "acp-correlation-transcripts.json"
)
CANONICAL_TRANSCRIPT_SHA256 = (
    "cacde7827aadf186db2ce1af1ea6f3b6d109504fa94945633bd9c0d96106b882"
)


def canonical_cases() -> dict[str, list[dict]]:
    raw = CANONICAL_TRANSCRIPT.read_bytes()
    assert hashlib.sha256(raw).hexdigest() == CANONICAL_TRANSCRIPT_SHA256
    document = json.loads(raw)
    assert document["schema"] == "ai.primeintellect.prime-agent/v1"
    return document["cases"]


def test_canonical_p2_fixture_is_exact_bytes_and_all_cases() -> None:
    cases = canonical_cases()
    assert set(cases) == {
        "success",
        "error_terminal",
        "error_incomplete",
        "cancelled",
        "late_child",
        "global_sequence_turn_two",
    }


def test_canonical_cases_preserve_reviewed_named_transcripts() -> None:
    cases = canonical_cases()

    success = cases["success"]
    assert [event["phase"] for event in success] == [
        "event",
        "responseBoundary",
        "terminalQuiescence",
    ]
    assert success[-1]["outcome"] == "result"
    assert success[-1]["quiescence"] == {
        "outstandingSubagents": 0,
        "remainingAutonomousContinuations": 0,
    }

    error_terminal = cases["error_terminal"]
    assert [event["phase"] for event in error_terminal] == [
        "responseBoundary",
        "terminalQuiescence",
    ]
    assert [event["outcome"] for event in error_terminal] == ["error", "error"]

    error_incomplete = cases["error_incomplete"]
    assert error_incomplete == [
        {
            "promptTurnId": 1,
            "eventSequence": 31,
            "phase": "responseBoundary",
            "outcome": "error",
        }
    ]

    assert cases["cancelled"] == []
    assert cases["late_child"] == [
        {
            "promptTurnId": 1,
            "eventSequence": 41,
            "phase": "event",
            "child": {"id": "late", "status": "done"},
        }
    ]

    global_sequence_turn_two = cases["global_sequence_turn_two"]
    assert [event["promptTurnId"] for event in global_sequence_turn_two] == [2, 2]
    assert [event["eventSequence"] for event in global_sequence_turn_two] == [51, 52]
    assert [event["phase"] for event in global_sequence_turn_two] == [
        "responseBoundary",
        "terminalQuiescence",
    ]


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


class _MetaClient:
    def __init__(self, initial=None):
        self.turn_acp_meta = dict(initial or {})
        self.output_changed = asyncio.Condition()

    async def emit(self, event):
        async with self.output_changed:
            self.turn_acp_meta.setdefault("ns", []).append(event)
            self.output_changed.notify_all()


@pytest.mark.asyncio
async def test_late_metadata_keeps_full_grace_before_the_first_event(monkeypatch):
    """A first event arriving after the settle interval must not be dropped.

    Waiting only for the stream to go quiet collapses the grace window down to
    the settle interval while nothing has arrived yet.
    """
    runner = load_runner_without_acp_dependency(monkeypatch)
    client = _MetaClient()

    async def emit_late():
        await asyncio.sleep(0.3)  # after settle, well inside the grace window
        await client.emit({"late": True})

    task = asyncio.create_task(emit_late())
    await runner.wait_for_late_metadata(client)
    # Snapshot at RETURN time: awaiting the producer first would let a dropped
    # event land afterwards and make the assertion vacuous.
    collected = len(client.turn_acp_meta.get("ns", []))
    await task
    assert collected == 1


@pytest.mark.asyncio
async def test_late_metadata_collects_a_trailing_update(monkeypatch):
    """The bucket is cleared once the response is built, so stragglers are lost."""
    runner = load_runner_without_acp_dependency(monkeypatch)
    client = _MetaClient()

    async def emit_two():
        await asyncio.sleep(0.01)
        await client.emit({"first": True})
        await asyncio.sleep(0.01)
        await client.emit({"trailing": True})

    task = asyncio.create_task(emit_two())
    await runner.wait_for_late_metadata(client)
    collected = len(client.turn_acp_meta.get("ns", []))
    await task
    assert collected == 2


@pytest.mark.asyncio
async def test_late_metadata_does_not_delay_a_turn_that_already_has_metadata(
    monkeypatch,
):
    """Otherwise every prompt pays a fixed delay it does not need."""
    runner = load_runner_without_acp_dependency(monkeypatch)
    started = asyncio.get_event_loop().time()
    await runner.wait_for_late_metadata(_MetaClient({"ns": [{"done": True}]}))
    elapsed = asyncio.get_event_loop().time() - started
    assert elapsed < runner.LATE_UPDATE_GRACE_SECONDS / 2, elapsed


@pytest.mark.asyncio
async def test_prompt_starts_metadata_collection_at_the_prompt_boundary(monkeypatch):
    """Session-start updates must not shorten the first prompt update's grace."""
    runner = load_runner_without_acp_dependency(monkeypatch)

    class PromptClient(_MetaClient):
        def __init__(self):
            super().__init__({"ns": [{"from_session_start": True}]})
            self.visible_reply = ""
            self.message_id = None
            self.tool_calls = {}

        def reset(self):
            self.visible_reply = ""
            self.message_id = None
            self.tool_calls = {}

    class Connection:
        async def prompt(self, **kwargs):
            client.visible_reply = "reply"
            asyncio.create_task(emit_prompt_metadata())
            return types.SimpleNamespace(stop_reason="end_turn")

    client = PromptClient()

    async def emit_prompt_metadata():
        # Longer than the settle window but inside the first-event grace period.
        await asyncio.sleep(0.3)
        await client.emit({"from_prompt": True})

    reply = await runner.prompt(
        client,
        Connection(),
        None,
        "session",
        {"messages": [{"role": "user", "content": "hi"}], "system_prompt": ""},
        is_new=True,
    )

    assert reply == "reply"
    assert client.turn_acp_meta == {"ns": [{"from_prompt": True}]}
