"""ACP metadata accumulation preserves ordered, namespaced extension events."""

import asyncio
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
    source = inspect.getsource(ACP.run)
    # The public one-shot path has a mandatory trace; this preserves the
    # lifecycle refactor's model-turn invariant rather than silently dropping it.
    assert "calls_before = len(trace.calls)" in source


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


class _PromptMetaClient(_MetaClient):
    """Small lifecycle-aware stand-in for `VerifiersACPClient` prompt tests."""

    def __init__(self):
        super().__init__()
        self.visible_reply = ""
        self.message_id = None
        self.tool_calls = {}
        self._open = False
        self.unattributed_acp_meta = {}
        self._ambiguous = False

    def reset(self):
        self.visible_reply = ""
        self.message_id = None
        self.tool_calls = {}

    def begin_prompt_metadata(self, *, expected):
        if self._ambiguous:
            raise RuntimeError("ACP metadata arrived after its prompt lifecycle closed")
        self.turn_acp_meta = {}
        self._open = expected

    def close_prompt_metadata(self):
        self._open = False

    def require_terminal_metadata(self, *, expected):
        # Timing tests use a deliberately minimal producer fake. Strict producer
        # schema validation is covered against VerifiersACPClient below.
        return None

    async def emit(self, event):
        async with self.output_changed:
            target = self.turn_acp_meta if self._open else self.unattributed_acp_meta
            target.setdefault("ns", []).append(event)
            if not self._open:
                self._ambiguous = True
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
    await runner.wait_for_late_metadata(client, expected=True)
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
    await runner.wait_for_late_metadata(client, expected=True)
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
    await runner.wait_for_late_metadata(
        _MetaClient({"ns": [{"done": True}]}), expected=True
    )
    elapsed = asyncio.get_event_loop().time() - started
    assert elapsed < runner.LATE_UPDATE_GRACE_SECONDS / 2, elapsed


@pytest.mark.asyncio
async def test_prompt_starts_metadata_collection_at_the_prompt_boundary(monkeypatch):
    """Session-start updates must not shorten the first prompt update's grace."""
    runner = load_runner_without_acp_dependency(monkeypatch)

    client = _PromptMetaClient()

    class Connection:
        async def prompt(self, **kwargs):
            client.visible_reply = "reply"
            asyncio.create_task(emit_prompt_metadata())
            return types.SimpleNamespace(stop_reason="end_turn")

    async def emit_prompt_metadata():
        # Longer than the settle window but inside the first-event grace period.
        await asyncio.sleep(0.3)
        await client.emit({"from_prompt": True})

    reply = await runner.prompt(
        client,
        Connection(),
        None,
        "session",
        {
            "messages": [{"role": "user", "content": "hi"}],
            "system_prompt": "",
            "metadata_expected": True,
        },
        is_new=True,
    )

    assert reply == "reply"
    assert client.turn_acp_meta == {"ns": [{"from_prompt": True}]}


@pytest.mark.asyncio
async def test_no_metadata_prompt_has_no_grace_delay(monkeypatch):
    """Unnegotiated ACP preserves the old zero-metadata latency path."""
    runner = load_runner_without_acp_dependency(monkeypatch)
    client = _PromptMetaClient()

    class Connection:
        async def prompt(self, **kwargs):
            client.visible_reply = "reply"
            return types.SimpleNamespace(stop_reason="end_turn")

    started = asyncio.get_running_loop().time()
    assert (
        await runner.prompt(
            client,
            Connection(),
            None,
            "session",
            {"messages": [{"role": "user", "content": "hi"}], "system_prompt": ""},
            is_new=True,
        )
        == "reply"
    )
    assert asyncio.get_running_loop().time() - started < 0.1


@pytest.mark.asyncio
async def test_delayed_event_after_closed_turn_fails_next_turn(monkeypatch):
    """ACP 0.11 has no prompt id, so late metadata cannot be guessed forward."""
    runner = load_runner_without_acp_dependency(monkeypatch)
    client = _PromptMetaClient()

    class Connection:
        async def prompt(self, **kwargs):
            client.visible_reply = "reply"
            return types.SimpleNamespace(stop_reason="end_turn")

    config = {
        "messages": [{"role": "user", "content": "hi"}],
        "system_prompt": "",
        "metadata_expected": True,
    }
    assert await runner.prompt(
        client, Connection(), None, "session", config, is_new=True
    )
    await client.emit({"late": True})
    assert client.turn_acp_meta == {}
    assert client.unattributed_acp_meta == {"ns": [{"late": True}]}
    with pytest.raises(RuntimeError, match="after its prompt lifecycle closed"):
        await runner.prompt(client, Connection(), None, "session", config, is_new=False)


class _SessionLifetimeProducer:
    """ACP producer fake whose notifications outlive a prompt response.

    The transport and producer share one client for the entire session, as a
    real ACP agent does. This deliberately differs from a unit-level `emit`:
    the producer notification is scheduled by the connection after it resolves
    the first prompt response.
    """

    def __init__(self, runner, client):
        self.runner = runner
        self.client = client
        self.response_returned = asyncio.Event()
        self.late_update = None
        self.calls = 0

    async def prompt(self, **kwargs):
        self.calls += 1
        self.client.visible_reply = f"reply {self.calls}"
        if self.calls == 1:
            await self._emit_terminal_envelope()
            self.late_update = asyncio.create_task(self._emit_after_response())
            self.response_returned.set()
        return types.SimpleNamespace(stop_reason="end_turn")

    async def _emit_terminal_envelope(self):
        for event in (
            {
                "promptTurnId": 1,
                "eventSequence": 1,
                "phase": "responseBoundary",
                "outcome": "result",
            },
            {
                "promptTurnId": 1,
                "eventSequence": 2,
                "phase": "terminalQuiescence",
                "outcome": "result",
                "terminalQuiescence": {
                    "outstandingSubagents": 0,
                    "remainingAutonomousContinuations": 0,
                },
            },
        ):
            update = self.runner.SessionInfoUpdate()
            update.field_meta = {"ns": event}
            await self.client.session_update("session", update)

    async def _emit_after_response(self):
        await self.response_returned.wait()
        await asyncio.sleep(self.runner.LATE_UPDATE_GRACE_SECONDS * 2)
        update = self.runner.SessionInfoUpdate()
        update.field_meta = {"ns": {"producer": "after-response"}}
        await self.client.session_update("session", update)


@pytest.mark.asyncio
async def test_session_lifetime_producer_quarantines_post_response_metadata(
    monkeypatch,
):
    """Without #806 correlation/silence, the next prompt fails rather than lies.

    ACP 0.11 cannot attribute the producer event to either turn after prompt
    one closes. Preserve it as infrastructure evidence and fail prompt two
    promptly; do not weaken quarantine while #806 remains unresolved.
    """
    runner = load_runner_without_acp_dependency(monkeypatch)
    monkeypatch.setattr(runner, "LATE_METADATA_SETTLE_SECONDS", 0.001)
    monkeypatch.setattr(runner, "LATE_UPDATE_GRACE_SECONDS", 0.01)
    client = runner.VerifiersACPClient()
    producer = _SessionLifetimeProducer(runner, client)
    config = {
        "messages": [{"role": "user", "content": "hi"}],
        "system_prompt": "",
        "metadata_expected": True,
    }

    assert (
        await runner.prompt(client, producer, None, "session", config, is_new=True)
        == "reply 1"
    )
    assert producer.late_update is not None
    await producer.late_update
    assert client.turn_acp_meta["ns"][0]["phase"] == "responseBoundary"
    assert client.unattributed_acp_meta == {"ns": [{"producer": "after-response"}]}

    with pytest.raises(RuntimeError, match="refusing to attach"):
        await asyncio.wait_for(
            runner.prompt(client, producer, None, "session", config, is_new=False),
            timeout=0.1,
        )
    assert producer.calls == 1


@pytest.mark.asyncio
async def test_metadata_lifecycle_preserves_order_and_quarantines_late_events(
    monkeypatch,
):
    """Two turns cannot share an uncorrelated ACP 0.11 extension event."""
    runner = load_runner_without_acp_dependency(monkeypatch)
    client = runner.VerifiersACPClient()

    def update(event):
        value = runner.SessionInfoUpdate()
        value.field_meta = {"ns": event}
        return value

    client.begin_prompt_metadata(expected=True)
    # A legacy arbitrary event must never become trace-visible merely because it
    # arrived while the lifecycle was open.
    await client.session_update("session", update({"turn": 1, "state": "start"}))
    assert client.turn_acp_meta == {}
    with pytest.raises(RuntimeError, match="invalid promptTurnId"):
        client.require_terminal_metadata(expected=True)
    client.close_prompt_metadata()

    # A post-close producer update is equally quarantined and blocks a new turn.
    await client.session_update("session", update({"turn": 1, "state": "late"}))
    assert len(client.unattributed_acp_meta["ns"]) == 2
    with pytest.raises(RuntimeError, match="refusing to attach"):
        client.begin_prompt_metadata(expected=True)


@pytest.mark.asyncio
async def test_strict_metadata_requires_ordered_same_turn_terminal_envelope(
    monkeypatch,
):
    """Only result/error + responseBoundary + terminalQuiescence may attach."""
    runner = load_runner_without_acp_dependency(monkeypatch)
    client = runner.VerifiersACPClient()

    def update(event):
        value = runner.SessionInfoUpdate()
        value.field_meta = {"ns": event}
        return value

    client.begin_prompt_metadata(expected=True)
    await client.session_update(
        "session",
        update(
            {
                "promptTurnId": 1,
                "eventSequence": 1,
                "phase": "responseBoundary",
                "outcome": "result",
            }
        ),
    )
    await client.session_update(
        "session",
        update(
            {
                "promptTurnId": 1,
                "eventSequence": 2,
                "phase": "terminalQuiescence",
                "outcome": "result",
                "terminalQuiescence": {
                    "outstandingSubagents": 0,
                    "remainingAutonomousContinuations": 0,
                },
            }
        ),
    )
    client.require_terminal_metadata(expected=True)
    assert [event["phase"] for event in client.turn_acp_meta["ns"]] == [
        "responseBoundary",
        "terminalQuiescence",
    ]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "events, error",
    [
        (
            [
                {
                    "promptTurnId": 0,
                    "eventSequence": 1,
                    "phase": "responseBoundary",
                    "outcome": "result",
                }
            ],
            "promptTurnId",
        ),
        (
            [
                {
                    "promptTurnId": 2,
                    "eventSequence": 1,
                    "phase": "responseBoundary",
                    "outcome": "result",
                }
            ],
            "foreign",
        ),
        (
            [
                {
                    "promptTurnId": 1,
                    "eventSequence": 1,
                    "phase": "terminalQuiescence",
                    "outcome": "result",
                }
            ],
            "preceded",
        ),
        (
            [
                {
                    "promptTurnId": 1,
                    "eventSequence": 1,
                    "phase": "responseBoundary",
                    "outcome": "result",
                },
                {
                    "promptTurnId": 1,
                    "eventSequence": 1,
                    "phase": "terminalQuiescence",
                    "outcome": "result",
                    "terminalQuiescence": {
                        "outstandingSubagents": 0,
                        "remainingAutonomousContinuations": 0,
                    },
                },
            ],
            "regressed",
        ),
        (
            [
                {
                    "promptTurnId": 1,
                    "eventSequence": 1,
                    "phase": "responseBoundary",
                    "outcome": "result",
                },
                {
                    "promptTurnId": 1,
                    "eventSequence": 2,
                    "phase": "terminalQuiescence",
                    "outcome": "result",
                    "terminalQuiescence": {
                        "outstandingSubagents": 1,
                        "remainingAutonomousContinuations": 0,
                    },
                },
            ],
            "explicit zero",
        ),
    ],
)
async def test_strict_metadata_rejects_adversarial_envelopes(
    monkeypatch, events, error
):
    runner = load_runner_without_acp_dependency(monkeypatch)
    client = runner.VerifiersACPClient()
    client.begin_prompt_metadata(expected=True)
    for event in events:
        value = runner.SessionInfoUpdate()
        value.field_meta = {"ns": event}
        await client.session_update("session", value)
    with pytest.raises(RuntimeError, match=error):
        client.require_terminal_metadata(expected=True)
    assert client.unattributed_acp_meta


@pytest.mark.asyncio
async def test_end_turn_never_establishes_metadata_correlation(monkeypatch):
    """A transport stop reason is not producer result/error or quiescence evidence."""
    runner = load_runner_without_acp_dependency(monkeypatch)
    client = runner.VerifiersACPClient()
    client.begin_prompt_metadata(expected=True)
    with pytest.raises(RuntimeError, match="lacks a correlated"):
        client.require_terminal_metadata(expected=True)


@pytest.mark.asyncio
async def test_correlated_error_remains_available_for_persistent_stream(monkeypatch):
    """The terminal error envelope is preserved before prompt surfaces failure."""
    runner = load_runner_without_acp_dependency(monkeypatch)
    client = runner.VerifiersACPClient()
    client.begin_prompt_metadata(expected=True)
    for event in (
        {
            "promptTurnId": 1,
            "eventSequence": 1,
            "phase": "responseBoundary",
            "outcome": "error",
        },
        {
            "promptTurnId": 1,
            "eventSequence": 2,
            "phase": "terminalQuiescence",
            "outcome": "error",
            "terminalQuiescence": {
                "outstandingSubagents": 0,
                "remainingAutonomousContinuations": 0,
            },
        },
    ):
        update = runner.SessionInfoUpdate()
        update.field_meta = {"ns": event}
        await client.session_update("session", update)
    with pytest.raises(RuntimeError, match="correlated error"):
        client.require_terminal_metadata(expected=True)
    assert client.turn_acp_meta["ns"][-1]["outcome"] == "error"
