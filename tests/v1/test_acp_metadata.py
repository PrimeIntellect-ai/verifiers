"""Offline conformance tests for opted-in ACP extension metadata.

ACP 0.11 extension events have no prompt correlation id.  These tests therefore
exercise the deterministic boundary: preserve ordered metadata received while an
explicitly opted-in prompt is open, and quarantine any later event rather than
attaching it to a future turn.
"""

import asyncio
import importlib.util
import sys
import types
from pathlib import Path

import pytest

from verifiers.v1.acp import ACP, _record_acp_meta


class _Trace:
    def __init__(self, info=None):
        self.info = {} if info is None else info


def load_runner_without_acp_dependency(monkeypatch: pytest.MonkeyPatch):
    """Load the runner with only the ACP names exercised by these offline tests."""
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
        "test_acp_metadata_runner",
        Path(__file__).parents[2] / "verifiers/v1/acp/runner.py",
    )
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_metadata_history_is_namespaced_ordered_and_additive() -> None:
    """Trace evidence preserves the producer's event order without flattening."""
    trace = _Trace({"existing": "kept"})
    namespace = "ai.primeintellect.prime-agent"

    _record_acp_meta(
        trace,
        {
            namespace: [{"subagents": [{"id": "child", "status": "running"}]}],
            "other.namespace": [{"value": 1}],
        },
    )
    _record_acp_meta(
        trace,
        {namespace: [{"subagents": [{"id": "child", "status": "completed"}]}]},
    )

    assert trace.info == {
        "existing": "kept",
        "acp_meta": {
            namespace: [
                {"subagents": [{"id": "child", "status": "running"}]},
                {"subagents": [{"id": "child", "status": "completed"}]},
            ],
            "other.namespace": [{"value": 1}],
        },
    }
    # Empty input cannot mutate unrelated task evidence.
    before = dict(trace.info)
    _record_acp_meta(trace, {})
    assert trace.info == before


def test_one_shot_acp_contract_exposes_trace_and_metadata_opt_in() -> None:
    """One-shot and live paths share an explicit, inspectable metadata contract."""
    import inspect

    assert "trace" in inspect.signature(ACP.run).parameters
    assert "trace" in inspect.signature(ACP._run).parameters
    assert "trace=trace" in inspect.getsource(ACP.run)
    assert ACP().metadata_expected is False
    assert ACP(metadata_expected=True).metadata_expected is True


class _Connection:
    def __init__(self, client, *, emit=None):
        self.client = client
        self.emit = emit
        self.calls = 0

    async def prompt(self, **kwargs):
        self.calls += 1
        self.client.visible_reply = f"reply {self.calls}"
        if self.emit is not None:
            self.emit()
        return types.SimpleNamespace(stop_reason="stop")


@pytest.mark.asyncio
async def test_prompt_collects_opted_in_metadata_at_its_own_boundary(monkeypatch):
    """A delayed update inside the bounded prompt grace belongs to that prompt."""
    runner = load_runner_without_acp_dependency(monkeypatch)
    monkeypatch.setattr(runner, "LATE_METADATA_SETTLE_SECONDS", 0.002)
    monkeypatch.setattr(runner, "LATE_UPDATE_GRACE_SECONDS", 0.08)
    client = runner.VerifiersACPClient()

    async def emit():
        await asyncio.sleep(0.02)
        update = runner.SessionInfoUpdate()
        update.field_meta = {"ns": {"turn": 1, "state": "complete"}}
        await client.session_update("session", update)

    connection = _Connection(client, emit=lambda: asyncio.create_task(emit()))
    config = {
        "messages": [{"role": "user", "content": "hi"}],
        "system_prompt": "",
        "metadata_expected": True,
    }

    assert (
        await runner.prompt(client, connection, None, "session", config, is_new=True)
        == "reply 1"
    )
    assert client.turn_acp_meta == {"ns": [{"turn": 1, "state": "complete"}]}
    assert client.acp_meta == client.turn_acp_meta


@pytest.mark.asyncio
async def test_unattributed_post_prompt_metadata_is_quarantined_not_reused(monkeypatch):
    """No event is guessed onto the next prompt without ACP correlation."""
    runner = load_runner_without_acp_dependency(monkeypatch)
    monkeypatch.setattr(runner, "LATE_METADATA_SETTLE_SECONDS", 0.001)
    monkeypatch.setattr(runner, "LATE_UPDATE_GRACE_SECONDS", 0.01)
    client = runner.VerifiersACPClient()
    connection = _Connection(client)
    config = {
        "messages": [{"role": "user", "content": "hi"}],
        "system_prompt": "",
        "metadata_expected": True,
    }

    assert (
        await runner.prompt(client, connection, None, "session", config, is_new=True)
        == "reply 1"
    )
    update = runner.SessionInfoUpdate()
    update.field_meta = {"ns": {"turn": 1, "state": "late"}}
    await client.session_update("session", update)

    assert client.acp_meta == {}
    assert client.unattributed_acp_meta == {"ns": [{"turn": 1, "state": "late"}]}
    with pytest.raises(RuntimeError, match="refusing to attach"):
        await runner.prompt(client, connection, None, "session", config, is_new=False)
    assert connection.calls == 1


@pytest.mark.asyncio
async def test_unopted_prompt_has_no_metadata_wait_or_capture(monkeypatch):
    """Ordinary ACP harnesses retain the historical zero-metadata fast path."""
    runner = load_runner_without_acp_dependency(monkeypatch)
    client = runner.VerifiersACPClient()
    connection = _Connection(client)
    config = {"messages": [{"role": "user", "content": "hi"}], "system_prompt": ""}

    assert await asyncio.wait_for(
        runner.prompt(client, connection, None, "session", config, is_new=True),
        timeout=0.1,
    ) == "reply 1"
    assert client.turn_acp_meta == {}
    assert client.acp_meta == {}
