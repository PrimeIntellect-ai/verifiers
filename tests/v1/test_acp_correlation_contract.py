"""Negative offline conformance contracts for uncorrelated ACP 0.11 updates.

The installed v1 adapter pins agent-client-protocol 0.11.  That surface does not
provide a producer-issued prompt id for ``SessionInfoUpdate`` metadata, so this
suite proves the conservative current behavior: extension metadata is not made
scoreable or attached to another prompt merely because an ACP ``end_turn`` was
observed.  Exact metadata correlation belongs to a producer/consumer contract
with opaque producer IDs, not to stop reasons or a client-side grace window.
"""

import asyncio
import importlib.util
import sys
import types
from pathlib import Path

import pytest


def load_runner_without_acp_dependency(monkeypatch: pytest.MonkeyPatch):
    """Load the standalone runner using local protocol-shaped test doubles only."""

    class RequestError(Exception):
        def __init__(self, data):
            super().__init__("request failed")
            self.data = data

    acp = types.ModuleType("acp")
    acp.PROTOCOL_VERSION = "0.11"
    acp.Client = object
    acp.RequestError = RequestError
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
        "test_acp_correlation_contract_runner",
        Path(__file__).parents[2] / "verifiers/v1/acp/runner.py",
    )
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def config(*, allow_empty_tool_reply: bool = False) -> dict:
    return {
        "messages": [{"role": "user", "content": "do work"}],
        "system_prompt": "",
        "allow_empty_tool_reply": allow_empty_tool_reply,
    }


async def completed_tool_only_turn(runner, client) -> types.SimpleNamespace:
    """Emit a normal ACP tool lifecycle, then an empty ``end_turn`` reply."""
    tool = runner.ToolCall()
    tool.tool_call_id = "tool-1"
    tool.status = "pending"
    await client.session_update("session", tool)
    completed = runner.ToolCallUpdate()
    completed.tool_call_id = "tool-1"
    completed.status = "completed"
    await client.session_update("session", completed)
    return types.SimpleNamespace(stop_reason="end_turn")


@pytest.mark.asyncio
async def test_end_turn_tool_only_is_a_reply_exception_not_metadata_correlation(
    monkeypatch,
):
    """Tool completion permits an empty reply, but creates no scoreable metadata."""
    runner = load_runner_without_acp_dependency(monkeypatch)
    client = runner.VerifiersACPClient()

    class Connection:
        async def prompt(self, **kwargs):
            return await completed_tool_only_turn(runner, client)

    reply = await runner.prompt(
        client,
        Connection(),
        None,
        "session",
        config(allow_empty_tool_reply=True),
        is_new=True,
    )
    assert reply == ""
    assert client.tool_calls == {"tool-1": "completed"}
    # `end_turn` is a transport response.  It supplies neither a turn ID nor a
    # metadata history that a task could score as correlated evidence.
    assert "acp_meta" not in vars(client)
    assert "prompt_turn_id" not in vars(client)


@pytest.mark.asyncio
async def test_end_turn_without_a_completed_tool_is_not_a_clean_metadata_turn(
    monkeypatch,
):
    """An empty end_turn cannot turn missing metadata into a successful result."""
    runner = load_runner_without_acp_dependency(monkeypatch)
    client = runner.VerifiersACPClient()

    class Connection:
        async def prompt(self, **kwargs):
            return types.SimpleNamespace(stop_reason="end_turn")

    with pytest.raises(RuntimeError, match="produced no visible reply"):
        await runner.prompt(
            client,
            Connection(),
            None,
            "session",
            config(allow_empty_tool_reply=True),
            is_new=True,
        )
    assert "acp_meta" not in vars(client)


@pytest.mark.asyncio
async def test_delayed_prior_turn_metadata_cannot_attach_after_next_turn_opens(
    monkeypatch,
):
    """A P1 update delivered while P2 runs remains unscoreable without opaque IDs."""
    runner = load_runner_without_acp_dependency(monkeypatch)
    client = runner.VerifiersACPClient()
    # This object has the shape an extension update could carry, but the pinned
    # ACP 0.11 adapter has no declared metadata update type or correlation key.
    prior_turn_meta = types.SimpleNamespace(
        field_meta={
            "ai.primeintellect.prime-agent": {
                "promptTurnId": "untrusted-P1",
                "phase": "terminalQuiescence",
            }
        }
    )

    class Connection:
        async def prompt(self, **kwargs):
            # This delivery point represents P1's delayed notification after P2
            # opened.  ACP 0.11 gives the adapter no trusted correlation key.
            await client.session_update("session", prior_turn_meta)
            chunk = runner.AgentMessageChunk()
            chunk.content = runner.TextContentBlock()
            chunk.content.text = "P2 reply"
            await client.session_update("session", chunk)
            return types.SimpleNamespace(stop_reason="end_turn")

    assert (
        await runner.prompt(
            client, Connection(), None, "session", config(), is_new=False
        )
        == "P2 reply"
    )
    # The runner does not retain SessionInfoUpdate metadata at all, hence cannot
    # grace-window attach P1 evidence to P2 or make it reachable by scoring.
    assert "acp_meta" not in vars(client)
    assert "turn_acp_meta" not in vars(client)
    assert client.visible_reply == "P2 reply"


@pytest.mark.asyncio
async def test_metadata_arrival_order_cannot_be_promoted_to_correlation(
    monkeypatch,
):
    """Ordered P1/P2-looking fields are opaque until producer IDs are verified."""
    runner = load_runner_without_acp_dependency(monkeypatch)
    client = runner.VerifiersACPClient()
    seen = []
    for sequence, claimed_turn in ((10, "P1"), (11, "P2")):
        seen.append(
            types.SimpleNamespace(
                field_meta={
                    "ai.primeintellect.prime-agent": {
                        "eventSequence": sequence,
                        "promptTurnId": claimed_turn,
                    }
                }
            )
        )

    for update in seen:
        await client.session_update("session", update)

    # Arrival order and strings that look like IDs are not an ACP 0.11
    # correlation contract.  No history/order is exposed to the score surface.
    assert "acp_meta" not in vars(client)
    assert "event_sequence" not in vars(client)
    assert "prompt_turn_id" not in vars(client)


@pytest.mark.asyncio
async def test_request_error_remains_authoritative_when_error_metadata_is_opaque(
    monkeypatch,
):
    """Opaque metadata cannot convert a provider error into end_turn success."""
    runner = load_runner_without_acp_dependency(monkeypatch)
    client = runner.VerifiersACPClient()
    error_meta = types.SimpleNamespace(
        field_meta={
            "ai.primeintellect.prime-agent": {
                "phase": "responseBoundary",
                "outcome": "error",
            }
        }
    )

    class Connection:
        async def prompt(self, **kwargs):
            await client.session_update("session", error_meta)
            raise runner.RequestError({"details": "provider rejected request"})

    with pytest.raises(RuntimeError, match="provider rejected request"):
        await runner.prompt(
            client,
            Connection(),
            None,
            "session",
            config(allow_empty_tool_reply=True),
            is_new=True,
        )
    assert client.visible_reply == ""
    assert "acp_meta" not in vars(client)
