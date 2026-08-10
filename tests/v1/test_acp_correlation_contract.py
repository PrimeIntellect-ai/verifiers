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


# These source-level fixtures intentionally model the producer ABI rather than
# importing P2.  V4 is the negative conformance surface: it records what a
# future producer/consumer integration must prove with notify-gated tests, not
# a claim that the current Python ACP 0.11 adapter has those opaque IDs.
P2_TERMINAL_FIXTURE = {
    "response": {
        "promptTurnId": 7,
        "eventSequence": 41,
        "phase": "responseBoundary",
        "outcome": "result",
    },
    "terminal": {
        "promptTurnId": 7,
        "eventSequence": 42,
        "phase": "terminalQuiescence",
        "outcome": "result",
        "quiescence": {
            "outstandingSubagents": 0,
            "remainingAutonomousContinuations": 0,
        },
    },
}


def scoreable_terminal(
    events: list[dict], *, cancelled: bool, completion_cut_sealed: bool
) -> bool:
    """Pure gate for P2's early completion-cut contract.

    A producer atomically seals a complete zero-quiescence pair *before* it
    queues either notification. Cancellation before that cut is unscoreable;
    cancellation after the cut, including while either notify is gated, cannot
    relabel the durable pair as cancelled. An ACP ``end_turn`` is deliberately
    absent because it is never causal evidence.
    """
    if len(events) != 2 or (cancelled and not completion_cut_sealed):
        return False
    response, terminal = events
    counters = terminal.get("quiescence")
    return (
        response.get("phase") == "responseBoundary"
        and response.get("outcome") == "result"
        and terminal.get("phase") == "terminalQuiescence"
        and terminal.get("outcome") == "result"
        and type(response.get("promptTurnId")) is int
        and response["promptTurnId"] > 0
        and type(terminal.get("promptTurnId")) is int
        and terminal["promptTurnId"] > 0
        and terminal["promptTurnId"] == response["promptTurnId"]
        and type(response.get("eventSequence")) is int
        and response["eventSequence"] > 0
        and type(terminal.get("eventSequence")) is int
        and terminal["eventSequence"] > 0
        and terminal["eventSequence"] > response["eventSequence"]
        and counters
        == {
            "outstandingSubagents": 0,
            "remainingAutonomousContinuations": 0,
        }
    )


def test_p2_terminal_publish_fixture_requires_producer_pair_not_end_turn():
    """A completed terminal is a producer pair, never an ACP stop reason."""
    events = [P2_TERMINAL_FIXTURE["response"], P2_TERMINAL_FIXTURE["terminal"]]
    assert scoreable_terminal(events, cancelled=False, completion_cut_sealed=True)
    # ``end_turn`` is intentionally not accepted by the helper and cannot make
    # a partial/uncorrelated transcript scoreable.
    assert not scoreable_terminal(
        [P2_TERMINAL_FIXTURE["terminal"]],
        cancelled=False,
        completion_cut_sealed=True,
    )


def test_cancel_before_completion_cut_is_unscoreable_and_has_no_pair():
    """A pre-cut cancel must publish no response/terminal completion evidence."""
    assert not scoreable_terminal([], cancelled=True, completion_cut_sealed=False)


def test_cancel_before_completion_cut_rejects_even_a_claimed_full_pair():
    """A consumer must not trust a pair unless producer sealing was irrevocable."""
    events = [P2_TERMINAL_FIXTURE["response"], P2_TERMINAL_FIXTURE["terminal"]]
    assert not scoreable_terminal(events, cancelled=True, completion_cut_sealed=False)


def test_cancel_after_irrevocable_cut_during_response_publish_is_durable():
    """Notify-gated response publication cannot relabel a sealed pair cancelled."""
    events = [P2_TERMINAL_FIXTURE["response"], P2_TERMINAL_FIXTURE["terminal"]]
    assert scoreable_terminal(events, cancelled=True, completion_cut_sealed=True)


def test_cancel_after_irrevocable_cut_during_terminal_publish_is_durable():
    """Terminal-drain cancellation is likewise after the producer completion cut.

    P2's notify-gated production test establishes this linearization; this V4
    negative fixture only locks its consumer-facing consequence.
    """
    events = [P2_TERMINAL_FIXTURE["response"], P2_TERMINAL_FIXTURE["terminal"]]
    assert scoreable_terminal(events, cancelled=True, completion_cut_sealed=True)


def test_terminal_gate_rejects_bool_nonpositive_and_foreign_ids():
    """Opaque IDs/sequences must be strict positive integers on both records."""
    events = [P2_TERMINAL_FIXTURE["response"], P2_TERMINAL_FIXTURE["terminal"]]
    for record_index, field, value in (
        (0, "promptTurnId", True),
        (1, "promptTurnId", False),
        (0, "promptTurnId", 0),
        (1, "promptTurnId", -1),
        (0, "eventSequence", True),
        (1, "eventSequence", False),
        (0, "eventSequence", 0),
        (1, "eventSequence", -1),
    ):
        mutated = [dict(record) for record in events]
        mutated[record_index][field] = value
        assert not scoreable_terminal(
            mutated, cancelled=False, completion_cut_sealed=True
        ), (record_index, field, value)

    foreign_turn = [dict(record) for record in events]
    foreign_turn[1]["promptTurnId"] = 8
    assert not scoreable_terminal(
        foreign_turn, cancelled=False, completion_cut_sealed=True
    )

    nonincreasing = [dict(record) for record in events]
    nonincreasing[1]["eventSequence"] = nonincreasing[0]["eventSequence"]
    assert not scoreable_terminal(
        nonincreasing, cancelled=False, completion_cut_sealed=True
    )


def test_error_or_nonzero_terminal_never_becomes_a_successful_completion_pair():
    """Error and incomplete terminal transcripts stay observable but unscoreable."""
    error_terminal = {
        **P2_TERMINAL_FIXTURE["terminal"],
        "outcome": "error",
    }
    incomplete = {
        **P2_TERMINAL_FIXTURE["terminal"],
        "quiescence": {
            "outstandingSubagents": 1,
            "remainingAutonomousContinuations": 0,
        },
    }
    assert not scoreable_terminal(
        [P2_TERMINAL_FIXTURE["response"], error_terminal],
        cancelled=False,
        completion_cut_sealed=True,
    )
    assert not scoreable_terminal(
        [P2_TERMINAL_FIXTURE["response"], incomplete],
        cancelled=False,
        completion_cut_sealed=True,
    )
