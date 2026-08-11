"""ACP 0.11 correlation conformance contracts.

The v1 adapter makes ``SessionInfoUpdate.field_meta`` observable through global
``acp_meta`` and prompt-scoped ``turn_acp_meta`` histories. Those histories are
intentionally evidence, not verification: an ACP ``end_turn``, arrival order,
a late-update grace period, or opaque metadata that merely claims a turn ID
cannot establish trusted terminal correlation. Verification remains a
producer/consumer contract using the explicitly scoped producer evidence ABI
below.
"""

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


def opaque_update(runner, event: dict):
    update = runner.SessionInfoUpdate()
    update.field_meta = {"ai.primeintellect.prime-agent": event}
    return update


# This is deliberately the complete, scoped producer ABI used by this
# conformance file.  It does not invent a second producer-ID namespace:
# ``promptTurnId`` and ``eventSequence`` identify an emitted producer record.
# A candidate is trusted as structured producer evidence only when all four
# fields are well-formed.  Whether two candidates establish a completed turn is
# the stricter ``scoreable_terminal`` question below.
VALID_PRODUCER_PHASE_OUTCOMES = frozenset(
    {
        ("responseBoundary", "result"),
        ("responseBoundary", "error"),
        ("terminalQuiescence", "result"),
        ("terminalQuiescence", "error"),
    }
)


def has_verified_producer_correlation(events: list[dict]) -> bool:
    """Return whether every record satisfies the one producer evidence ABI.

    ACP extension metadata remains opaque to the adapter: arrival order,
    ``end_turn``, and partial or merely turn-looking claims cannot establish
    trusted producer correlation.
    """
    return bool(events) and all(
        type(event.get("promptTurnId")) is int
        and event["promptTurnId"] > 0
        and type(event.get("eventSequence")) is int
        and event["eventSequence"] > 0
        and (event.get("phase"), event.get("outcome"))
        in VALID_PRODUCER_PHASE_OUTCOMES
        for event in events
    )


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
async def test_end_turn_tool_only_is_not_trusted_correlation(monkeypatch):
    """A tool-only empty reply has no metadata evidence and no correlation proof."""
    runner = load_runner_without_acp_dependency(monkeypatch)
    client = runner.VerifiersACPClient()

    class Connection:
        async def prompt(self, **kwargs):
            return await completed_tool_only_turn(runner, client)

    reply = await runner.prompt(
        client, Connection(), None, "session", config(allow_empty_tool_reply=True), is_new=True
    )
    assert reply == ""
    assert client.tool_calls == {"tool-1": "completed"}
    assert client.acp_meta == {}
    assert client.turn_acp_meta == {}
    assert not has_verified_producer_correlation([])


@pytest.mark.asyncio
async def test_end_turn_without_completed_tool_has_no_correlation_evidence(monkeypatch):
    """A stop reason cannot make absent observable metadata trustworthy."""
    runner = load_runner_without_acp_dependency(monkeypatch)
    client = runner.VerifiersACPClient()

    class Connection:
        async def prompt(self, **kwargs):
            return types.SimpleNamespace(stop_reason="end_turn")

    with pytest.raises(RuntimeError, match="produced no visible reply"):
        await runner.prompt(client, Connection(), None, "session", config(allow_empty_tool_reply=True), is_new=True)
    assert client.acp_meta == {}
    assert client.turn_acp_meta == {}
    assert not has_verified_producer_correlation([])


@pytest.mark.asyncio
async def test_late_opaque_metadata_is_observable_but_not_trusted(monkeypatch):
    """A delayed prior-turn-looking update is retained, not grace-window joined."""
    runner = load_runner_without_acp_dependency(monkeypatch)
    client = runner.VerifiersACPClient()
    prior_turn_meta = {
        "promptTurnId": "untrusted-P1",
        "phase": "terminalQuiescence",
    }

    class Connection:
        async def prompt(self, **kwargs):
            await client.session_update("session", opaque_update(runner, prior_turn_meta))
            chunk = runner.AgentMessageChunk()
            chunk.content = runner.TextContentBlock()
            chunk.content.text = "P2 reply"
            await client.session_update("session", chunk)
            return types.SimpleNamespace(stop_reason="end_turn")

    assert await runner.prompt(client, Connection(), None, "session", config(), is_new=False) == "P2 reply"
    namespace = "ai.primeintellect.prime-agent"
    assert client.acp_meta[namespace] == [prior_turn_meta]
    assert client.turn_acp_meta[namespace] == [prior_turn_meta]
    assert not has_verified_producer_correlation(client.turn_acp_meta[namespace])


@pytest.mark.asyncio
async def test_arrival_order_is_observable_but_not_trusted_correlation(monkeypatch):
    """Ordered opaque P1/P2-looking metadata is history, never proof by itself."""
    runner = load_runner_without_acp_dependency(monkeypatch)
    client = runner.VerifiersACPClient()
    events = [
        {"eventSequence": 10, "promptTurnId": "P1"},
        {"eventSequence": 11, "promptTurnId": "P2"},
    ]

    for event in events:
        await client.session_update("session", opaque_update(runner, event))

    namespace = "ai.primeintellect.prime-agent"
    assert client.acp_meta[namespace] == events
    assert client.turn_acp_meta[namespace] == events
    assert not has_verified_producer_correlation(client.turn_acp_meta[namespace])


@pytest.mark.asyncio
async def test_opaque_error_metadata_remains_observable_and_error_authoritative(monkeypatch):
    """Metadata storage does not convert a provider error into end_turn success."""
    runner = load_runner_without_acp_dependency(monkeypatch)
    client = runner.VerifiersACPClient()
    error_meta = {"phase": "responseBoundary", "outcome": "error"}

    class Connection:
        async def prompt(self, **kwargs):
            await client.session_update("session", opaque_update(runner, error_meta))
            raise runner.RequestError({"details": "provider rejected request"})

    with pytest.raises(RuntimeError, match="provider rejected request"):
        await runner.prompt(client, Connection(), None, "session", config(allow_empty_tool_reply=True), is_new=True)
    namespace = "ai.primeintellect.prime-agent"
    assert client.acp_meta[namespace] == [error_meta]
    assert client.turn_acp_meta[namespace] == [error_meta]
    assert not has_verified_producer_correlation(client.turn_acp_meta[namespace])


# Conformance/source-level fixtures intentionally model the producer ABI rather
# than importing P2. They lock the scoreable terminal producer-pair contract
# without claiming that the ACP adapter can itself authenticate the producer.
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
    """Conformance/source-level pure gate for P2's completion-cut contract.

    A producer atomically seals a complete zero-quiescence pair *before* it
    queues either notification. Cancellation before that cut is unscoreable;
    cancellation after the cut, including while either notify is gated, cannot
    relabel the durable pair as cancelled. An ACP ``end_turn`` is deliberately
    absent because it is never causal evidence.
    """
    if (
        len(events) != 2
        or (cancelled and not completion_cut_sealed)
        or not has_verified_producer_correlation(events)
    ):
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


def test_producer_abi_rejects_opaque_or_partial_claims():
    """Only all four scoped ABI fields make an event trusted evidence."""
    assert not has_verified_producer_correlation([{"opaque": "claim"}])
    assert not has_verified_producer_correlation(
        [{"promptTurnId": 7, "eventSequence": 41, "phase": "responseBoundary"}]
    )
    assert not has_verified_producer_correlation(
        [{
            "promptTurnId": 7,
            "eventSequence": 41,
            "phase": "opaqueBoundary",
            "outcome": "result",
        }]
    )


def test_p2_terminal_publish_fixture_requires_producer_pair_not_end_turn():
    """A completed terminal is a producer pair, never an ACP stop reason."""
    events = [P2_TERMINAL_FIXTURE["response"], P2_TERMINAL_FIXTURE["terminal"]]
    assert has_verified_producer_correlation(events)
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

    P2's notify-gated production test establishes this linearization; this
    conformance/source-level fixture locks its consumer-facing consequence.
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
