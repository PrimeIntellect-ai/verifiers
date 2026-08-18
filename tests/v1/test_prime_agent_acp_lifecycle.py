"""Focused tests for the Prime Agent ACP lifecycle consumer."""

import asyncio
import importlib.util
import sys
import types
from pathlib import Path

import pytest

from verifiers.v1.acp import _record_lifecycle_status
from verifiers.v1.harnesses.prime_agent.harness import PrimeAgentHarnessConfig
from verifiers.v1.utils.score import read_answer_file_or_last_reply

NAMESPACE = "ai.primeintellect.prime-agent"
CONFIG = {
    "user_contents": ["task"],
    "system_prompt": "",
    "lifecycle_meta_namespace": NAMESPACE,
}


def load_runner(monkeypatch: pytest.MonkeyPatch):
    acp = types.ModuleType("acp")
    acp.PROTOCOL_VERSION = "0.11"
    acp.Client = object

    class RequestError(RuntimeError):
        def __init__(self, message, *, data=None):
            super().__init__(message)
            self.data = data

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
    path = Path(__file__).parents[2] / "verifiers/v1/acp/runner.py"
    spec = importlib.util.spec_from_file_location("prime_agent_acp_runner", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.setattr(module, "LATE_REPLY_GRACE_SECONDS", 0.01)
    return module


def event(sequence, phase="event", turn=1, **values):
    metadata = {
        "promptTurnId": turn,
        "eventSequence": sequence,
        "phase": phase,
        **values,
    }
    if phase == "responseBoundary":
        metadata.setdefault("terminalQuiescenceExpected", True)
    return metadata


def update(runner, metadata, text=None):
    if text is None:
        value = sys.modules["acp.schema"].SessionInfoUpdate()
    else:
        content = runner.TextContentBlock()
        content.text = text
        value = runner.AgentMessageChunk()
        value.content = content
        value.message_id = "message"
    value.field_meta = {NAMESPACE: metadata}
    return value


def tool_update(runner, metadata, status="completed"):
    value = runner.ToolCallUpdate()
    value.tool_call_id = "tool"
    value.status = status
    value.field_meta = {NAMESPACE: metadata}
    return value


async def run_prompt(runner, client, updates, stop_reason="end_turn", config=CONFIG):
    class Connection:
        async def prompt(self, **kwargs):
            for value in updates:
                await client.session_update("session", value)
            return types.SimpleNamespace(stop_reason=stop_reason)

    return await runner.prompt(
        client, Connection(), None, "session", config, is_new=True
    )


@pytest.mark.asyncio
async def test_only_correlated_terminal_quiescence_completes(monkeypatch):
    runner = load_runner(monkeypatch)
    client = runner.VerifiersACPClient()
    updates = [
        update(runner, event(1, compaction={"summary": "not an answer"}), "compact"),
        update(runner, event(2, refinement={"status": "complete"}), "refine"),
        update(runner, event(3, subagents=[{"id": "child"}]), "child"),
        update(runner, event(4, turn=0), "foreign"),
        update(runner, event(5), "final answer"),
        update(runner, event(6, "responseBoundary", outcome="result")),
        update(
            runner,
            event(
                7,
                "terminalQuiescence",
                outcome="result",
                quiescence={
                    "outstandingSubagents": 0,
                    "remainingAutonomousContinuations": 3,
                },
            ),
        ),
    ]

    result = await run_prompt(runner, client, updates, "max_turn_requests")

    assert result["reply"] == "final answer"
    assert result["stop_reason"] == "max_turn_requests"
    assert result["lifecycle"]["phase"] == "terminalQuiescence"


@pytest.mark.asyncio
async def test_outer_timeout_bounds_waiting_text_without_terminal(monkeypatch):
    runner = load_runner(monkeypatch)
    client = runner.VerifiersACPClient()
    updates = [
        update(runner, event(1), "waiting for children"),
        update(runner, event(2, "responseBoundary", outcome="result")),
    ]

    with pytest.raises(asyncio.TimeoutError):
        await asyncio.wait_for(run_prompt(runner, client, updates), timeout=0.03)


@pytest.mark.asyncio
async def test_prompt_waits_for_delayed_terminal_quiescence(monkeypatch):
    runner = load_runner(monkeypatch)
    client = runner.VerifiersACPClient()

    class Connection:
        async def prompt(self, **kwargs):
            await client.session_update(
                "session", update(runner, event(1), "final answer")
            )
            await client.session_update(
                "session",
                update(runner, event(2, "responseBoundary", outcome="result")),
            )

            async def settle():
                await asyncio.sleep(0.03)
                await client.session_update(
                    "session",
                    update(
                        runner,
                        event(
                            3,
                            "terminalQuiescence",
                            outcome="result",
                            quiescence={
                                "outstandingSubagents": 0,
                                "remainingAutonomousContinuations": 0,
                            },
                        ),
                    ),
                )

            asyncio.create_task(settle())
            return types.SimpleNamespace(stop_reason="end_turn")

    result = await runner.prompt(
        client, Connection(), None, "session", CONFIG, is_new=True
    )

    assert result["reply"] == "final answer"
    assert result["lifecycle"]["phase"] == "terminalQuiescence"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "events",
    [
        [
            event(1, "responseBoundary", turn=True, outcome="result"),
            event(
                2,
                "terminalQuiescence",
                turn=True,
                outcome="result",
                quiescence={
                    "outstandingSubagents": False,
                    "remainingAutonomousContinuations": 0,
                },
            ),
        ],
        [
            event(1, "responseBoundary", outcome="result"),
            event(
                2,
                "terminalQuiescence",
                outcome="result",
                quiescence={
                    "outstandingSubagents": False,
                    "remainingAutonomousContinuations": 0,
                },
            ),
        ],
        [
            event(
                1,
                "terminalQuiescence",
                outcome="result",
                quiescence={
                    "outstandingSubagents": 0,
                    "remainingAutonomousContinuations": 0,
                },
            )
        ],
    ],
)
async def test_malformed_or_unordered_terminal_never_completes(monkeypatch, events):
    runner = load_runner(monkeypatch)
    client = runner.VerifiersACPClient()
    updates = [update(runner, value) for value in events]

    with pytest.raises(RuntimeError, match="Prime Agent"):
        await run_prompt(runner, client, updates)


@pytest.mark.asyncio
@pytest.mark.parametrize("sequences", [[1, 1], [2, 1]])
async def test_non_monotonic_lifecycle_sequence_fails(monkeypatch, sequences):
    runner = load_runner(monkeypatch)
    client = runner.VerifiersACPClient()
    updates = [
        update(runner, event(sequences[0]), "answer"),
        update(runner, event(sequences[1], "responseBoundary", outcome="result")),
    ]

    with pytest.raises(RuntimeError, match="eventSequence"):
        await run_prompt(runner, client, updates)


@pytest.mark.asyncio
async def test_terminal_error_is_not_autonomous_completion(monkeypatch):
    runner = load_runner(monkeypatch)
    client = runner.VerifiersACPClient()
    updates = [
        update(runner, event(1), "partial answer"),
        update(runner, event(2, "responseBoundary", outcome="error")),
        update(
            runner,
            event(
                3,
                "terminalQuiescence",
                outcome="error",
                quiescence={
                    "outstandingSubagents": 0,
                    "remainingAutonomousContinuations": 0,
                },
            ),
        ),
    ]

    with pytest.raises(RuntimeError, match="terminal lifecycle error"):
        await run_prompt(runner, client, updates)


def test_prime_agent_lifecycle_is_explicitly_opt_in():
    assert PrimeAgentHarnessConfig().require_terminal_quiescence is False
    assert (
        PrimeAgentHarnessConfig(
            require_terminal_quiescence=True
        ).require_terminal_quiescence
        is True
    )


@pytest.mark.asyncio
async def test_legacy_agent_remains_compatible_without_lifecycle_namespace(monkeypatch):
    runner = load_runner(monkeypatch)
    client = runner.VerifiersACPClient()
    legacy_config = {**CONFIG, "lifecycle_meta_namespace": None}

    result = await run_prompt(
        runner,
        client,
        [update(runner, event(1), "legacy answer")],
        config=legacy_config,
    )

    assert result == {
        "reply": "legacy answer",
        "stop_reason": "end_turn",
        "response_boundary": None,
        "lifecycle": None,
    }
    assert legacy_config["lifecycle_meta_namespace"] is None


@pytest.mark.asyncio
async def test_lifecycle_state_persists_across_prompt_turns(monkeypatch):
    runner = load_runner(monkeypatch)
    client = runner.VerifiersACPClient()
    first = [
        update(runner, event(1), "first answer"),
        update(runner, event(2, "responseBoundary", outcome="result")),
        update(
            runner,
            event(
                3,
                "terminalQuiescence",
                outcome="result",
                quiescence={
                    "outstandingSubagents": 0,
                    "remainingAutonomousContinuations": 0,
                },
            ),
        ),
    ]
    second = [
        update(runner, event(4, turn=2), "second answer"),
        update(
            runner,
            event(5, "responseBoundary", turn=2, outcome="result"),
        ),
        update(
            runner,
            event(
                6,
                "terminalQuiescence",
                turn=2,
                outcome="result",
                quiescence={
                    "outstandingSubagents": 0,
                    "remainingAutonomousContinuations": 0,
                },
            ),
        ),
    ]

    first_result = await run_prompt(runner, client, first)
    second_result = await run_prompt(runner, client, second)

    assert first_result["reply"] == "first answer"
    assert second_result["reply"] == "second answer"
    assert second_result["lifecycle"]["promptTurnId"] == 2


@pytest.mark.asyncio
async def test_request_error_waits_for_correlated_terminal_error(monkeypatch):
    runner = load_runner(monkeypatch)
    client = runner.VerifiersACPClient()

    class Connection:
        async def prompt(self, **kwargs):
            async def settle():
                await asyncio.sleep(0)
                await client.session_update(
                    "session",
                    update(runner, event(1, "responseBoundary", outcome="error")),
                )
                await client.session_update(
                    "session",
                    update(
                        runner,
                        event(
                            2,
                            "terminalQuiescence",
                            outcome="error",
                            quiescence={
                                "outstandingSubagents": 0,
                                "remainingAutonomousContinuations": 0,
                            },
                        ),
                    ),
                )

            asyncio.create_task(settle())
            raise runner.RequestError(
                "request failed", data={"details": "model request failed"}
            )

    with pytest.raises(RuntimeError, match="model request failed"):
        await runner.prompt(client, Connection(), None, "session", CONFIG, is_new=True)
    assert client.terminal_quiescence is not None
    assert client.terminal_quiescence["outcome"] == "error"


@pytest.mark.asyncio
async def test_foreign_tool_update_cannot_complete_current_turn(monkeypatch):
    runner = load_runner(monkeypatch)
    client = runner.VerifiersACPClient()
    updates = [
        tool_update(runner, event(1, turn=0)),
        update(runner, event(2, "responseBoundary", outcome="result")),
        update(
            runner,
            event(
                3,
                "terminalQuiescence",
                outcome="result",
                quiescence={
                    "outstandingSubagents": 0,
                    "remainingAutonomousContinuations": 0,
                },
            ),
        ),
    ]
    config = {**CONFIG, "allow_empty_tool_reply": True}

    class Connection:
        async def prompt(self, **kwargs):
            for value in updates:
                await client.session_update("session", value)
            return types.SimpleNamespace(stop_reason="end_turn")

    with pytest.raises(RuntimeError, match="no visible reply"):
        await runner.prompt(client, Connection(), None, "session", config, is_new=True)


@pytest.mark.asyncio
async def test_precommit_request_error_does_not_wait_for_terminal(monkeypatch):
    runner = load_runner(monkeypatch)
    client = runner.VerifiersACPClient()

    class Connection:
        async def prompt(self, **kwargs):
            async def publish_boundary():
                await asyncio.sleep(0)
                await client.session_update(
                    "session",
                    update(
                        runner,
                        event(
                            1,
                            "responseBoundary",
                            outcome="error",
                            terminalQuiescenceExpected=False,
                        ),
                    ),
                )

            asyncio.create_task(publish_boundary())
            raise runner.RequestError(
                "request failed", data={"details": "admission failed"}
            )

    with pytest.raises(RuntimeError, match="admission failed"):
        await runner.prompt(client, Connection(), None, "session", CONFIG, is_new=True)
    assert client.response_boundary is not None
    assert client.response_boundary["terminalQuiescenceExpected"] is False
    assert client.terminal_quiescence is None


@pytest.mark.asyncio
async def test_malformed_lifecycle_envelope_fails_promptly(monkeypatch):
    runner = load_runner(monkeypatch)
    client = runner.VerifiersACPClient()
    malformed = sys.modules["acp.schema"].SessionInfoUpdate()
    malformed.field_meta = {NAMESPACE: "not an object"}

    with pytest.raises(RuntimeError, match="metadata must be an object"):
        await run_prompt(runner, client, [malformed])


@pytest.mark.asyncio
async def test_ignored_update_kind_wakes_lifecycle_error_waiter(monkeypatch):
    runner = load_runner(monkeypatch)
    client = runner.VerifiersACPClient()

    class IgnoredUpdate:
        def __init__(self):
            self.field_meta = {NAMESPACE: "not an object"}

    class Connection:
        async def prompt(self, **kwargs):
            await client.session_update(
                "session", update(runner, event(1), "partial answer")
            )

            async def publish_malformed_update():
                await asyncio.sleep(0)
                await client.session_update("session", IgnoredUpdate())

            asyncio.create_task(publish_malformed_update())
            return types.SimpleNamespace(stop_reason="end_turn")

    with pytest.raises(RuntimeError, match="metadata must be an object"):
        await asyncio.wait_for(
            runner.prompt(client, Connection(), None, "session", CONFIG, is_new=True),
            timeout=0.03,
        )


def test_lifecycle_status_is_separate_from_benchmark_reward():
    trace = types.SimpleNamespace(
        info={}, rewards={"benchmark": 0.75}, stop_condition=None
    )
    boundary = event(8, "responseBoundary", turn=4, outcome="result")
    terminal = event(
        9,
        "terminalQuiescence",
        turn=4,
        outcome="result",
        quiescence={"outstandingSubagents": 0},
    )
    _record_lifecycle_status(
        trace,
        NAMESPACE,
        {
            "ok": True,
            "reply": "main answer",
            "stop_reason": "max_turn_requests",
            "response_boundary": boundary,
            "lifecycle": terminal,
        },
    )

    assert trace.info["acp_lifecycle"][NAMESPACE][0] == {
        "prompt_turn_id": 4,
        "stop_reason": "max_turn_requests",
        "infrastructure_status": "ok",
        "autonomous_completion": True,
        "response_boundary": boundary,
        "terminal_quiescence": terminal,
    }
    assert trace.info["acp_answer_fallback"] == "main answer"
    assert trace.rewards == {"benchmark": 0.75}
    assert trace.stop_condition is None


@pytest.mark.asyncio
async def test_answer_fallback_cannot_select_child_branch_text():
    class MissingAnswerRuntime:
        async def read(self, path):
            raise FileNotFoundError(path)

    trace = types.SimpleNamespace(
        info={"acp_answer_fallback": "main answer"},
        last_reply="child branch text",
    )
    answer = await read_answer_file_or_last_reply(
        MissingAnswerRuntime(), "/missing/answer", trace
    )

    assert answer == "main answer"
