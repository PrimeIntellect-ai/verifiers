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
    assert "await wait_for_late_metadata(client)" in prompt_source
    helper_start = runner.index("async def wait_for_late_metadata")
    helper_end = runner.index("\n\ndef content_blocks", helper_start)
    helper_source = runner[helper_start:helper_end]
    metadata_wait = "client.output_changed.wait_for(lambda: bool(client.turn_acp_meta))"
    assert metadata_wait in helper_source
    # A turn which already has its SessionInfoUpdate must return immediately,
    # rather than waiting out the late-update grace period on every prompt.
    assert "if client.turn_acp_meta:" in helper_source
    assert "return" in helper_source
    error_start = runner.index("            except Exception as error:")
    error_end = runner.index("            write_packet", error_start)
    error_source = runner[error_start:error_end]
    assert 'response["meta"]' not in error_source
    assert "session.client.turn_acp_meta = {}" in error_source


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


@pytest.mark.asyncio
async def test_acp_runner_skips_metadata_grace_when_turn_already_has_metadata(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Metadata received before prompt completion must not add a fixed delay."""
    runner = load_runner_without_acp_dependency(monkeypatch)

    class NoWaitCondition:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            return None

        async def wait_for(self, predicate):
            raise AssertionError("a metadata-bearing turn must not wait")

    class Client:
        def __init__(self) -> None:
            self.visible_reply = "DONE"
            self.tool_calls = {}
            self.turn_acp_meta = {"ai.primeintellect.prime-agent": [{"goal": {}}]}
            self.output_changed = NoWaitCondition()

        def reset(self) -> None:
            # Production reset clears reply/tool state but intentionally preserves
            # metadata until the stream response serializes this turn.
            pass

    class Connection:
        async def prompt(self, **kwargs):
            return types.SimpleNamespace(stop_reason="end_turn")

    reply = await asyncio.wait_for(
        runner.prompt(
            Client(),
            Connection(),
            types.SimpleNamespace(
                prompt_capabilities=types.SimpleNamespace(image=False)
            ),
            "session",
            {"messages": [{"role": "user", "content": "hi"}], "system_prompt": ""},
            is_new=True,
        ),
        timeout=0.1,
    )

    assert reply == "DONE"
