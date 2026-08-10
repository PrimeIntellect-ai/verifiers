"""The ACP wrapper keeps long-lived Prime process streams active."""

import asyncio
import importlib.util
import sys
import types

import pytest

from verifiers.v1.acp import ACP_SOURCE, _packet, _PacketReader


def load_runner_without_acp_dependency(monkeypatch: pytest.MonkeyPatch):
    """Load the standalone runner with only the ACP names this unit path needs."""
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
        "TextContentBlock",
        "ToolCall",
        "ToolCallUpdate",
    ):
        setattr(schema, name, type(name, (), {}))
    monkeypatch.setitem(sys.modules, "acp", acp)
    monkeypatch.setitem(sys.modules, "acp.schema", schema)
    spec = importlib.util.spec_from_loader("test_acp_runner", loader=None)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    exec(compile(ACP_SOURCE, "acp_runner.py", "exec"), module.__dict__)  # noqa: S102
    return module


@pytest.mark.asyncio
async def test_acp_reader_ignores_process_keepalives() -> None:
    async def stream():
        yield _packet({"type": "keepalive"}) + _packet(
            {"ok": True, "reply": "finished"}
        )

    reader = _PacketReader(stream())

    assert await reader.read() == {"ok": True, "reply": "finished"}


@pytest.mark.asyncio
async def test_acp_runner_emits_keepalives_while_session_lives(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runner = load_runner_without_acp_dependency(monkeypatch)

    class Stream:
        def __init__(self) -> None:
            self.data = bytearray()

        def write(self, data: bytes) -> None:
            self.data.extend(data)

        def flush(self) -> None:
            pass

    stream = Stream()
    task = asyncio.create_task(runner.emit_keepalives(stream, interval=0.005))
    await asyncio.sleep(0.03)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    keepalive = _packet({"type": "keepalive"})
    assert len(stream.data) >= len(keepalive) * 3
    assert bytes(stream.data) == keepalive * (len(stream.data) // len(keepalive))
