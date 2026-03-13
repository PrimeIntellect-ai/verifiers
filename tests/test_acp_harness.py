from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

import verifiers as vf
from verifiers.envs.experimental.harnesses import acp_agent
from verifiers.envs.experimental.harnesses.acp_agent import ACPHarness
from verifiers.types import Messages


class _FakeTextContentBlock:
    def __init__(self, text: str):
        self.text = text


class _FakeAgentMessageChunk:
    def __init__(self, content: _FakeTextContentBlock):
        self.content = content


def _install_fake_acp(monkeypatch, events: dict):
    def text_block(text: str):
        events["text_block_input"] = text
        return {"type": "text", "text": text}

    class _FakeConn:
        def __init__(self, collector):
            self.collector = collector

        async def initialize(self, protocol_version):
            events["protocol_version"] = protocol_version

        async def new_session(self, cwd, mcp_servers):
            events["cwd"] = cwd
            events["mcp_servers"] = mcp_servers
            return SimpleNamespace(
                session_id="session-1",
                models=SimpleNamespace(current_model_id="default-model"),
            )

        async def set_session_model(self, model_id, session_id):
            events["set_session_model"] = {
                "model_id": model_id,
                "session_id": session_id,
            }

        async def prompt(self, session_id, prompt):
            events["prompt"] = {"session_id": session_id, "prompt": prompt}
            await self.collector.session_update(
                session_id,
                _FakeAgentMessageChunk(_FakeTextContentBlock("hello ")),
            )
            await self.collector.session_update(
                session_id,
                _FakeAgentMessageChunk(_FakeTextContentBlock("world")),
            )
            return SimpleNamespace(stop_reason=events.get("stop_reason", "end_turn"))

    class _FakeProcessContext:
        def __init__(self, to_client, command, cwd):
            self.to_client = to_client
            self.command = command
            self.cwd = cwd

        async def __aenter__(self):
            events["command"] = list(self.command)
            events["spawn_cwd"] = self.cwd
            collector = self.to_client(object())
            return _FakeConn(collector), object()

        async def __aexit__(self, exc_type, exc, tb):
            return False

    def spawn_agent_process(to_client, *command, cwd):
        return _FakeProcessContext(to_client, command, cwd)

    monkeypatch.setattr(acp_agent, "PROTOCOL_VERSION", "2026-03-01")
    monkeypatch.setattr(acp_agent, "spawn_agent_process", spawn_agent_process)
    monkeypatch.setattr(acp_agent, "text_block", text_block)
    monkeypatch.setattr(acp_agent, "AgentMessageChunk", _FakeAgentMessageChunk)
    monkeypatch.setattr(acp_agent, "TextContentBlock", _FakeTextContentBlock)


def _user_prompt(text: str) -> Messages:
    return [vf.UserMessage(content=text)]


@pytest.mark.asyncio
async def test_acp_harness_streams_text_and_marks_completion(monkeypatch):
    events: dict = {}
    _install_fake_acp(monkeypatch, events)

    harness = ACPHarness(
        command=("opencode", "acp"),
        cwd="/tmp/acp-project",
        session_model_id="openai/gpt-5.4",
        system_prompt="Solve carefully.",
        mcp_servers=[{"name": "search"}],
        timeout_seconds=5.0,
    )
    state = vf.State(trajectory=[])

    response = await harness.get_model_response(
        env=None,
        state=state,
        prompt=_user_prompt("What is 2+2?"),
    )

    assert response.message.content == "hello world"
    assert response.message.finish_reason == "stop"
    assert response.model == "openai/gpt-5.4"
    assert events["command"] == ["opencode", "acp"]
    assert events["cwd"] == "/tmp/acp-project"
    assert events["mcp_servers"] == [{"name": "search"}]
    assert events["text_block_input"] == "Solve carefully.\n\nWhat is 2+2?"
    assert events["prompt"]["prompt"] == [
        {"type": "text", "text": "Solve carefully.\n\nWhat is 2+2?"}
    ]
    assert events["set_session_model"] == {
        "model_id": "openai/gpt-5.4",
        "session_id": "session-1",
    }
    assert response.message.is_truncated is False

    env = SimpleNamespace(
        record_model_response=AsyncMock(),
        normalize_response=lambda response: response,
    )
    state["agent_completed"] = False
    await harness.add_model_response(
        env=env,
        state=state,
        prompt_messages=_user_prompt("What is 2+2?"),
        response=response,
    )
    assert state["agent_completed"] is True
    assert state["agent_exit_code"] == 0
    assert state["agent_stdout"] == "hello world"
    assert state["agent_stderr"] == ""
    env.record_model_response.assert_awaited_once()


@pytest.mark.asyncio
async def test_acp_harness_maps_length_stop_reason(monkeypatch):
    events = {"stop_reason": "max_tokens"}
    _install_fake_acp(monkeypatch, events)

    harness = ACPHarness(timeout_seconds=5.0)
    response = await harness.get_model_response(
        env=None,
        state=vf.State(),
        prompt=_user_prompt("Explain this."),
    )

    assert response.message.finish_reason == "length"
    assert response.message.is_truncated is True


@pytest.mark.asyncio
async def test_acp_harness_tracks_timeout_as_agent_failure(monkeypatch):
    async def fake_run_prompt(**kwargs):
        del kwargs
        raise vf.InfraError("Timed out waiting for ACP agent after 5.0 seconds.")

    harness = ACPHarness(timeout_seconds=5.0)
    monkeypatch.setattr(harness, "run_acp_prompt", fake_run_prompt)

    state = vf.State(trajectory=[])
    with pytest.raises(vf.InfraError, match="Timed out waiting for ACP agent"):
        await harness.get_model_response(
            env=None,
            state=state,
            prompt=_user_prompt("Explain this."),
        )

    assert state["agent_timed_out"] is True
    assert state["agent_exit_code"] == 124
    assert "Timed out waiting for ACP agent" in state["agent_stderr"]
