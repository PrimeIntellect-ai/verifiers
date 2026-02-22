import builtins

import pytest

from verifiers.clients.human_cli_client import HumanCLIClient
from verifiers.errors import ModelError
from verifiers.types import SystemMessage, Tool, UserMessage


@pytest.mark.asyncio
async def test_human_cli_client_returns_multiline_response(monkeypatch):
    responses = iter(["first line", "second line", ":wq"])
    monkeypatch.setattr(builtins, "input", lambda: next(responses))

    client = HumanCLIClient()
    response = await client.get_response(
        prompt=[UserMessage(content="hello")],
        model="test-model",
        sampling_args={},
    )

    assert response.model == "test-model"
    assert response.message.content == "first line\nsecond line"
    assert response.message.finish_reason == "stop"
    assert response.message.tool_calls is None


@pytest.mark.asyncio
async def test_human_cli_client_reprompts_on_empty_response(monkeypatch):
    responses = iter(["", ":wq", "final answer", ":wq"])
    monkeypatch.setattr(builtins, "input", lambda: next(responses))

    client = HumanCLIClient()
    response = await client.get_response(
        prompt=[UserMessage(content="hello")],
        model="test-model",
        sampling_args={},
    )

    assert response.message.content == "final answer"


@pytest.mark.asyncio
async def test_human_cli_client_rejects_tool_calls(monkeypatch):
    responses = iter(["answer", ":wq"])
    monkeypatch.setattr(builtins, "input", lambda: next(responses))

    client = HumanCLIClient()
    with pytest.raises(ModelError, match="text-only"):
        await client.get_response(
            prompt=[UserMessage(content="hello")],
            model="test-model",
            sampling_args={},
            tools=[
                Tool(
                    name="my_tool",
                    description="test tool",
                    parameters={"type": "object", "properties": {}},
                )
            ],
        )


@pytest.mark.asyncio
async def test_human_cli_client_propagates_keyboard_interrupt(monkeypatch):
    def raise_interrupt():
        raise KeyboardInterrupt

    monkeypatch.setattr(builtins, "input", raise_interrupt)

    client = HumanCLIClient()
    with pytest.raises(KeyboardInterrupt):
        await client.get_response(
            prompt=[UserMessage(content="hello")],
            model="test-model",
            sampling_args={},
        )


@pytest.mark.asyncio
async def test_human_cli_client_renders_prompt_without_crashing(monkeypatch):
    responses = iter(["done", ":wq"])
    monkeypatch.setattr(builtins, "input", lambda: next(responses))

    client = HumanCLIClient()
    response = await client.get_response(
        prompt=[
            SystemMessage(content="You are helpful"),
            UserMessage(content="Solve this"),
        ],
        model="test-model",
        sampling_args={},
    )

    assert response.message.content == "done"
