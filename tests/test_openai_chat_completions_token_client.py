from typing import Any, cast

import pytest

from verifiers.clients.openai_chat_completions_client import OpenAIChatCompletionsClient
from verifiers.clients.openai_chat_completions_token_client import (
    OpenAIChatCompletionsTokenClient,
)
from verifiers.types import State


class _NoopClient:
    base_url = "http://localhost:8000/v1"

    def with_options(self, **kwargs):  # noqa: ANN003
        return self


class _RecordingClient(_NoopClient):
    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    async def post(self, path: str, body: dict[str, Any], cast_to: type) -> Any:
        self.calls.append({"path": path, "body": body, "cast_to": cast_to})
        return {"ok": True, "path": path, "body": body}


class _PromptIdTestClient(OpenAIChatCompletionsTokenClient):
    def __init__(self, full_prompt_ids: list[int]) -> None:
        super().__init__(_NoopClient())
        self._full_prompt_ids = full_prompt_ids

    async def to_native_prompt(self, messages):  # type: ignore[override]
        return cast(Any, messages), {}

    async def tokenize(  # type: ignore[override]
        self,
        messages,
        tools,
        model,
        extra_kwargs: dict = {},
        **kwargs,
    ) -> list[int]:
        if isinstance(messages, str):
            assert messages == "World!"
            return [777]

        if messages == [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "World!"},
        ]:
            assert extra_kwargs == {"add_generation_prompt": False}
            return [1, 777, 999]

        return self._full_prompt_ids


class _NoTokenizeClient(OpenAIChatCompletionsTokenClient):
    def __init__(self) -> None:
        super().__init__(_NoopClient())

    async def to_native_prompt(self, messages):  # type: ignore[override]
        return cast(Any, messages), {}

    async def tokenize(  # type: ignore[override]
        self,
        messages,
        tools,
        model,
        extra_kwargs: dict = {},
        **kwargs,
    ) -> list[int]:
        raise AssertionError("tokenize should not be called without a prefix match")


def _make_step(
    prompt: list[dict[str, str]],
    completion: list[dict[str, str]],
    prompt_ids: list[int],
    completion_ids: list[int],
) -> dict[str, Any]:
    return {
        "prompt": prompt,
        "completion": completion,
        "tokens": {
            "prompt_ids": prompt_ids,
            "completion_ids": completion_ids,
        },
    }


@pytest.mark.asyncio
async def test_get_prompt_ids_uses_largest_message_prefix_match():
    client = _PromptIdTestClient(full_prompt_ids=[1, 2, 3, 4, 999, 5])
    state = cast(
        State,
        {
            "model": "test-model",
            "trajectory": [
                _make_step(
                    prompt=[{"role": "user", "content": "u1"}],
                    completion=[{"role": "assistant", "content": "a1"}],
                    prompt_ids=[1],
                    completion_ids=[2],
                ),
                _make_step(
                    prompt=[
                        {"role": "user", "content": "u1"},
                        {"role": "assistant", "content": "a1"},
                        {"role": "user", "content": "u2"},
                    ],
                    completion=[{"role": "assistant", "content": "a2"}],
                    prompt_ids=[1, 2, 3],
                    completion_ids=[4],
                ),
            ],
        },
    )
    prompt_messages = cast(
        Any,
        [
            {"role": "user", "content": "u1"},
            {"role": "assistant", "content": "a1"},
            {"role": "user", "content": "u2"},
            {"role": "assistant", "content": "a2"},
            {"role": "user", "content": "u3"},
        ],
    )

    prompt_ids = await client.get_prompt_ids(state, prompt_messages, oai_tools=None)

    assert prompt_ids == [1, 2, 3, 4, 999, 5]


@pytest.mark.asyncio
async def test_get_prompt_ids_returns_none_when_no_prefix_match():
    client = _NoTokenizeClient()
    state = cast(
        State,
        {
            "model": "test-model",
            "trajectory": [
                _make_step(
                    prompt=[{"role": "user", "content": "old"}],
                    completion=[{"role": "assistant", "content": "reply"}],
                    prompt_ids=[1],
                    completion_ids=[2],
                )
            ],
        },
    )

    prompt_ids = await client.get_prompt_ids(
        state,
        cast(Any, [{"role": "user", "content": "new"}]),
        oai_tools=None,
    )

    assert prompt_ids is None


@pytest.mark.asyncio
async def test_get_native_response_falls_back_to_super_when_no_prefix_match(
    monkeypatch: pytest.MonkeyPatch,
):
    client = OpenAIChatCompletionsTokenClient(_NoopClient())
    sentinel = {"source": "super"}
    calls: list[dict[str, Any]] = []

    async def fake_get_prompt_ids(self, state, prompt_messages, oai_tools):  # noqa: ANN001
        return None

    async def fake_super_get_native_response(  # noqa: ANN001
        self,
        prompt,
        model,
        sampling_args,
        tools=None,
        **kwargs,
    ):
        calls.append(
            {
                "prompt": prompt,
                "model": model,
                "sampling_args": sampling_args,
                "tools": tools,
            }
        )
        return sentinel

    monkeypatch.setattr(
        OpenAIChatCompletionsTokenClient, "get_prompt_ids", fake_get_prompt_ids
    )
    monkeypatch.setattr(
        OpenAIChatCompletionsClient,
        "get_native_response",
        fake_super_get_native_response,
    )

    state = cast(
        State,
        {
            "model": "test-model",
            "trajectory": [
                _make_step(
                    prompt=[{"role": "user", "content": "u1"}],
                    completion=[{"role": "assistant", "content": "a1"}],
                    prompt_ids=[1],
                    completion_ids=[2],
                )
            ],
        },
    )
    prompt = cast(Any, [{"role": "user", "content": "u2"}])

    response = await client.get_native_response(
        prompt=prompt,
        model="test-model",
        sampling_args={},
        tools=None,
        state=state,
    )

    assert response is sentinel
    assert len(calls) == 1
    assert calls[0]["prompt"] == prompt


@pytest.mark.asyncio
async def test_get_native_response_uses_token_route_when_prompt_ids_available(
    monkeypatch: pytest.MonkeyPatch,
):
    recording_client = _RecordingClient()
    client = OpenAIChatCompletionsTokenClient(recording_client)

    async def fake_get_prompt_ids(self, state, prompt_messages, oai_tools):  # noqa: ANN001
        return [10, 20]

    monkeypatch.setattr(
        OpenAIChatCompletionsTokenClient, "get_prompt_ids", fake_get_prompt_ids
    )

    state = cast(
        State,
        {
            "model": "test-model",
            "trajectory": [
                _make_step(
                    prompt=[{"role": "user", "content": "u1"}],
                    completion=[{"role": "assistant", "content": "a1"}],
                    prompt_ids=[1],
                    completion_ids=[2],
                )
            ],
        },
    )
    prompt = cast(Any, [{"role": "user", "content": "u2"}])

    response = await client.get_native_response(
        prompt=prompt,
        model="test-model",
        sampling_args={},
        tools=None,
        state=state,
    )

    assert response["ok"] is True
    assert len(recording_client.calls) == 1
    assert recording_client.calls[0]["path"] == "/chat/completions/tokens"
    assert recording_client.calls[0]["body"]["tokens"] == [10, 20]
