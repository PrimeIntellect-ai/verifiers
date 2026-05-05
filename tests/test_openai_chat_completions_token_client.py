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

    async def post(
        self, path: str, body: dict[str, Any], cast_to: type, **kwargs: Any
    ) -> Any:
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

    async def fake_get_prompt_ids(  # noqa: ANN001
        self, state, prompt_messages, oai_tools, chat_template_kwargs=None
    ):
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

    async def fake_get_prompt_ids(  # noqa: ANN001
        self, state, prompt_messages, oai_tools, chat_template_kwargs=None
    ):
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


# ---------------------------------------------------------------------------
# dynamo_chat_nvext transport (Dynamo bis/dynamo-rl)
# ---------------------------------------------------------------------------

class _StubRenderer:
    """Renderer stand-in for the dynamo_chat_nvext transport tests.

    Returns deterministic ids so we can assert on body shape without pulling
    in a real HuggingFace tokenizer download. ``render_ids`` returns a
    fixed sequence; ``get_stop_token_ids`` returns a marker pair.
    """

    def __init__(self) -> None:
        self.render_calls: list[dict[str, Any]] = []

    def render_ids(
        self,
        messages,
        *,
        tools=None,
        add_generation_prompt: bool = False,
    ) -> list[int]:
        self.render_calls.append(
            {
                "messages": messages,
                "tools": tools,
                "add_generation_prompt": add_generation_prompt,
            }
        )
        # Encode the call shape into ids so tests can disambiguate the two
        # bridge tokenize calls without a real tokenizer.
        return [42, len(messages), int(add_generation_prompt)]

    def get_stop_token_ids(self) -> list[int]:
        return [99, 100]


class _DynamoTestClient(OpenAIChatCompletionsTokenClient):
    """Dynamo-transport TITO client with a stubbed renderer.

    Subclass override is the cleanest way to inject the stub without going
    through ``ClientConfig`` (which would require a real ``api_base_url``
    and ``setup_client`` to construct the AsyncOpenAI). The recording
    client captures the eventual ``self.client.post(...)`` call.
    """

    _stub_renderer: _StubRenderer

    def __init__(self, recording_client) -> None:
        super().__init__(recording_client)
        self._stub_renderer = _StubRenderer()

    @property
    def renderer_transport(self) -> str:  # type: ignore[override]
        return "dynamo_chat_nvext"

    def _get_renderer(self, model: str):  # type: ignore[override]
        return self._stub_renderer


@pytest.mark.asyncio
async def test_local_tokenize_uses_renderer_under_dynamo_transport():
    """Bridge tokenize must NOT hit any HTTP route under dynamo_chat_nvext.

    Goes straight through ``_local_tokenize`` -> ``renderer.render_ids``.
    The recording client would record any errant POST; we assert it sees
    none.
    """
    recording_client = _RecordingClient()
    client = _DynamoTestClient(recording_client)

    ids_full = await client.tokenize(
        messages=[{"role": "user", "content": "u"}],
        tools=None,
        model="test-model",
    )
    ids_base = await client.tokenize(
        messages=[{"role": "user", "content": "u"}],
        tools=None,
        model="test-model",
        extra_kwargs={"add_generation_prompt": False},
    )

    # Both calls hit the renderer, neither hit the wire.
    assert recording_client.calls == []
    assert client._stub_renderer.render_calls[0]["add_generation_prompt"] is True
    assert client._stub_renderer.render_calls[1]["add_generation_prompt"] is False
    # And the stub encodes that into the returned ids' last element.
    assert ids_full[-1] == 1
    assert ids_base[-1] == 0


@pytest.mark.asyncio
async def test_get_native_response_uses_dynamo_chat_nvext_under_transport(
    monkeypatch: pytest.MonkeyPatch,
):
    """Dynamo transport must POST to /chat/completions with nvext.token_data.

    Mirrors test_get_native_response_uses_token_route_when_prompt_ids_available
    but for the new transport.
    """
    recording_client = _RecordingClient()
    client = _DynamoTestClient(recording_client)

    async def fake_get_prompt_ids(self, state, prompt_messages, oai_tools):  # noqa: ANN001
        return [10, 20, 30]

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
        sampling_args={"max_completion_tokens": 16, "temperature": 0.5},
        tools=None,
        state=state,
    )

    assert response["ok"] is True
    assert len(recording_client.calls) == 1
    call = recording_client.calls[0]

    # Wire-shape assertions: route, nvext.token_data, stop_token_ids,
    # placeholder messages, sampling fields promoted.
    assert call["path"] == "/chat/completions"
    body = call["body"]
    assert body["nvext"]["token_data"] == [10, 20, 30]
    assert body["nvext"]["extra_fields"] == ["completion_token_ids"]
    assert body["stop_token_ids"] == [99, 100]
    assert body["messages"] == [{"role": "user", "content": "(token-in mode)"}]
    assert body["max_completion_tokens"] == 16
    assert body["temperature"] == 0.5
    assert body["logprobs"] is True
    assert body["stream"] is False

    # No /chat/completions/tokens, no /tokenize for the dynamo transport.
    assert all(
        c["path"] != "/chat/completions/tokens" and not c["path"].endswith("/tokenize")
        for c in recording_client.calls
    )
