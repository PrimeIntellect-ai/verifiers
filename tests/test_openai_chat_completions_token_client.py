from typing import Any, cast

import httpx
import msgpack
import pytest
from openai.types.chat import ChatCompletion

from verifiers.clients.openai_chat_completions_client import (
    OpenAIChatCompletionsClient,
    _parse_chat_completion_with_routed_experts_sidecar,
)
from verifiers.clients.openai_chat_completions_token_client import (
    OpenAIChatCompletionsTokenClient,
)
from verifiers.types import State
from verifiers.utils.serve_utils import msgpack_encoder


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
        if cast_to is httpx.Response:
            return httpx.Response(200, json=_chat_completion_payload())
        return {"ok": True, "path": path, "body": body}


def _chat_completion_payload(routed_data: str | None = None) -> dict[str, Any]:
    choice: dict[str, Any] = {
        "index": 0,
        "message": {"role": "assistant", "content": "ok"},
        "finish_reason": "stop",
        "logprobs": {
            "content": [
                {
                    "token": "ok",
                    "bytes": None,
                    "logprob": -0.125,
                    "top_logprobs": [],
                }
            ]
        },
        "token_ids": [30],
    }
    if routed_data is not None:
        choice["routed_experts"] = {
            "data": routed_data,
            "shape": [3, 1, 1],
        }
    return {
        "id": "chatcmpl-test",
        "object": "chat.completion",
        "created": 1,
        "model": "test-model",
        "prompt_token_ids": [10, 20],
        "choices": [choice],
        "usage": {
            "prompt_tokens": 2,
            "completion_tokens": 1,
            "total_tokens": 3,
        },
    }


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

    assert isinstance(response, ChatCompletion)
    assert response.choices[0].message.content == "ok"
    assert len(recording_client.calls) == 1
    assert recording_client.calls[0]["path"] == "/chat/completions/tokens"
    assert recording_client.calls[0]["cast_to"] is httpx.Response
    assert recording_client.calls[0]["body"]["tokens"] == [10, 20]


def test_parse_chat_completion_strips_routed_experts_data_sidecar():
    routed_data = "AQIDBAUG"
    raw = httpx.Response(
        200,
        json=_chat_completion_payload(routed_data=routed_data),
    ).content

    response = _parse_chat_completion_with_routed_experts_sidecar(raw)

    routed_experts = response.choices[0].model_extra["routed_experts"]
    assert routed_experts["shape"] == [3, 1, 1]
    assert isinstance(routed_experts["data"], memoryview)
    assert routed_experts["data"].obj is raw
    assert routed_experts["data"].tobytes() == routed_data.encode()


@pytest.mark.asyncio
async def test_from_native_response_preserves_routed_experts_sidecar():
    routed_data = "AQIDBAUG"
    raw = httpx.Response(
        200,
        json=_chat_completion_payload(routed_data=routed_data),
    ).content
    native = _parse_chat_completion_with_routed_experts_sidecar(raw)
    client = OpenAIChatCompletionsClient(_NoopClient())

    response = await client.from_native_response(native)

    assert response.message.tokens is not None
    routed_experts = response.message.tokens.routed_experts
    assert routed_experts is not None
    assert routed_experts["shape"] == [3, 1, 1]
    assert isinstance(routed_experts["data"], memoryview)
    assert routed_experts["data"].tobytes() == routed_data.encode()

    packed = msgpack.packb(
        response.model_dump(mode="python", warnings=False),
        default=msgpack_encoder,
        use_bin_type=True,
    )
    unpacked = msgpack.unpackb(packed, raw=False)
    unpacked_routed = unpacked["message"]["tokens"]["routed_experts"]
    assert unpacked_routed["data"] == routed_data.encode()
    assert unpacked_routed["shape"] == [3, 1, 1]
