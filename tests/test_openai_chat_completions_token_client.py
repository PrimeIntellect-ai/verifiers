import base64
import json
from typing import Any, cast

import httpx
import pytest

from verifiers.clients.openai_chat_completions_client import OpenAIChatCompletionsClient
from verifiers.clients.openai_chat_completions_token_client import (
    OpenAIChatCompletionsTokenClient,
)
from verifiers.types import State
from verifiers.utils.client_utils import post_chat_completion_with_routed_experts_sidecar


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
        return httpx.Response(
            200,
            json={
                "id": path,
                "object": "chat.completion",
                "created": 1,
                "model": body["model"],
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": "ok"},
                        "finish_reason": "stop",
                    }
                ],
                "ok": True,
                "path": path,
                "body": body,
            },
        )


class _DynamoRoutedExpertsClient(_NoopClient):
    async def post(
        self, path: str, body: dict[str, Any], cast_to: type, **kwargs: Any
    ) -> Any:
        payload = {
            "id": "x",
            "object": "chat.completion",
            "created": 1,
            "model": "test-model",
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": "ok"},
                    "finish_reason": "stop",
                }
            ],
            "nvext": {
                "engine_data": {
                    "completion_token_ids": [10],
                    "routed_experts": {
                        "data": base64.b64encode(b"abc").decode("ascii"),
                        "shape": [3, 1, 1],
                        "start": 0,
                        "dtype": "uint8",
                    },
                }
            },
        }
        return httpx.Response(
            200,
            content=json.dumps(payload, separators=(",", ":")).encode("utf-8"),
        )


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

    assert response.model_extra["ok"] is True
    assert len(recording_client.calls) == 1
    assert recording_client.calls[0]["path"] == "/chat/completions/tokens"
    assert recording_client.calls[0]["body"]["tokens"] == [10, 20]


@pytest.mark.asyncio
async def test_post_dynamo_scrubs_vllm_only_and_forwards_sampling():
    """dynamo wire body: vLLM-only keys scrubbed, standard sampling args
    forwarded, nvext token_data + passthrough preserved."""
    recording_client = _RecordingClient()
    client = OpenAIChatCompletionsTokenClient(recording_client)

    await client._post_dynamo(
        prompt=cast(Any, [{"role": "user", "content": ""}]),
        prompt_ids=[1, 2, 3],
        model="test-model",
        tools=None,
        sampling_args={
            "temperature": 0.5,
            "presence_penalty": 0.2,
            "reasoning_effort": "high",  # arbitrary key: full parity, not an allowlist
            "spaces_between_special_tokens": False,  # vLLM-only — must be scrubbed
            "extra_body": {
                "return_token_ids": True,  # vLLM-only — must be scrubbed
                "nvext": {"extra_fields": ["engine_data"]},
                "cache_salt": "ckpt-1",
            },
        },
        extra_headers=None,
    )

    body = recording_client.calls[0]["body"]
    assert "return_token_ids" not in body
    assert "spaces_between_special_tokens" not in body
    assert body["presence_penalty"] == 0.2
    assert body["temperature"] == 0.5
    assert body["reasoning_effort"] == "high"
    assert body["nvext"]["token_data"] == [1, 2, 3]
    assert body["nvext"]["extra_fields"] == ["engine_data"]
    assert body["cache_salt"] == "ckpt-1"


@pytest.mark.asyncio
async def test_post_dynamo_uses_placeholder_messages():
    recording_client = _RecordingClient()
    client = OpenAIChatCompletionsTokenClient(recording_client)

    await client._post_dynamo(
        prompt=cast(Any, [{"role": "user", "content": "real prompt"}]),
        prompt_ids=[1, 2, 3],
        model="test-model",
        tools=None,
        sampling_args={"extra_body": {"nvext": {"extra_fields": ["engine_data"]}}},
        extra_headers=None,
    )

    assert recording_client.calls[0]["body"]["messages"] == [
        {"role": "user", "content": ""}
    ]


@pytest.mark.asyncio
async def test_sidecar_helper_reattaches_dynamo_engine_routed_experts():
    response = await post_chat_completion_with_routed_experts_sidecar(
        _DynamoRoutedExpertsClient(),
        "/chat/completions",
        body={},
    )

    routed = response.model_extra["nvext"]["engine_data"]["routed_experts"]
    assert isinstance(routed["data"], memoryview)
    assert routed["data"].tobytes() == base64.b64encode(b"abc")


@pytest.mark.asyncio
async def test_graft_engine_data_synthesizes_logprobs_when_content_less():
    """engine_data.completion_logprobs must be grafted even when the choice
    carries a content-less logprobs object (not only when absent)."""
    from openai.types.chat import ChatCompletion

    client = OpenAIChatCompletionsClient(_NoopClient())
    native = ChatCompletion.model_validate(
        {
            "id": "x",
            "object": "chat.completion",
            "created": 1,
            "model": "test-model",
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": "ok"},
                    "finish_reason": "stop",
                    "logprobs": {"content": None},  # present but content-less
                }
            ],
            "nvext": {
                "engine_data": {
                    "completion_token_ids": [10, 11],
                    "prompt_token_ids": [1, 2, 3],
                    "completion_logprobs": [-0.1, -0.2],
                }
            },
        }
    )

    vf_response = await client.from_native_response(native)
    tokens = vf_response.message.tokens
    assert tokens is not None  # would be None before the fix (TITO lost)
    assert tokens.completion_ids == [10, 11]
    assert tokens.prompt_ids == [1, 2, 3]
    assert tokens.completion_logprobs == [-0.1, -0.2]


@pytest.mark.asyncio
async def test_parse_tokens_reads_dynamo_engine_routed_experts():
    from openai.types.chat import ChatCompletion

    client = OpenAIChatCompletionsClient(_NoopClient())
    native = ChatCompletion.model_validate(
        {
            "id": "x",
            "object": "chat.completion",
            "created": 1,
            "model": "test-model",
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": "ok"},
                    "finish_reason": "stop",
                    "logprobs": {
                        "content": [
                            {
                                "token": "ok",
                                "logprob": -0.1,
                                "bytes": [111, 107],
                                "top_logprobs": [],
                            }
                        ]
                    },
                }
            ],
            "nvext": {
                "engine_data": {
                    "completion_token_ids": [10],
                    "prompt_token_ids": [1, 2, 3],
                    "completion_logprobs": [-0.1],
                    "routed_experts": {
                        "data": "QUJD",
                        "shape": [3, 1, 1],
                        "start": 0,
                        "dtype": "uint8",
                    },
                }
            },
        }
    )

    vf_response = await client.from_native_response(native)
    tokens = vf_response.message.tokens

    assert tokens is not None
    assert tokens.routed_experts == {
        "data": "QUJD",
        "shape": [3, 1, 1],
        "start": 0,
        "dtype": "uint8",
    }
