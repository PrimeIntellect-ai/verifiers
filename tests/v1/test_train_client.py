import base64
import io
from contextlib import asynccontextmanager
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest

from verifiers.v1.clients.train import TrainClient, response_from_generate
from verifiers.v1.dialects.chat import ChatDialect
from verifiers.v1.types import SamplingConfig


def _result(routed_experts):
    return {
        "request_id": "request-1",
        "prompt_ids": [1, 2],
        "completion_ids": [3],
        "completion_logprobs": [-0.1],
        "content": "ok",
        "reasoning_content": None,
        "tool_calls": [],
        "finish_reason": "stop",
        "routed_experts": routed_experts,
        "multi_modal_data": None,
        "prompt_attribution": None,
    }


def _native_payload(array: np.ndarray) -> str:
    buffer = io.BytesIO()
    np.save(buffer, array, allow_pickle=False)
    return base64.b64encode(buffer.getvalue()).decode("ascii")


def test_response_from_generate_normalizes_native_routed_experts():
    array = np.arange(12, dtype=np.uint16).reshape(3, 2, 2)

    response = response_from_generate(
        _result(_native_payload(array)),
        "test-model",
        routed_experts_prompt_start=4,
    )

    assert response.tokens is not None
    payload = response.tokens.routed_experts
    assert payload is not None
    assert payload["shape"] == [3, 2, 2]
    assert payload["dtype"] == "uint16"
    assert payload["start"] == 4
    assert base64.b64decode(payload["data"]) == array.tobytes()


def test_response_from_generate_preserves_legacy_routed_experts():
    legacy = {
        "data": "AQIDBA==",
        "shape": [2, 1, 2],
        "dtype": "uint8",
        "start": 3,
    }

    response = response_from_generate(
        _result(legacy),
        "test-model",
        routed_experts_prompt_start=9,
    )

    assert response.tokens is not None
    assert response.tokens.routed_experts is legacy


@pytest.mark.parametrize(
    ("payload", "message"),
    [
        ("not-base64", "valid non-pickled NumPy array"),
        (_native_payload(np.zeros((2, 2), dtype=np.uint8)), "rank 3"),
        (_native_payload(np.zeros((2, 1, 2), dtype=np.float32)), "uint8 or uint16"),
    ],
)
def test_response_from_generate_rejects_invalid_native_routed_experts(
    payload: str, message: str
):
    with pytest.raises(ValueError, match=message):
        response_from_generate(
            _result(payload),
            "test-model",
            routed_experts_prompt_start=0,
        )


def test_response_from_generate_rejects_pickled_routed_experts():
    array = np.array([[[{"expert": 1}]]], dtype=object)
    buffer = io.BytesIO()
    np.save(buffer, array, allow_pickle=True)

    with pytest.raises(ValueError, match="non-pickled"):
        response_from_generate(
            _result(base64.b64encode(buffer.getvalue()).decode("ascii")),
            "test-model",
            routed_experts_prompt_start=0,
        )


@pytest.mark.asyncio
async def test_train_client_promotes_cache_salt_out_of_sampling_params():
    captured = {}
    client = object.__new__(TrainClient)
    client.client = object()
    client.config = SimpleNamespace(
        renderer_model_name=None,
        renderer=None,
        multiplex=1,
    )

    class FakeRenderer:
        def render(self, *args, **kwargs):
            return SimpleNamespace(
                token_ids=[1, 2],
                multi_modal_data=None,
            )

    class FakeSlot:
        renderer = FakeRenderer()

        async def run(self, function):
            return function()

    class FakePool:
        def __init__(self, *args, **kwargs):
            pass

        @asynccontextmanager
        async def acquire(self):
            yield FakeSlot()

    async def fake_generate(**kwargs):
        captured.update(kwargs)
        return _result(None)

    with (
        patch("verifiers.v1.clients.train.ElasticRendererPool", FakePool),
        patch("renderers.client.generate", side_effect=fake_generate),
    ):
        await client.get_response(
            dialect=ChatDialect(),
            body={
                "model": "test-model",
                "messages": [{"role": "user", "content": "hi"}],
            },
            sampling=SamplingConfig(
                max_tokens=16,
                extra_body={"cache_salt": "policy-1", "top_k": 20},
            ),
        )

    assert captured["cache_salt"] == "policy-1"
    assert captured["sampling_params"] == {"max_tokens": 16, "top_k": 20}
