"""TrainClient's exact-token seams: `generate_tokens` (prompt_ids posted
verbatim, sampling via `wire_args`, prompt_logprobs passed through) and
`parse_completion`."""

import asyncio
from types import SimpleNamespace

import pytest

from verifiers.v1.clients.train import ElasticRendererPool, RendererSlot, TrainClient
from verifiers.v1.configs.client import TrainClientConfig
from verifiers.v1.types import SamplingConfig


class _FakeRenderer:
    def get_stop_token_ids(self) -> list[int]:
        return [151645]

    def parse_response(self, completion_ids, tools=None):
        return SimpleNamespace(
            content=f"parsed:{completion_ids}", reasoning_content=None, tool_calls=[]
        )


@pytest.fixture
def client(monkeypatch) -> TrainClient:
    async def fake_grow(self):
        return RendererSlot(_FakeRenderer())

    monkeypatch.setattr(ElasticRendererPool, "grow", fake_grow)
    return TrainClient(TrainClientConfig(base_url="http://engine:8000/v1"))


def test_generate_tokens_posts_prompt_ids_verbatim(client, monkeypatch) -> None:
    import renderers.client as renderers_client

    captured: dict = {}

    async def fake_generate(**kwargs):
        captured.update(kwargs)
        return {
            "request_id": "r1",
            "prompt_ids": kwargs["prompt_ids"],
            "completion_ids": [7, 8],
            "completion_logprobs": [-0.1, -0.2],
            "finish_reason": "stop",
            "prompt_logprobs": [None, {"2": {"logprob": -0.5}}, {"3": {"logprob": -1.0}}],
        }

    monkeypatch.setattr(renderers_client, "generate", fake_generate)

    result = asyncio.run(
        client.generate_tokens(
            "model-x",
            [1, 2, 3],
            SamplingConfig(max_tokens=4, extra_body={"prompt_logprobs": 1, "top_k": 20}),
        )
    )

    assert captured["prompt_ids"] == [1, 2, 3]
    assert captured["messages"] == []
    assert captured["model"] == "model-x"
    # wire_args flattens extra_body under the typed fields
    assert captured["sampling_params"] == {"prompt_logprobs": 1, "top_k": 20, "max_tokens": 4}
    assert result["completion_ids"] == [7, 8]
    assert result["prompt_logprobs"] == [None, {"2": {"logprob": -0.5}}, {"3": {"logprob": -1.0}}]
    assert result["stop_token_ids"] == [151645]


def test_parse_completion_round_trips_through_the_renderer(client) -> None:
    parsed = asyncio.run(client.parse_completion("model-x", [5, 6]))
    assert parsed.content == "parsed:[5, 6]"
