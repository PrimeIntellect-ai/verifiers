import asyncio
from collections import Counter
from unittest.mock import AsyncMock, Mock

import pytest
import renderers
import renderers.client

import verifiers.v1.serve.server as serve_server
from verifiers.v1.clients import EvalClientConfig, TrainClientConfig
from verifiers.v1.clients.train import TrainClient
from verifiers.v1.dialects import ChatDialect
from verifiers.v1.serve.server import EnvServer
from verifiers.v1.types import SamplingConfig


def test_env_server_client_cache_keys(monkeypatch):
    resolve = Mock(side_effect=lambda _: object())
    monkeypatch.setattr(serve_server, "resolve_client", resolve)
    server = object.__new__(EnvServer)
    server._clients = {}

    pinned = TrainClientConfig(renderer_model_name="base-model")
    pinned_clients = [server._client(pinned, f"adapter-{i}") for i in range(8)]
    assert len({id(client) for client in pinned_clients}) == 1

    server._clients.clear()
    unpinned = TrainClientConfig()
    assert server._client(unpinned, "adapter-0") is not server._client(
        unpinned, "adapter-1"
    )

    server._clients.clear()
    eval_config = EvalClientConfig()
    assert server._client(eval_config, "model-0") is not server._client(
        eval_config, "model-1"
    )


async def test_pinned_train_client_routes_512_requests_through_one_pool(monkeypatch):
    server = object.__new__(EnvServer)
    server._clients = {}
    config = TrainClientConfig(
        base_url="http://127.0.0.1:1", renderer_model_name="base-model"
    )
    adapters = [f"adapter-{i}" for i in range(8)]
    contexts = [
        server._context(config, adapter, SamplingConfig())
        for adapter in adapters
        for _ in range(64)
    ]
    shared_client = contexts[0].client
    assert isinstance(shared_client, TrainClient)

    renderer = object()
    create_pool = Mock(return_value=renderer)
    monkeypatch.setattr(renderers, "create_renderer_pool", create_pool)

    generate_mock = AsyncMock(
        return_value={
            "request_id": "response",
            "content": "ok",
            "finish_reason": "stop",
            "prompt_ids": [1] * 160,
            "completion_ids": [2],
            "completion_logprobs": [-0.1],
        }
    )
    monkeypatch.setattr(renderers.client, "generate", generate_mock)
    responses = []
    for start in range(0, len(contexts), 128):
        responses.extend(
            await asyncio.gather(
                *(
                    ctx.client.get_response(
                        ChatDialect(),
                        {"messages": [{"role": "user", "content": "hello"}]},
                        ctx.model,
                        ctx.sampling,
                    )
                    for ctx in contexts[start : start + 128]
                )
            )
        )

    assert create_pool.call_args.args == ("base-model", None)
    assert create_pool.call_args.kwargs == {"size": 1}
    assert create_pool.call_count == 1
    assert Counter(call.kwargs["model"] for call in generate_mock.call_args_list) == {
        adapter: 64 for adapter in adapters
    }
    assert sum(response.usage.total_tokens for response in responses) == 82_432

    close = AsyncMock()
    monkeypatch.setattr(shared_client, "close", close)
    for client in server._clients.values():
        await client.close()
    close.assert_awaited_once()


async def test_renderer_pool_initialization_failure_is_cached(monkeypatch):
    failure = RuntimeError("renderer failed")
    create_pool = Mock(side_effect=failure)
    monkeypatch.setattr(renderers, "create_renderer_pool", create_pool)
    client = TrainClient(AsyncMock(), renderer_model_name="base-model")

    results = await asyncio.gather(
        *(client._renderer_pool(f"adapter-{i}") for i in range(32)),
        return_exceptions=True,
    )

    assert create_pool.call_count == 1
    assert all(str(result) == "renderer failed" for result in results)
    with pytest.raises(RuntimeError, match="renderer failed"):
        await client._renderer_pool("another-adapter")
    assert create_pool.call_count == 1
