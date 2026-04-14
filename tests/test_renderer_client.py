from unittest.mock import patch

import pytest

import verifiers as vf
from verifiers.errors import EmptyModelResponseError
from verifiers.clients.renderer_client import RendererClient
from renderers import RendererPool


def test_renderer_client_honors_configured_renderer_name():
    RendererClient._shared_pools.clear()

    client = object.__new__(RendererClient)
    client._renderer = None
    client._pool_size = 1
    client._config = vf.ClientConfig(client_type="renderer", renderer="qwen3_vl")

    with (
        patch(
            "transformers.AutoTokenizer.from_pretrained", return_value=object()
        ) as tokenizer_mock,
        patch(
            "verifiers.clients.renderer_client.create_renderer", return_value="renderer"
        ) as create_renderer_mock,
    ):
        pool = client._get_renderer_or_pool("Qwen/Qwen3-VL-4B-Instruct")

    assert isinstance(pool, RendererPool)
    tokenizer_mock.assert_called_once_with(
        "Qwen/Qwen3-VL-4B-Instruct", trust_remote_code=True
    )
    create_renderer_mock.assert_called_once_with(
        tokenizer_mock.return_value, renderer="qwen3_vl"
    )


def test_renderer_client_uses_renderer_model_name_override():
    RendererClient._shared_pools.clear()

    client = object.__new__(RendererClient)
    client._renderer = None
    client._pool_size = 1
    client._config = vf.ClientConfig(
        client_type="renderer",
        renderer="qwen3_vl",
        renderer_model_name="Qwen/Qwen3-VL-4B-Instruct",
    )

    with (
        patch(
            "transformers.AutoTokenizer.from_pretrained", return_value=object()
        ) as tokenizer_mock,
        patch(
            "verifiers.clients.renderer_client.create_renderer", return_value="renderer"
        ) as create_renderer_mock,
    ):
        pool = client._get_renderer_or_pool("r8-smoke")

    assert isinstance(pool, RendererPool)
    tokenizer_mock.assert_called_once_with(
        "Qwen/Qwen3-VL-4B-Instruct", trust_remote_code=True
    )
    create_renderer_mock.assert_called_once_with(
        tokenizer_mock.return_value, renderer="qwen3_vl"
    )


@pytest.mark.asyncio
async def test_renderer_client_accepts_dict_native_response_with_content():
    client = object.__new__(RendererClient)

    await client.raise_from_native_response({"content": "done"})


@pytest.mark.asyncio
async def test_renderer_client_rejects_empty_dict_native_response():
    client = object.__new__(RendererClient)

    with pytest.raises(EmptyModelResponseError):
        await client.raise_from_native_response({})
