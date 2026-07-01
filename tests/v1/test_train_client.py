from unittest.mock import call, patch

from verifiers.v1.clients.train import TrainClient


def test_train_client_renderer_pool_cache_splits_chat_template_kwargs() -> None:
    client = TrainClient(openai=object(), pool_size=2)
    pools = [object(), object(), object()]

    with patch("renderers.create_renderer_pool", side_effect=pools) as create_pool:
        first = client._renderer_pool(
            "Qwen/Qwen3-8B",
            chat_template_kwargs={"enable_thinking": False},
        )
        same = client._renderer_pool(
            "Qwen/Qwen3-8B",
            chat_template_kwargs={"enable_thinking": False},
        )
        changed = client._renderer_pool(
            "Qwen/Qwen3-8B",
            chat_template_kwargs={"enable_thinking": True},
        )
        default = client._renderer_pool("Qwen/Qwen3-8B")

    assert first is pools[0]
    assert same is first
    assert changed is pools[1]
    assert default is pools[2]
    assert create_pool.call_args_list == [
        call(
            "Qwen/Qwen3-8B",
            None,
            size=2,
            chat_template_kwargs={"enable_thinking": False},
        ),
        call(
            "Qwen/Qwen3-8B",
            None,
            size=2,
            chat_template_kwargs={"enable_thinking": True},
        ),
        call("Qwen/Qwen3-8B", None, size=2),
    ]
