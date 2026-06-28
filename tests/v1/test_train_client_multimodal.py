import pytest
from renderers.base import MultiModalData, PlaceholderRange, RenderedTokens

import verifiers.v1 as vf
from verifiers.v1 import graph
from verifiers.v1.clients.train import (
    TrainClient,
)
from verifiers.v1.dialects import ChatDialect
from verifiers.v1.types import TurnTokens
from verifiers.v1.utils import multimodal


DATA_URL = "data:image/png;base64,aGVsbG8="


def _qwen_item(grid, *, raw_image_id=None):
    item = {
        "kind": "prime_raw_mm_item",
        "version": 1,
        "modality": "image",
        "family": "qwen_vl",
        "layout_fingerprint": "f" * 32,
        "payload": {"image_grid_thw": grid},
    }
    if raw_image_id is not None:
        item["raw_image_id"] = raw_image_id
    return item


def test_offload_images_inplace_rewrites_wire_and_typed_messages(monkeypatch):
    def fake_offload(url, image_dir):
        assert image_dir is None
        if url == DATA_URL:
            return "file:///tmp/run/assets/images/hello.png", 5
        return None

    monkeypatch.setattr(multimodal, "_offload_image_url", fake_offload)

    body = {
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "look"},
                    {"type": "image_url", "image_url": {"url": DATA_URL}},
                ],
            }
        ]
    }
    typed = vf.UserMessage(
        content=[vf.ImageUrlContentPart(image_url=vf.ImageUrlSource(url=DATA_URL))]
    )

    stats = multimodal.offload_images_inplace([body, typed])

    assert stats.images_rewritten == 2
    assert stats.bytes_written == 10
    assert body["messages"][0]["content"][1]["image_url"]["url"] == (
        "file:///tmp/run/assets/images/hello.png"
    )
    assert isinstance(typed.content, list)
    assert typed.content[0].image_url.url == "file:///tmp/run/assets/images/hello.png"


def test_offload_images_inplace_rejects_non_file_image_urls(monkeypatch):
    monkeypatch.setattr(multimodal, "_offload_image_url", lambda *_: None)

    body = {
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {"url": "https://example.com/image.png"},
                    }
                ],
            }
        ]
    }

    with pytest.raises(RuntimeError, match="file:// run image assets"):
        multimodal.offload_images_inplace(body)


def test_prepare_images_inplace_inline_preserves_data_urls():
    body = {
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": DATA_URL}},
                ],
            }
        ]
    }

    stats = multimodal.prepare_images_inplace(body, storage="inline")

    assert stats.images_rewritten == 0
    assert stats.bytes_written == 0
    assert body["messages"][0]["content"][0]["image_url"]["url"] == DATA_URL


def test_prepare_images_inplace_inline_rejects_remote_urls():
    body = {
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {"url": "https://example.com/image.png"},
                    }
                ],
            }
        ]
    }

    with pytest.raises(RuntimeError, match="data:image"):
        multimodal.prepare_images_inplace(body, storage="inline")


@pytest.mark.asyncio
async def test_train_client_bridges_multimodal_prompt_with_previous_sidecar(
    monkeypatch,
):
    import renderers.client as renderer_client

    captured = {}
    image_msg = vf.UserMessage(
        content=[
            vf.ImageUrlContentPart(
                image_url=vf.ImageUrlSource(url="file:///run/assets/images/a.png")
            )
        ]
    )
    previous_mm = MultiModalData(
        mm_hashes={"image": ["a" * 16]},
        mm_placeholders={"image": [PlaceholderRange(offset=1, length=2)]},
        mm_items={"image": [_qwen_item([1, 1, 2])]},
    )
    trace = vf.Trace(task=vf.Task(idx=0, prompt="x"))
    graph.prepare_turn(trace, [image_msg]).commit(
        vf.Response(
            id="a",
            created=0,
            model="t",
            message=vf.AssistantMessage(content="a1"),
            finish_reason="stop",
            tokens=TurnTokens(
                prompt_ids=[10, 11],
                completion_ids=[20],
                message_spans=[(0, 1)],
                multi_modal_data=previous_mm,
            ),
        )
    )
    next_msg = vf.UserMessage(content="next")
    turn = graph.prepare_turn(
        trace, [image_msg, vf.AssistantMessage(content="a1"), next_msg]
    )

    class FakeRenderer:
        is_multimodal = True

        def bridge_to_next_turn(
            self,
            previous_prompt_ids,
            previous_completion_ids,
            new_messages,
            *,
            tools=None,
            previous_multi_modal_data=None,
        ):
            captured["previous_prompt_ids"] = previous_prompt_ids
            captured["previous_completion_ids"] = previous_completion_ids
            captured["new_messages"] = new_messages
            captured["previous_multi_modal_data"] = previous_multi_modal_data
            return RenderedTokens(
                token_ids=[10, 11, 20, 30, 31],
                message_indices=[-1, -1, -1, 0, -1],
                sampled_mask=[False] * 5,
                is_content=[False] * 5,
                message_roles=["user"],
                multi_modal_data=previous_multi_modal_data,
            )

    async def fake_maybe_offload(renderer, fn):
        return fn()

    async def fake_generate(**kwargs):
        captured["generate_kwargs"] = kwargs
        return {
            "request_id": "r",
            "finish_reason": "stop",
            "content": "done",
            "prompt_ids": kwargs["prompt_ids"],
            "completion_ids": [99],
            "completion_logprobs": [-0.5],
            "prompt_attribution": kwargs["prompt_attribution"],
            "multi_modal_data": kwargs["multi_modal_data"],
        }

    monkeypatch.setattr(renderer_client, "_maybe_offload", fake_maybe_offload)
    monkeypatch.setattr(renderer_client, "generate", fake_generate)

    client = TrainClient(openai=object())
    client._pool = FakeRenderer()
    response = await client.get_response(
        ChatDialect(),
        {"messages": []},
        "model",
        vf.SamplingConfig(max_tokens=1),
        session_id="trace",
        turn=turn,
    )

    assert response.message.content == "done"
    assert captured["previous_prompt_ids"] == [10, 11]
    assert captured["previous_completion_ids"] == [20]
    bridged_mm = captured["previous_multi_modal_data"]
    assert bridged_mm.mm_hashes == previous_mm.mm_hashes
    assert bridged_mm.mm_placeholders["image"][0].length == 2
    assert captured["generate_kwargs"]["multi_modal_data"] is bridged_mm
