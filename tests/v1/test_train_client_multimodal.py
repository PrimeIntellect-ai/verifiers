from pathlib import Path
from urllib.parse import urlparse

import pytest
from renderers.base import MultiModalData, PlaceholderRange, RenderedTokens

import verifiers.v1 as vf
from verifiers.v1 import graph
from verifiers.v1.clients.train import (
    TrainClient,
    _generate_with_image_ref_retry,
)
from verifiers.v1.dialects import ChatDialect
from verifiers.v1.types import TurnTokens
from verifiers.v1.utils import multimodal


DATA_URL = "data:image/png;base64,aGVsbG8="


def test_offload_images_inplace_has_builtin_data_url_fallback(tmp_path):
    body = {
        "messages": [
            {
                "role": "user",
                "content": [{"type": "image_url", "image_url": {"url": DATA_URL}}],
            }
        ]
    }

    stats = multimodal.offload_images_inplace(body, image_dir=tmp_path)

    url = body["messages"][0]["content"][0]["image_url"]["url"]
    parsed = urlparse(url)
    assert stats.images_rewritten == 1
    assert stats.bytes_written == 5
    assert parsed.scheme == "file"
    assert Path(parsed.path).read_bytes() == b"hello"


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
        mm_items={"image": [{"image_grid_thw": [1, 1, 2]}]},
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
    assert captured["generate_kwargs"]["materialize_all_image_refs"] is False


@pytest.mark.asyncio
async def test_generate_retries_missing_mm_cache_by_materializing_image_refs(
    monkeypatch,
):
    import renderers.client as renderer_client

    calls = []

    class MissingCache(Exception):
        body = {"error": {"type": "missing_mm_cache_item"}}

    async def fake_generate(**kwargs):
        calls.append(kwargs["materialize_all_image_refs"])
        if len(calls) == 1:
            raise MissingCache()
        return {"ok": True}

    monkeypatch.setattr(renderer_client, "generate", fake_generate)
    mm = MultiModalData(
        mm_hashes={"image": ["a" * 16]},
        mm_placeholders={"image": [PlaceholderRange(offset=0, length=1)]},
        mm_items={"image": [{"image_grid_thw": [1, 1, 1]}]},
    )

    result = await _generate_with_image_ref_retry(
        client=object(),
        renderer=object(),
        messages=[],
        model="m",
        multi_modal_data=mm,
    )

    assert result == {"ok": True}
    assert calls == [False, True]


@pytest.mark.asyncio
async def test_generate_does_not_retry_missing_cache_for_raw_image_refs(
    monkeypatch,
):
    import renderers.client as renderer_client

    calls = []

    class MissingCache(Exception):
        body = {"error": {"type": "missing_mm_cache_item"}}

    async def fake_generate(**kwargs):
        calls.append(kwargs["materialize_all_image_refs"])
        raise MissingCache()

    monkeypatch.setattr(renderer_client, "generate", fake_generate)
    mm = MultiModalData(
        mm_hashes={"image": ["a" * 16]},
        mm_placeholders={"image": [PlaceholderRange(offset=0, length=1)]},
        mm_items={"image": [{"image_grid_thw": [1, 1, 1], "raw_image_id": "a.png"}]},
    )

    with pytest.raises(MissingCache):
        await _generate_with_image_ref_retry(
            client=object(),
            renderer=object(),
            messages=[],
            model="m",
            multi_modal_data=mm,
        )

    assert calls == [False]
