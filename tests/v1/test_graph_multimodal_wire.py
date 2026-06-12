"""The multimodal sidecar must survive the env-server → orchestrator wire.

`MessageNode.multi_modal_data` is the only carrier of the renderer's pixel tensors. It rides
the wire as base64 (pydantic can't JSON numpy), so a full `to_wire()` → JSON → `model_validate`
round-trip must preserve every image byte-for-byte; otherwise the trainer forwards image-pad
tokens with no pixels.
"""

import json

import numpy as np
import verifiers.v1 as vf
from renderers.base import MultiModalData, PlaceholderRange


def _node(parent, message, mmd=None, **kw):
    return vf.MessageNode(parent=parent, message=message, multi_modal_data=mmd, **kw)


def _mmd(seed: int) -> MultiModalData:
    rng = np.random.default_rng(seed)
    pixel_values = rng.standard_normal((4, 1176)).astype(np.float32)
    image_grid_thw = np.array([[1, 2, 2]], dtype=np.int64)
    return MultiModalData(
        mm_hashes={"image": [f"hash-{seed}"]},
        mm_placeholders={"image": [PlaceholderRange(offset=3 + seed, length=4)]},
        mm_items={
            "image": [{"pixel_values": pixel_values, "image_grid_thw": image_grid_thw}]
        },
    )


def _build_trace() -> vf.Trace:
    task = vf.Task(idx=0, instruction="describe the images")
    trace = vf.Trace(task=task)
    trace.nodes = [
        _node(
            None,
            vf.UserMessage(content="look at this"),
            _mmd(1),
            token_ids=[1, 2],
            mask=[False, False],
        ),
        _node(
            0,
            vf.UserMessage(content="and this"),
            _mmd(2),
            token_ids=[3, 4],
            mask=[False, False],
        ),
        _node(
            1,
            vf.AssistantMessage(content="two cats"),
            None,
            sampled=True,
            token_ids=[5, 6],
            mask=[True, True],
            logprobs=[-0.1, -0.2],
        ),
    ]
    return trace


def test_multi_modal_data_survives_full_json_wire_round_trip():
    trace = _build_trace()

    # Mirror the env server exactly: to_wire() → msgpack(json) → model_validate on the caller.
    restored = vf.Trace.model_validate(json.loads(json.dumps(trace.to_wire())))

    assert len(restored.nodes) == len(trace.nodes)
    for original, node in zip(trace.nodes, restored.nodes):
        before, after = original.multi_modal_data, node.multi_modal_data
        if before is None:
            assert after is None
            continue
        assert after is not None
        assert after.mm_hashes == before.mm_hashes
        assert after.mm_placeholders == before.mm_placeholders
        assert after.mm_items.keys() == before.mm_items.keys()
        for modality, items in before.mm_items.items():
            restored_items = after.mm_items[modality]
            assert len(restored_items) == len(items)
            for orig_item, new_item in zip(items, restored_items):
                assert orig_item.keys() == new_item.keys()
                for key, val in orig_item.items():
                    assert np.array_equal(new_item[key], val)
                    assert new_item[key].dtype == val.dtype
                    assert new_item[key].shape == val.shape

    # The branch view (what training reads) must now see the concatenated images.
    branch = restored.branches[-1]
    assert branch.multi_modal_data is not None
    assert len(branch.multi_modal_data.mm_items["image"]) == 2


def test_multi_modal_data_none_round_trips_to_none():
    task = vf.Task(idx=0, instruction="text only")
    trace = vf.Trace(task=task)
    trace.nodes = [
        _node(None, vf.UserMessage(content="hi"), None, token_ids=[1], mask=[False])
    ]

    restored = vf.Trace.model_validate(json.loads(json.dumps(trace.to_wire())))

    assert restored.nodes[0].multi_modal_data is None
    assert restored.branches[-1].multi_modal_data is None
