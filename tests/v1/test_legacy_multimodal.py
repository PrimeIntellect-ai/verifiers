"""Legacy bridge: multimodal is native v1 only."""

import pytest

from verifiers.v1.legacy import rollout_output_to_trace


def _text_only_out() -> dict:
    return {
        "model": "test",
        "reward": 0.0,
        "trajectory": [
            {
                "prompt": [{"role": "user", "content": "hi"}],
                "response": {
                    "message": {"role": "assistant", "content": "hello"},
                    "finish_reason": "stop",
                },
                "tokens": {
                    "prompt_ids": [1],
                    "completion_ids": [2],
                    "completion_logprobs": [0.0],
                },
            }
        ],
    }


def test_legacy_bridge_rejects_image_content_parts():
    out = _text_only_out()
    out["trajectory"][0]["prompt"] = [
        {
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": "file:///x.png"}},
                {"type": "text", "text": "what is this?"},
            ],
        }
    ]
    with pytest.raises(RuntimeError, match="does not support multimodal"):
        rollout_output_to_trace(out, task_idx=0)


def test_legacy_bridge_rejects_multi_modal_data_sidecar():
    out = _text_only_out()
    out["trajectory"][0]["tokens"]["multi_modal_data"] = {"mm_hashes": {"image": ["abc"]}}
    with pytest.raises(RuntimeError, match="does not support multimodal"):
        rollout_output_to_trace(out, task_idx=0)


def test_legacy_bridge_still_maps_text_rollouts():
    trace = rollout_output_to_trace(_text_only_out(), task_idx=0)
    assert len(trace.nodes) >= 2
    assert trace.reward == 0.0
