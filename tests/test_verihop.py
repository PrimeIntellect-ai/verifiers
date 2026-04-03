"""Tests for VeriHop synthesizer, rubric helpers, and add_image."""

import time
from io import BytesIO

import pytest
from PIL import Image

from verifiers.messages import add_image

from verihop.rubrics import VeriHopRubric
from verihop.synthesizer import parse_grounding_boxes, parse_hop_answer, synthesize

_TIMING = {
    "start_time": time.time(),
    "generation_ms": 0.0,
    "scoring_ms": 0.0,
    "total_ms": 0.0,
}


def test_synthesize_row_shape():
    ds = synthesize(num_samples=2, seed=0)
    assert len(ds) == 2
    row = ds[0]
    assert "prompt" in row and "answer" in row and "info" in row
    vh = row["info"]["verihop"]
    assert len(vh["hops"]) == 3
    assert vh["hops"][-1]["ground_truth"] == row["answer"]


def test_parse_hop_answer():
    t = "Reasoning <hop_answer> 42 </hop_answer> tail"
    assert parse_hop_answer(t) == "42"


def test_parse_grounding_boxes():
    t = '<grounding bbox="0, 0, 10, 20" desc="x"/> other <grounding bbox="1,2,3,4"/>'
    boxes = parse_grounding_boxes(t)
    assert len(boxes) == 2
    assert boxes[0] == (0.0, 0.0, 10.0, 20.0)


@pytest.mark.asyncio
async def test_verihop_rubric_outcome():
    rubric = VeriHopRubric(process_weight=0.0, outcome_weight=1.0)
    completion = [
        {"role": "assistant", "content": "Final \\boxed{7}"},
    ]
    state = {
        "prompt": [],
        "completion": completion,
        "answer": "7",
        "task": "verihop",
        "info": {"verihop": {"hops": []}},
        "verihop_collected": [],
        "timing": dict(_TIMING),
    }
    await rubric.score_rollout(state)
    assert state["reward"] == 1.0


@pytest.mark.asyncio
async def test_verihop_rubric_process():
    rubric = VeriHopRubric(process_weight=1.0, outcome_weight=0.0, min_grounding_iou=0.0)
    hops = [
        {"ground_truth": "2", "grounding_norm": (0.0, 0.0, 100.0, 100.0)},
        {"ground_truth": "3", "grounding_norm": None},
    ]
    completion = [
        {
            "role": "assistant",
            "content": (
                '<grounding bbox="0,0,50,50" desc="a"/> '
                "<hop_answer>2</hop_answer>"
            ),
        },
        {"role": "assistant", "content": "<hop_answer>3</hop_answer>"},
    ]
    state = {
        "prompt": [],
        "completion": completion,
        "answer": "5",
        "task": "verihop",
        "info": {"verihop": {"hops": hops}},
        "verihop_collected": ["2", "3"],
        "timing": dict(_TIMING),
    }
    await rubric.score_rollout(state)
    assert state["reward"] > 0.0


def test_add_image_bytes():
    img = Image.new("RGB", (8, 8), color=(255, 0, 0))
    buf = BytesIO()
    img.save(buf, format="PNG")
    raw = buf.getvalue()
    msg: dict = {"role": "user", "content": "hi"}
    add_image(msg, raw)
    assert isinstance(msg["content"], list)
    part = msg["content"][-1]
    assert part["type"] == "image_url"
    assert part["image_url"]["url"].startswith("data:image/png;base64,")
