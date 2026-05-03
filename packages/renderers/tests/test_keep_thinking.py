"""Verify ``keep_thinking=True`` retains <think> blocks across user turns.

The default (``keep_thinking=False``) matches each model's chat template:
prior-turn ``reasoning_content`` is stripped so only the most-recent
assistant turn carries thinking. ``keep_thinking=True`` flips that — every
assistant turn's thinking lands in the rendered output.
"""

from __future__ import annotations

import pytest
from transformers import AutoTokenizer

from renderers import (
    GLM5Renderer,
    GLM45Renderer,
    KimiK25Renderer,
    MiniMaxM2Renderer,
    Nemotron3Renderer,
    Qwen3Renderer,
    Qwen35Renderer,
    Qwen36Renderer,
    create_renderer,
)

# Renderer class + canonical HF model id.
KEEP_THINKING_RENDERERS = [
    (Qwen3Renderer, "Qwen/Qwen3-8B"),
    (Qwen35Renderer, "Qwen/Qwen3.5-9B"),
    (Qwen36Renderer, "Qwen/Qwen3.6-35B-A3B"),
    (GLM5Renderer, "zai-org/GLM-5"),
    (GLM45Renderer, "THUDM/GLM-4.5-Air"),
    (MiniMaxM2Renderer, "MiniMaxAI/MiniMax-M2.5"),
    (Nemotron3Renderer, "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16"),
    (KimiK25Renderer, "moonshotai/Kimi-K2.5"),
]


def _multi_turn_messages():
    return [
        {"role": "user", "content": "Hi"},
        {
            "role": "assistant",
            "reasoning_content": "PRIORTHINK_MARKER",
            "content": "Hello!",
        },
        {"role": "user", "content": "Bye"},
        {
            "role": "assistant",
            "reasoning_content": "LATERTHINK_MARKER",
            "content": "Goodbye!",
        },
    ]


@pytest.mark.parametrize(
    "renderer_cls,hf_model",
    KEEP_THINKING_RENDERERS,
    ids=[m for _, m in KEEP_THINKING_RENDERERS],
)
def test_keep_thinking_retains_prior_turn_reasoning(renderer_cls, hf_model):
    tokenizer = AutoTokenizer.from_pretrained(hf_model, trust_remote_code=True)
    msgs = _multi_turn_messages()

    default_renderer = renderer_cls(tokenizer)
    keep_renderer = renderer_cls(tokenizer, keep_thinking=True)

    default_text = tokenizer.decode(default_renderer.render_ids(msgs))
    keep_text = tokenizer.decode(keep_renderer.render_ids(msgs))

    # Both renders carry the latest turn's thinking marker.
    assert "LATERTHINK_MARKER" in keep_text
    # keep_thinking=True must additionally retain the prior turn's marker.
    assert "PRIORTHINK_MARKER" not in default_text
    assert "PRIORTHINK_MARKER" in keep_text


def test_create_renderer_forwards_keep_thinking_to_model_renderer():
    tokenizer = AutoTokenizer.from_pretrained(
        "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16", trust_remote_code=True
    )

    default_renderer = create_renderer(tokenizer, renderer="nemotron3")
    keep_renderer = create_renderer(tokenizer, renderer="nemotron3", keep_thinking=True)

    default_text = tokenizer.decode(default_renderer.render_ids(_multi_turn_messages()))
    keep_text = tokenizer.decode(keep_renderer.render_ids(_multi_turn_messages()))

    assert "PRIORTHINK_MARKER" not in default_text
    assert "PRIORTHINK_MARKER" in keep_text


def test_create_renderer_preserves_keep_thinking_renderer_default():
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-8B", trust_remote_code=True)

    renderer = create_renderer(tokenizer, renderer="qwen3-keep-thinking")

    text = tokenizer.decode(renderer.render_ids(_multi_turn_messages()))
    assert "PRIORTHINK_MARKER" in text


def test_create_renderer_rejects_keep_thinking_for_unsupported_renderer():
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-8B", trust_remote_code=True)

    with pytest.raises(ValueError, match="does not support keep_thinking"):
        create_renderer(tokenizer, renderer="default", keep_thinking=True)
