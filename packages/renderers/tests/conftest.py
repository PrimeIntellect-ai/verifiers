"""Shared fixtures for renderer tests.

Each (model_name, renderer_name) pair gets a tokenizer + renderer.
The same barrage of tests runs against every pair.
"""

import pytest
from transformers import AutoTokenizer

from renderers import create_renderer

# (HuggingFace model name, renderer name or "auto")
#
# This is the baseline matrix for render-parity and parse tests. Models
# here must pass the full test_render_ids barrage against
# apply_chat_template. Additional models for narrower tests (e.g. the
# roundtrip test) live with their own parametrization in the test file.
#
# Not yet here: Kimi K2.5 / K2.6 (template diverges fundamentally from
# K2), GPT-OSS (missing harmony system-preamble implementation), and any
# GLM-5.1 tool-cycle cases. See test_roundtrip.py for the wider matrix.
RENDERER_MODELS = [
    ("Qwen/Qwen3-8B", "auto"),
    ("Qwen/Qwen3.5-9B", "auto"),
    ("Qwen/Qwen3-VL-4B-Instruct", "auto"),
    ("zai-org/GLM-5", "auto"),
    ("zai-org/GLM-5.1", "auto"),
    ("zai-org/GLM-4.7-Flash", "auto"),
    ("THUDM/GLM-4.5-Air", "auto"),
    ("MiniMaxAI/MiniMax-M2.5", "auto"),
    ("moonshotai/Kimi-K2-Instruct", "auto"),
    ("nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16", "auto"),
    ("nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-BF16", "auto"),
    ("Qwen/Qwen2.5-0.5B-Instruct", "default"),
]

_cache: dict[str, tuple] = {}


def _load(model_name: str, renderer_name: str):
    key = f"{model_name}:{renderer_name}"
    if key not in _cache:
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        renderer = create_renderer(tokenizer, renderer=renderer_name)
        _cache[key] = (tokenizer, renderer)
    return _cache[key]


def pytest_generate_tests(metafunc):
    if "model_name" in metafunc.fixturenames:
        metafunc.parametrize(
            "model_name,renderer_name",
            RENDERER_MODELS,
            ids=[m for m, _ in RENDERER_MODELS],
        )


@pytest.fixture
def tokenizer(model_name, renderer_name):
    t, _ = _load(model_name, renderer_name)
    return t


@pytest.fixture
def renderer(model_name, renderer_name):
    _, r = _load(model_name, renderer_name)
    return r
