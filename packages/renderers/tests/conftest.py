"""Shared fixtures for renderer tests.

Each (model_name, renderer_name) pair gets a tokenizer + renderer.
The same barrage of tests runs against every pair.
"""

import pytest
from transformers import AutoTokenizer

from renderers import create_renderer

# (HuggingFace model name, renderer name or "auto")
RENDERER_MODELS = [
    ("Qwen/Qwen3-8B", "auto"),
    ("Qwen/Qwen3.5-9B", "auto"),
    ("zai-org/GLM-5", "auto"),
    ("zai-org/GLM-4.7-Flash", "auto"),
    ("THUDM/GLM-4.5-Air", "auto"),
    ("MiniMaxAI/MiniMax-M2.5", "auto"),
    ("PrimeIntellect/INTELLECT-3.1", "auto"),  # auto → DefaultRenderer (BPE merges)
    # ("moonshotai/Kimi-K2.5", "default"),  # TODO: needs custom KimiRenderer
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
