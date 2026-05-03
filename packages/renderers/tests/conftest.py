"""Shared fixtures for renderer tests.

Each (model_name, renderer_name) pair gets a tokenizer + renderer.
The same barrage of tests runs against every pair.
"""

import os

import pytest
from transformers import AutoTokenizer

from renderers import create_renderer

# (HuggingFace model name, renderer name or "auto")
#
# Baseline matrix for render-parity, parse, and per-token-attribution
# tests. Models here are exercised by every shared test in this folder.
# Additional models for narrower tests (e.g. roundtrip) live with their
# own parametrization in the test file.
RENDERER_MODELS = [
    ("Qwen/Qwen3-8B", "auto"),
    ("Qwen/Qwen3.5-9B", "auto"),
    ("Qwen/Qwen3.6-35B-A3B", "auto"),
    ("Qwen/Qwen3-VL-4B-Instruct", "auto"),
    ("zai-org/GLM-5", "auto"),
    ("zai-org/GLM-5.1", "auto"),
    ("zai-org/GLM-4.7-Flash", "auto"),
    ("THUDM/GLM-4.5-Air", "auto"),
    ("MiniMaxAI/MiniMax-M2.5", "auto"),
    ("moonshotai/Kimi-K2-Instruct", "auto"),
    ("moonshotai/Kimi-K2.5", "auto"),
    ("moonshotai/Kimi-K2.6", "auto"),
    ("nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16", "auto"),
    ("nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-BF16", "auto"),
    ("openai/gpt-oss-20b", "gpt_oss"),
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


# Tests that compare the renderer output (or downstream tokens) against
# HF's ``apply_chat_template`` — or that feed plain text through the
# parser expecting a non-empty content. Both fail for gpt-oss because:
#  1. Our GptOssRenderer matches openai-harmony / vLLM, not HF's Jinja
#     (they disagree on a trailing ``\n\n`` and the function-tools
#     layout). Harmony parity is covered separately in
#     ``test_gpt_oss_harmony_parity.py``.
#  2. The harmony parser only emits content from messages bracketed by
#     ``<|start|>...<|message|>...<|end|>`` channel markers; plain text
#     never matches a block, so the test's "Hello there!" probe
#     trivially returns empty content. The parsing-test fixtures aren't
#     designed for harmony format.
_GPT_OSS_HF_PARITY_TEST_FILES = {
    "test_render_ids.py",
    "test_build_helpers.py",
    "test_parse_response.py",
    "test_parse_response_robustness.py",
}


@pytest.fixture(autouse=True)
def _skip_gpt_oss_for_hf_parity_tests(request):
    callspec = getattr(request.node, "callspec", None)
    model_name = callspec.params.get("model_name") if callspec else None
    if model_name != "openai/gpt-oss-20b":
        return
    test_file = os.path.basename(str(request.node.fspath))
    if test_file in _GPT_OSS_HF_PARITY_TEST_FILES:
        pytest.skip(
            f"{model_name}: renderer matches openai-harmony / vLLM, not HF "
            "apply_chat_template — see test_gpt_oss_harmony_parity.py"
        )
