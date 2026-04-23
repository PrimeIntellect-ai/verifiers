"""Cross-renderer bridge contract tests.

Verifies for every hand-coded renderer that ``bridge_to_next_turn``:

  1. Extends ``prev_prompt_ids + prev_completion_ids`` verbatim.
  2. Refuses assistant-role messages in ``new_messages``.
  3. Synthesises the canonical turn close on truncation iff
     ``synthesize_close_on_truncation`` is True.
  4. On clean stop, the extension is compatible with a fresh render:
     decoding the extension should contain the new-message content and a
     generation-prompt-looking tail.

DefaultRenderer is excluded because it intentionally returns None (it
doesn't know its template's close). That path is exercised by the caller
fallback in ``test_renderer_e2e.py``.
"""

from __future__ import annotations

from functools import lru_cache

import pytest


# (HF model name, renderer name) — one representative per renderer class.
_BRIDGE_MODELS = [
    ("Qwen/Qwen3-8B", "auto"),
    ("Qwen/Qwen3.5-9B", "auto"),
    ("zai-org/GLM-5", "auto"),
    ("zai-org/GLM-5.1", "auto"),
    ("THUDM/GLM-4.5-Air", "auto"),
    ("MiniMaxAI/MiniMax-M2.5", "auto"),
    ("moonshotai/Kimi-K2-Instruct", "auto"),
    ("moonshotai/Kimi-K2.5", "auto"),
    ("nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16", "auto"),
    ("openai/gpt-oss-20b", "gpt_oss"),
]


@lru_cache(maxsize=None)
def _load(model_name: str, renderer_name: str):
    from transformers import AutoTokenizer

    from renderers import create_renderer

    tok = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    return tok, create_renderer(tok, renderer=renderer_name)


def pytest_generate_tests(metafunc):
    if "br_model" in metafunc.fixturenames:
        metafunc.parametrize(
            "br_model,br_renderer_name",
            _BRIDGE_MODELS,
            ids=[m for m, _ in _BRIDGE_MODELS],
        )


@pytest.fixture
def br_tokenizer(br_model, br_renderer_name):
    return _load(br_model, br_renderer_name)[0]


@pytest.fixture
def br_renderer(br_model, br_renderer_name):
    return _load(br_model, br_renderer_name)[1]


def _simulate_prior_turn(renderer):
    """Build a (prev_prompt, prev_completion) pair that a real rollout
    would produce for a one-turn prior with a clean stop.

    Strategy: render ``[system, user]`` with gen_prompt=True to get
    prev_prompt, then render ``[system, user, assistant]`` without
    gen_prompt, and take the diff as prev_completion. We then trim
    prev_completion to the last close token so it matches what vLLM
    actually hands back (vLLM stops at the close token and excludes the
    trailing template scaffolding).
    """
    prior = [
        {"role": "system", "content": "You are helpful."},
        {"role": "user", "content": "Hi."},
    ]
    assistant = [{"role": "assistant", "content": "Hello!"}]

    prev_prompt = renderer.render_ids(prior, add_generation_prompt=True)
    full_with_assistant = renderer.render_ids(
        prior + assistant, add_generation_prompt=False
    )
    prev_completion = list(full_with_assistant[len(prev_prompt):])

    # Trim past any trailing scaffolding the template emits AFTER the
    # close (e.g. chatml's trailing ``\n``). vLLM only returns tokens up
    # to and including the close itself.
    stop_ids = set(renderer.get_stop_token_ids())
    last_close = -1
    for i in range(len(prev_completion) - 1, -1, -1):
        if prev_completion[i] in stop_ids:
            last_close = i
            break
    if last_close >= 0:
        prev_completion = prev_completion[: last_close + 1]

    return prev_prompt, prev_completion


def test_bridge_extends_prev_verbatim_on_clean_stop(br_renderer, br_model):
    prev_prompt, prev_completion = _simulate_prior_turn(br_renderer)
    new_messages = [{"role": "user", "content": "What's 2+2?"}]

    bridged = br_renderer.bridge_to_next_turn(
        prev_prompt, prev_completion, new_messages
    )
    assert bridged is not None, f"{br_model}: bridge returned None on clean stop"

    prev = prev_prompt + prev_completion
    assert bridged[: len(prev)] == prev, (
        f"{br_model}: bridged does NOT extend prev_prompt + prev_completion"
    )
    assert len(bridged) > len(prev), (
        f"{br_model}: bridge did not emit any extension tokens"
    )


def test_bridge_rejects_assistant_in_extension(br_renderer):
    prev_prompt, prev_completion = _simulate_prior_turn(br_renderer)
    assert br_renderer.bridge_to_next_turn(
        prev_prompt,
        prev_completion,
        [{"role": "assistant", "content": "forbidden"}],
    ) is None


def test_bridge_rejects_empty_prev_or_new(br_renderer):
    _, prev_completion = _simulate_prior_turn(br_renderer)
    assert br_renderer.bridge_to_next_turn(
        [], prev_completion, [{"role": "user", "content": "x"}]
    ) is None
    prev_prompt, _ = _simulate_prior_turn(br_renderer)
    assert br_renderer.bridge_to_next_turn(prev_prompt, prev_completion, []) is None


def test_bridge_synthesises_close_on_truncation(br_renderer, br_model):
    prev_prompt, prev_completion = _simulate_prior_turn(br_renderer)
    # Drop the final close token to simulate a max_tokens truncation.
    prev_completion_trunc = prev_completion[:-1] if prev_completion else prev_completion
    if len(prev_completion_trunc) == 0:
        pytest.skip(
            f"{br_model}: simulated prior had no completion tokens — can't truncate"
        )

    bridged = br_renderer.bridge_to_next_turn(
        prev_prompt,
        prev_completion_trunc,
        [{"role": "user", "content": "What's 2+2?"}],
    )
    if not getattr(br_renderer, "synthesize_close_on_truncation", False):
        assert bridged is None, (
            f"{br_model}: renderer opted out of synth-close but bridge returned tokens"
        )
        return

    assert bridged is not None, (
        f"{br_model}: synth-close opt-in but bridge returned None on truncation"
    )
    prev_trunc = prev_prompt + prev_completion_trunc
    assert bridged[: len(prev_trunc)] == prev_trunc, (
        f"{br_model}: truncated-prior bridge did not keep prev tokens verbatim"
    )
    assert len(bridged) > len(prev_trunc), (
        f"{br_model}: synth-close produced no extra tokens"
    )


def test_bridge_extension_includes_new_message_text(br_renderer, br_tokenizer, br_model):
    prev_prompt, prev_completion = _simulate_prior_turn(br_renderer)
    new_messages = [{"role": "user", "content": "HELLO_SENTINEL_XYZ"}]

    bridged = br_renderer.bridge_to_next_turn(
        prev_prompt, prev_completion, new_messages
    )
    assert bridged is not None
    ext = bridged[len(prev_prompt) + len(prev_completion):]
    decoded = br_tokenizer.decode(ext, skip_special_tokens=False)
    assert "HELLO_SENTINEL_XYZ" in decoded, (
        f"{br_model}: new-message content missing from extension; got {decoded!r}"
    )
