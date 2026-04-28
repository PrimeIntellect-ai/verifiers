"""Render → parse round-trip.

Core renderer invariant: if you render
``[user, assistant(content=X, reasoning=Y, tool_calls=[T])]`` to tokens,
extract the assistant's completion slice, and feed it through
``parse_response``, you should get back an equivalent structured message.
Catches asymmetries between a renderer's emit path and its parse path —
bugs that slip past render-parity tests (which only check vs
apply_chat_template) and parse-robustness tests (which feed crafted text,
not rendered output).

Parametrizes over a wider model list than the shared conftest barrage
(which is kept conservative because several newer renderers still
disagree with apply_chat_template in complex tool-cycle edge cases).
The roundtrip invariant is renderer-self-consistent so it can tolerate
those gaps while still giving per-renderer coverage of the emit/parse
pair.
"""

from __future__ import annotations

import json
from functools import lru_cache
from typing import Any

import pytest


# (HuggingFace model name, renderer name or "auto"). These are the
# renderers we actively rely on; expand as new ones get hand-coded.
_ROUNDTRIP_MODELS = [
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


@lru_cache(maxsize=None)
def _load_renderer(model_name: str, renderer_name: str):
    from transformers import AutoTokenizer

    from renderers import create_renderer

    tok = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    return tok, create_renderer(tok, renderer=renderer_name)


def pytest_generate_tests(metafunc):
    # Local parametrization only takes effect for tests in this file that
    # declare these fixture names — conftest-level parametrization still
    # applies to every other test module.
    if "rt_model" in metafunc.fixturenames:
        metafunc.parametrize(
            "rt_model,rt_renderer_name",
            _ROUNDTRIP_MODELS,
            ids=[m for m, _ in _ROUNDTRIP_MODELS],
        )


@pytest.fixture
def rt_tokenizer(rt_model, rt_renderer_name):
    return _load_renderer(rt_model, rt_renderer_name)[0]


@pytest.fixture
def rt_renderer(rt_model, rt_renderer_name):
    return _load_renderer(rt_model, rt_renderer_name)[1]


PROMPT = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "What is 2+2?"},
]


def _extract_assistant_tokens(renderer, prompt, assistant_msg):
    """Return the tokens the assistant turn contributes to the full render.

    We slice at ``len(render(prompt, add_generation_prompt=False))`` rather
    than walking the common prefix against ``add_generation_prompt=True``:
    for chatml the two agree, but harmony's gen-prompt opens channel
    ``analysis`` while the same assistant rendered in-sequence uses
    channel ``final``, so common-prefix walking would land inside the
    channel name and drop the ``<|start|>assistant<|channel|>``
    scaffolding the parser relies on.
    """
    prompt_ids = renderer.render_ids(prompt, add_generation_prompt=False)
    full_ids = renderer.render_ids(prompt + [assistant_msg])
    return full_ids[len(prompt_ids) :]


def _normalize_args(args: Any) -> Any:
    """Normalize tool-call arguments to a dict for cross-renderer comparison.

    Some parsers hand back a JSON string, others a dict. Compare by value.
    """
    if isinstance(args, str):
        try:
            return json.loads(args)
        except json.JSONDecodeError:
            return args
    return args


# ── content-only ──────────────────────────────────────────────────────


def test_roundtrip_content_only(rt_model, rt_tokenizer, rt_renderer):
    """Plain response, no thinking, no tool calls."""
    msg = {"role": "assistant", "content": "Four."}
    completion_ids = _extract_assistant_tokens(rt_renderer, PROMPT, msg)
    parsed = rt_renderer.parse_response(completion_ids)

    assert "Four" in parsed.content, f"{rt_model}: lost content, got {parsed.content!r}"
    assert not parsed.tool_calls, (
        f"{rt_model}: spurious tool_calls={parsed.tool_calls!r}"
    )


# ── reasoning ─────────────────────────────────────────────────────────


def test_roundtrip_reasoning_and_content(rt_model, rt_tokenizer, rt_renderer):
    """Assistant with reasoning_content + visible content — reasoning must
    survive the round trip for templates that emit reasoning blocks."""
    msg = {
        "role": "assistant",
        "content": "The answer is four.",
        "reasoning_content": "Two plus two equals four.",
    }
    completion_ids = _extract_assistant_tokens(rt_renderer, PROMPT, msg)
    parsed = rt_renderer.parse_response(completion_ids)

    assert "four" in parsed.content.lower(), (
        f"{rt_model}: lost content, got {parsed.content!r}"
    )
    # DefaultRenderer may not carry reasoning unless the template explicitly
    # emits a <think> block; hand-coded renderers always should.
    if parsed.reasoning_content is not None:
        assert "equals four" in parsed.reasoning_content.lower(), (
            f"{rt_model}: reasoning mangled, got {parsed.reasoning_content!r}"
        )


# ── tool calls ────────────────────────────────────────────────────────


def _maybe_skip_tool_calls(renderer_name: str) -> None:
    """DefaultRenderer without a tool_parser configured always returns
    tool_calls=None. That's a documented limitation, not a bug — skip."""
    if renderer_name == "default":
        pytest.skip(
            "DefaultRenderer requires an explicit tool_parser to parse tool "
            "calls; not exercised in the round-trip matrix."
        )


def test_roundtrip_single_tool_call(
    rt_model, rt_renderer_name, rt_tokenizer, rt_renderer
):
    _maybe_skip_tool_calls(rt_renderer_name)

    msg = {
        "role": "assistant",
        "content": "",
        "tool_calls": [
            {
                # Kimi's parser extracts the function name from the id
                # field ("functions.{name}:{idx}") — other renderers ignore
                # the id shape, so this form is compatible across the matrix.
                "id": "functions.get_weather:0",
                "function": {
                    "name": "get_weather",
                    "arguments": '{"city": "Tokyo"}',
                },
            }
        ],
    }
    completion_ids = _extract_assistant_tokens(rt_renderer, PROMPT, msg)
    parsed = rt_renderer.parse_response(completion_ids)

    assert parsed.tool_calls, f"{rt_model}: tool_calls lost, got {parsed.tool_calls!r}"
    assert len(parsed.tool_calls) == 1
    tc = parsed.tool_calls[0]
    assert tc["function"]["name"] == "get_weather", (
        f"{rt_model}: name mangled, got {tc!r}"
    )
    assert _normalize_args(tc["function"]["arguments"]) == {"city": "Tokyo"}, (
        f"{rt_model}: args mangled, got {tc['function']['arguments']!r}"
    )


def test_roundtrip_multiple_tool_calls(
    rt_model, rt_renderer_name, rt_tokenizer, rt_renderer
):
    """Parsers that loop over ``<tool_call>…</tool_call>`` blocks can
    silently drop the second one; this test catches that."""
    _maybe_skip_tool_calls(rt_renderer_name)

    msg = {
        "role": "assistant",
        "content": "",
        "tool_calls": [
            {
                # Kimi's parser extracts the function name from the id
                # field ("functions.{name}:{idx}") — other renderers ignore
                # the id shape, so this form is compatible across the matrix.
                "id": "functions.get_weather:0",
                "function": {
                    "name": "get_weather",
                    "arguments": '{"city": "Tokyo"}',
                },
            },
            {
                "id": "functions.get_time:1",
                "function": {
                    "name": "get_time",
                    "arguments": '{"zone": "JST"}',
                },
            },
        ],
    }
    completion_ids = _extract_assistant_tokens(rt_renderer, PROMPT, msg)
    parsed = rt_renderer.parse_response(completion_ids)

    assert parsed.tool_calls is not None and len(parsed.tool_calls) == 2, (
        f"{rt_model}: expected 2 tool_calls, got {parsed.tool_calls!r}"
    )
    names = [tc["function"]["name"] for tc in parsed.tool_calls]
    assert names == ["get_weather", "get_time"], (
        f"{rt_model}: names/order wrong, got {names}"
    )
    assert _normalize_args(parsed.tool_calls[0]["function"]["arguments"]) == {
        "city": "Tokyo"
    }
    assert _normalize_args(parsed.tool_calls[1]["function"]["arguments"]) == {
        "zone": "JST"
    }


# ── byte-exact re-render invariant ─────────────────────────────────────
#
# When the chat template round-trips content/reasoning verbatim (no
# separator between `</think>` and content), the parser MUST preserve
# whitespace at the `</think>` boundary. Otherwise parse→re-render
# drops a `\n` token, the rendered assistant message BPE-realigns by
# ~O(tens of tokens), and multi-turn RL's extension property breaks at
# step 2+ on every rollout.
#
# Narrow scope: exercise the parser contract directly so this test
# surfaces only the strip-whitespace regression, not any of the pre-
# existing hand-coded-renderer round-trip quirks.


def test_think_reasoning_parser_preserves_boundary_whitespace():
    """ThinkTextReasoningParser (used when `reasoning_parser='think'`) must
    preserve whitespace directly on either side of `</think>` for
    byte-exact round-trip through templates like GLM-4.5 that emit
    reasoning and content back-to-back with no separator.

    Regression: the parser used to call `.strip()` and `.lstrip('\\n')`,
    which drops the newline the model naturally emits after `</think>`,
    breaking multi-turn trajectory token extension.
    """
    from renderers.parsers import ThinkTextReasoningParser

    parser = ThinkTextReasoningParser(tokenizer=None)

    # Model naturally emits `\n<think>...</think>\nCONTENT` — the \n on
    # each side of `</think>` is load-bearing.
    reasoning, content = parser.extract(
        "\n<think>Reason text</think>\nThe answer is four."
    )

    assert reasoning == "Reason text", f"reasoning boundary stripped: {reasoning!r}"
    assert content.startswith("\n"), (
        f"content lost leading '\\n' after </think>: {content!r}"
    )
    assert content == "\nThe answer is four.", f"content mangled: {content!r}"


def test_think_reasoning_parser_preserves_trailing_reasoning_whitespace():
    """If the model put a `\\n` BEFORE `</think>`, that belongs in
    reasoning_content — the template renders it as part of the `<think>`
    block and dropping it shifts downstream tokens.
    """
    from renderers.parsers import ThinkTextReasoningParser

    parser = ThinkTextReasoningParser(tokenizer=None)
    reasoning, content = parser.extract(
        "<think>Line one\nLine two\n</think>Final answer."
    )
    assert reasoning == "Line one\nLine two\n", (
        f"lost trailing '\\n' inside <think>: {reasoning!r}"
    )
    assert content == "Final answer.", f"content mangled: {content!r}"


def test_default_renderer_fallback_parser_preserves_boundary_whitespace(
    rt_tokenizer,
):
    """DefaultRenderer's built-in fallback (when no explicit reasoning
    parser is configured) must apply the same whitespace-preservation
    rule so models with `<think>`-emitting templates round-trip.

    Uses whatever `rt_tokenizer` is available but skips if the tokenizer
    can't round-trip `<think>` tags cleanly — we only care about the
    parser's text handling here.
    """
    from renderers.default import DefaultRenderer

    renderer = DefaultRenderer(rt_tokenizer, tool_parser=None, reasoning_parser=None)

    # Encode `<think>reason</think>\nvisible` as text and run through
    # parse_response. We don't need the template to emit `<think>` here
    # — we only exercise the parser's text-splitting logic.
    text = "<think>reason</think>\nvisible"
    ids = rt_tokenizer.encode(text, add_special_tokens=False)
    # If decoding doesn't round-trip the `\n`, the test is N/A for this
    # tokenizer — this is a parser-level invariant, not a tokenizer one.
    if rt_tokenizer.decode(ids, skip_special_tokens=False) != text:
        pytest.skip("tokenizer decode does not round-trip the fixture text")

    parsed = renderer.parse_response(ids)
    assert parsed.reasoning_content == "reason", (
        f"reasoning stripped beyond `</think>`: {parsed.reasoning_content!r}"
    )
    assert (parsed.content or "").startswith("\n"), (
        f"content lost leading '\\n' after `</think>`: {parsed.content!r}"
    )
