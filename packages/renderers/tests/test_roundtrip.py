"""Render → parse round-trip.

Core renderer invariant (per tinker's rendering notes): if you render
``[user, assistant(content=X, reasoning=Y, tool_calls=[T])]`` to tokens,
extract the assistant's completion slice, and feed it through
``parse_response``, you should get back an equivalent structured message.

Runs against every (model, renderer) pair in conftest.py so we catch any
asymmetry between a renderer's emit path and its parse path — bugs that
slip past render-parity tests (which only check vs apply_chat_template)
and parse-robustness tests (which feed crafted text, not rendered
output).
"""

from __future__ import annotations

import json
from typing import Any

import pytest


PROMPT = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "What is 2+2?"},
]


def _extract_assistant_tokens(renderer, prompt, assistant_msg):
    """Render [prompt, assistant_msg]; return the tokens AFTER the prompt's
    generation-prompt render — i.e. the slice the model would have produced."""
    prompt_ids = renderer.render_ids(prompt, add_generation_prompt=True)
    full_ids = renderer.render_ids(prompt + [assistant_msg])
    # full_ids starts with the prompt's prefix, but the "add_generation_prompt"
    # version of the prompt may include extra scaffolding tokens that only appear
    # before a generated assistant. Walk the common prefix to be safe.
    common = 0
    for a, b in zip(prompt_ids, full_ids):
        if a != b:
            break
        common += 1
    return full_ids[common:]


def _normalize_args(args: Any) -> Any:
    """Normalize tool-call arguments to a dict for cross-renderer comparison.

    Some parsers hand back a JSON string, others a dict. We compare by value.
    """
    if isinstance(args, str):
        try:
            return json.loads(args)
        except json.JSONDecodeError:
            return args
    return args


# ── content-only ──────────────────────────────────────────────────────


def test_roundtrip_content_only(model_name, tokenizer, renderer):
    """Plain response, no thinking, no tool calls."""
    msg = {"role": "assistant", "content": "Four."}
    completion_ids = _extract_assistant_tokens(renderer, PROMPT, msg)
    parsed = renderer.parse_response(completion_ids)

    assert "Four" in parsed.content, f"{model_name}: lost content, got {parsed.content!r}"
    assert not parsed.tool_calls, (
        f"{model_name}: spurious tool_calls={parsed.tool_calls!r}"
    )


# ── reasoning ─────────────────────────────────────────────────────────


def test_roundtrip_reasoning_and_content(model_name, tokenizer, renderer):
    """Assistant with reasoning_content + visible content — reasoning must
    survive the round trip for templates that emit reasoning blocks."""
    msg = {
        "role": "assistant",
        "content": "The answer is four.",
        "reasoning_content": "Two plus two equals four.",
    }
    completion_ids = _extract_assistant_tokens(renderer, PROMPT, msg)
    parsed = renderer.parse_response(completion_ids)

    assert "four" in parsed.content.lower(), (
        f"{model_name}: lost content, got {parsed.content!r}"
    )
    # DefaultRenderer may not carry reasoning unless the template explicitly
    # emits a <think> block; hand-coded renderers always should.
    if parsed.reasoning_content is not None:
        assert "equals four" in parsed.reasoning_content.lower(), (
            f"{model_name}: reasoning mangled, got "
            f"{parsed.reasoning_content!r}"
        )


# ── tool calls ────────────────────────────────────────────────────────


def _maybe_skip_tool_calls(model_name: str, renderer_name: str) -> None:
    """DefaultRenderer without a tool_parser configured always returns
    tool_calls=None. That's a documented limitation, not a bug — skip."""
    if renderer_name == "default":
        pytest.skip(
            "DefaultRenderer requires an explicit tool_parser to parse tool "
            "calls; not exercised in the round-trip matrix."
        )


def test_roundtrip_single_tool_call(
    model_name, renderer_name, tokenizer, renderer
):
    _maybe_skip_tool_calls(model_name, renderer_name)

    msg = {
        "role": "assistant",
        "content": "",
        "tool_calls": [
            {
                "id": "call_0",
                "function": {
                    "name": "get_weather",
                    "arguments": '{"city": "Tokyo"}',
                },
            }
        ],
    }
    completion_ids = _extract_assistant_tokens(renderer, PROMPT, msg)
    parsed = renderer.parse_response(completion_ids)

    assert parsed.tool_calls, (
        f"{model_name}: tool_calls lost, got {parsed.tool_calls!r}"
    )
    assert len(parsed.tool_calls) == 1
    tc = parsed.tool_calls[0]
    assert tc["function"]["name"] == "get_weather", (
        f"{model_name}: name mangled, got {tc!r}"
    )
    assert _normalize_args(tc["function"]["arguments"]) == {"city": "Tokyo"}, (
        f"{model_name}: args mangled, got {tc['function']['arguments']!r}"
    )


def test_roundtrip_multiple_tool_calls(
    model_name, renderer_name, tokenizer, renderer
):
    """Gap #2 from the parse-coverage audit: nothing exercises >1 tool call
    in a single assistant turn. Parsers that loop over tool_call blocks
    can silently drop the second one; this test catches that."""
    _maybe_skip_tool_calls(model_name, renderer_name)

    msg = {
        "role": "assistant",
        "content": "",
        "tool_calls": [
            {
                "id": "call_0",
                "function": {
                    "name": "get_weather",
                    "arguments": '{"city": "Tokyo"}',
                },
            },
            {
                "id": "call_1",
                "function": {
                    "name": "get_time",
                    "arguments": '{"zone": "JST"}',
                },
            },
        ],
    }
    completion_ids = _extract_assistant_tokens(renderer, PROMPT, msg)
    parsed = renderer.parse_response(completion_ids)

    assert parsed.tool_calls is not None and len(parsed.tool_calls) == 2, (
        f"{model_name}: expected 2 tool_calls, got "
        f"{parsed.tool_calls!r}"
    )
    names = [tc["function"]["name"] for tc in parsed.tool_calls]
    assert names == ["get_weather", "get_time"], (
        f"{model_name}: names/order wrong, got {names}"
    )
    assert _normalize_args(parsed.tool_calls[0]["function"]["arguments"]) == {
        "city": "Tokyo"
    }
    assert _normalize_args(parsed.tool_calls[1]["function"]["arguments"]) == {
        "zone": "JST"
    }
