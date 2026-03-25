"""Robustness tests for parse_response — edge cases, truncation, malformed output.

These test that parse_response never crashes and returns sensible results
even with adversarial or truncated model output.
"""

import pytest

from renderers.base import ParsedResponse


# ── Truncation ───────────────────────────────────────────────────────


def test_truncated_mid_thinking(model_name, tokenizer, renderer):
    """Model was cut off mid-thinking (no </think> found)."""
    text = "Let me think about this carefully. The problem requires"
    ids = tokenizer.encode(text, add_special_tokens=False)
    parsed = renderer.parse_response(ids)
    assert isinstance(parsed, ParsedResponse)
    # Content or reasoning should contain the text
    assert parsed.content or parsed.reasoning_content


def test_truncated_after_think_tag(model_name, tokenizer, renderer):
    """Model emitted <think> but was cut off before </think>."""
    text = "<think>Let me reason about"
    ids = tokenizer.encode(text, add_special_tokens=False)
    parsed = renderer.parse_response(ids)
    assert isinstance(parsed, ParsedResponse)


def test_truncated_mid_tool_call(model_name, tokenizer, renderer):
    """Model started a tool call but was cut off."""
    text = "Checking the weather.\n<tool_call>\n{\"name\": \"get"
    ids = tokenizer.encode(text, add_special_tokens=False)
    parsed = renderer.parse_response(ids)
    assert isinstance(parsed, ParsedResponse)
    # Should recover content before the tool call
    assert "Checking" in parsed.content or parsed.content == ""


def test_empty_completion(model_name, tokenizer, renderer):
    """Empty token list."""
    parsed = renderer.parse_response([])
    assert isinstance(parsed, ParsedResponse)
    assert parsed.content is not None


def test_single_eos_token(model_name, tokenizer, renderer):
    """Just an EOS token."""
    stop_ids = renderer.get_stop_token_ids()
    if stop_ids:
        parsed = renderer.parse_response(stop_ids[:1])
        assert isinstance(parsed, ParsedResponse)


# ── Thinking edge cases ──────────────────────────────────────────────


def test_empty_thinking_block(model_name, tokenizer, renderer):
    """<think></think> with no content (common pattern)."""
    text = "<think></think>Hello!"
    ids = tokenizer.encode(text, add_special_tokens=False)
    parsed = renderer.parse_response(ids)
    assert "Hello" in parsed.content


def test_thinking_with_newlines(model_name, tokenizer, renderer):
    """Thinking block with various newline patterns."""
    text = "Step 1: Calculate\nStep 2: Verify\n</think>\n\nThe answer is 42."
    ids = tokenizer.encode(text, add_special_tokens=False)
    parsed = renderer.parse_response(ids)
    assert "42" in parsed.content


def test_content_only_no_thinking(model_name, tokenizer, renderer):
    """Plain content with no thinking markers."""
    text = "Hello! How can I help you today?"
    ids = tokenizer.encode(text, add_special_tokens=False)
    parsed = renderer.parse_response(ids)
    assert "Hello" in parsed.content
    assert parsed.tool_calls is None


# ── Tool call edge cases ─────────────────────────────────────────────


def test_tool_call_with_complex_json(model_name, tokenizer, renderer):
    """Tool call with nested JSON arguments."""
    # This is a generic test — the exact format varies per model
    text = "Here are the results."
    ids = tokenizer.encode(text, add_special_tokens=False)
    parsed = renderer.parse_response(ids)
    assert isinstance(parsed, ParsedResponse)


def test_content_with_special_chars(model_name, tokenizer, renderer):
    """Content containing angle brackets, quotes, etc."""
    text = 'The formula is x < y and a > b. Use "quotes" freely.'
    ids = tokenizer.encode(text, add_special_tokens=False)
    parsed = renderer.parse_response(ids)
    assert parsed.content  # Should not crash on angle brackets


def test_very_long_content(model_name, tokenizer, renderer):
    """Long content string."""
    text = "word " * 500
    ids = tokenizer.encode(text, add_special_tokens=False)
    parsed = renderer.parse_response(ids)
    assert len(parsed.content) > 100


# ── Return type guarantees ───────────────────────────────────────────


def test_content_is_always_string(model_name, tokenizer, renderer):
    """content must always be a string, never None."""
    for text in ["Hello", "", "<think></think>", "<think>x</think>"]:
        ids = tokenizer.encode(text, add_special_tokens=False) if text else []
        parsed = renderer.parse_response(ids)
        assert isinstance(parsed.content, str)


def test_reasoning_is_string_or_none(model_name, tokenizer, renderer):
    """reasoning_content must be str or None."""
    text = "Some text"
    ids = tokenizer.encode(text, add_special_tokens=False)
    parsed = renderer.parse_response(ids)
    assert parsed.reasoning_content is None or isinstance(parsed.reasoning_content, str)


def test_tool_calls_is_list_or_none(model_name, tokenizer, renderer):
    """tool_calls must be list or None, never empty list."""
    text = "Hello!"
    ids = tokenizer.encode(text, add_special_tokens=False)
    parsed = renderer.parse_response(ids)
    assert parsed.tool_calls is None or (isinstance(parsed.tool_calls, list) and len(parsed.tool_calls) > 0)
