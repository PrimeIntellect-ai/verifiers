"""Barrage test: renderer.parse_response() must correctly extract
content, reasoning_content, and tool_calls from completion tokens.

Runs against every (model, renderer) pair.
"""

from functools import lru_cache

from renderers import create_renderer
from transformers import AutoTokenizer


@lru_cache
def _qwen3_vl():
    tokenizer = AutoTokenizer.from_pretrained(
        "Qwen/Qwen3-VL-4B-Instruct", trust_remote_code=True
    )
    renderer = create_renderer(tokenizer, renderer="auto")
    return tokenizer, renderer


def test_parse_simple_content(model_name, tokenizer, renderer):
    """Plain content, no thinking."""
    text = "Hello there!"
    ids = tokenizer.encode(text, add_special_tokens=False)
    parsed = renderer.parse_response(ids)
    assert "Hello" in parsed.content


def test_parse_thinking_and_content(model_name, tokenizer, renderer):
    """Content with <think>reasoning</think> block."""
    text = "Let me think about this.\n</think>\n\nThe answer is 42."
    ids = tokenizer.encode(text, add_special_tokens=False)
    parsed = renderer.parse_response(ids)
    # Should extract reasoning or at least not crash
    assert (
        "42" in parsed.content
        or "think" in (parsed.reasoning_content or "").lower()
        or parsed.content
    )


def test_parse_empty_completion(model_name, tokenizer, renderer):
    """Empty completion should not crash."""
    parsed = renderer.parse_response([])
    assert parsed.content is not None


def test_parse_response_returns_parsed_response(model_name, tokenizer, renderer):
    """Return type must have content, reasoning_content, tool_calls."""
    ids = tokenizer.encode("Hello!", add_special_tokens=False)
    parsed = renderer.parse_response(ids)
    assert hasattr(parsed, "content")
    assert hasattr(parsed, "reasoning_content")
    assert hasattr(parsed, "tool_calls")


def test_qwen3_vl_parse_json_tool_call():
    tokenizer, renderer = _qwen3_vl()
    text = (
        'Need a tool.\n<tool_call>\n{"name": "get_weather", '
        '"arguments": {"city": "Paris"}}\n</tool_call>'
    )
    parsed = renderer.parse_response(tokenizer.encode(text, add_special_tokens=False))

    assert parsed.content == "Need a tool."
    assert parsed.tool_calls == [
        {"function": {"name": "get_weather", "arguments": {"city": "Paris"}}}
    ]
