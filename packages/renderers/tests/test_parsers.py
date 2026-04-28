"""Tests for the pluggable ToolParser / ReasoningParser registry."""

from __future__ import annotations

import pytest
from transformers import AutoTokenizer

from renderers.parsers import (
    REASONING_PARSERS,
    TOOL_PARSERS,
    Qwen3ToolParser,
    ThinkTextReasoningParser,
    get_reasoning_parser,
    get_tool_parser,
)


def test_registries_nonempty():
    assert "qwen3" in TOOL_PARSERS
    assert "qwen3.5" in TOOL_PARSERS
    assert "glm" in TOOL_PARSERS
    assert "deepseek_v3" in TOOL_PARSERS
    assert "think" in REASONING_PARSERS


def test_unknown_parser_errors():
    tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B", trust_remote_code=True)
    with pytest.raises(ValueError, match="Unknown tool_parser"):
        get_tool_parser("does-not-exist", tok)
    with pytest.raises(ValueError, match="Unknown reasoning_parser"):
        get_reasoning_parser("does-not-exist", tok)


def test_qwen3_tool_parser_roundtrip():
    """Tokenize a Hermes-style tool call, parse it back out."""
    tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B", trust_remote_code=True)
    parser = get_tool_parser("qwen3", tok)
    assert isinstance(parser, Qwen3ToolParser)

    completion_text = 'hello\n<tool_call>\n{"name": "search", "arguments": {"q": "rain"}}\n</tool_call>'
    token_ids = tok.encode(completion_text, add_special_tokens=False)
    content_ids, tool_calls = parser.extract(list(token_ids))
    assert tool_calls is not None
    assert tool_calls[0]["function"]["name"] == "search"
    assert tool_calls[0]["function"]["arguments"] == {"q": "rain"}
    # content_ids should cover everything up to (but not including) <tool_call>
    content_text = tok.decode(content_ids, skip_special_tokens=False)
    assert "hello" in content_text
    assert "<tool_call>" not in content_text


def test_qwen3_tool_parser_no_tool_call():
    tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B", trust_remote_code=True)
    parser = get_tool_parser("qwen3", tok)
    ids = tok.encode("just plain text response", add_special_tokens=False)
    content_ids, tool_calls = parser.extract(list(ids))
    assert tool_calls is None
    assert content_ids == list(ids)


def test_think_reasoning_parser_extracts_block():
    tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B", trust_remote_code=True)
    parser = get_reasoning_parser("think", tok)
    assert isinstance(parser, ThinkTextReasoningParser)
    reasoning, content = parser.extract("<think>let me think</think>the answer")
    assert reasoning == "let me think"
    assert content == "the answer"


def test_think_reasoning_parser_no_block():
    tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B", trust_remote_code=True)
    parser = get_reasoning_parser("think", tok)
    reasoning, content = parser.extract("no reasoning here")
    assert reasoning is None
    assert content == "no reasoning here"


def test_default_renderer_uses_parsers():
    """DefaultRenderer + parsers should extract tool calls and reasoning."""
    from renderers import create_renderer

    tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B", trust_remote_code=True)
    renderer = create_renderer(
        tok, renderer="default", tool_parser="qwen3", reasoning_parser="think"
    )
    assert renderer.supports_tools is True

    completion = '<think>think</think>ok\n<tool_call>\n{"name": "f", "arguments": {}}\n</tool_call>'
    ids = tok.encode(completion, add_special_tokens=False)
    parsed = renderer.parse_response(list(ids))
    assert parsed.reasoning_content == "think"
    assert parsed.content.startswith("ok")
    assert parsed.tool_calls is not None
    assert parsed.tool_calls[0]["function"]["name"] == "f"


def test_default_renderer_without_parsers_is_backward_compatible():
    """Without parsers, DefaultRenderer still does basic <think> extraction."""
    from renderers import create_renderer

    tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B", trust_remote_code=True)
    renderer = create_renderer(tok, renderer="default")
    assert renderer.supports_tools is False

    ids = tok.encode("<think>r</think>a", add_special_tokens=False)
    parsed = renderer.parse_response(list(ids))
    assert parsed.reasoning_content == "r"
    assert parsed.content == "a"
    assert parsed.tool_calls is None
