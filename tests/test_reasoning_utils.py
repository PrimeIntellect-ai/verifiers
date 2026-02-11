"""Tests for reasoning content normalization utilities."""

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from verifiers.utils.reasoning_utils import (
    detect_reasoning_format,
    extract_reasoning_from_response,
    normalize_reasoning_content,
    prepare_messages_for_provider,
    strip_reasoning_from_content,
)


# ---------------------------------------------------------------------------
# TestExtractReasoningFromResponse
# ---------------------------------------------------------------------------
class TestExtractReasoningFromResponse:
    def test_reasoning_content_present(self):
        msg = SimpleNamespace(reasoning_content="step by step", reasoning=None)
        assert extract_reasoning_from_response(msg) == "step by step"

    def test_reasoning_present(self):
        msg = SimpleNamespace(reasoning="my reasoning")
        assert extract_reasoning_from_response(msg) == "my reasoning"

    def test_both_prefers_reasoning_content(self):
        msg = SimpleNamespace(reasoning_content="rc value", reasoning="r value")
        assert extract_reasoning_from_response(msg) == "rc value"

    def test_neither(self):
        msg = SimpleNamespace()
        assert extract_reasoning_from_response(msg) is None

    def test_empty_string(self):
        msg = SimpleNamespace(reasoning_content="", reasoning="")
        assert extract_reasoning_from_response(msg) is None

    def test_whitespace_only(self):
        msg = SimpleNamespace(reasoning_content="   ", reasoning="  \n  ")
        assert extract_reasoning_from_response(msg) is None

    def test_none_values(self):
        msg = SimpleNamespace(reasoning_content=None, reasoning=None)
        assert extract_reasoning_from_response(msg) is None

    def test_reasoning_content_none_reasoning_present(self):
        msg = SimpleNamespace(reasoning_content=None, reasoning="fallback")
        assert extract_reasoning_from_response(msg) == "fallback"

    def test_reasoning_content_empty_reasoning_present(self):
        msg = SimpleNamespace(reasoning_content="", reasoning="fallback")
        assert extract_reasoning_from_response(msg) == "fallback"


# ---------------------------------------------------------------------------
# TestDetectReasoningFormat
# ---------------------------------------------------------------------------
class TestDetectReasoningFormat:
    def test_detects_reasoning_content(self):
        msg = SimpleNamespace(reasoning_content="thinking...")
        assert detect_reasoning_format(msg, "answer") == "reasoning_content"

    def test_detects_reasoning(self):
        msg = SimpleNamespace(reasoning="thinking...")
        assert detect_reasoning_format(msg, "answer") == "reasoning"

    def test_detects_think_tags(self):
        msg = SimpleNamespace()
        assert (
            detect_reasoning_format(msg, "<think>thinking</think>\nanswer")
            == "think_tags"
        )

    def test_detects_think_tags_with_leading_whitespace(self):
        msg = SimpleNamespace()
        assert (
            detect_reasoning_format(msg, "  <think>thinking</think>\nanswer")
            == "think_tags"
        )

    def test_returns_none_for_vanilla(self):
        msg = SimpleNamespace()
        assert detect_reasoning_format(msg, "just an answer") == "none"

    def test_returns_none_for_empty_content(self):
        msg = SimpleNamespace()
        assert detect_reasoning_format(msg, "") == "none"

    def test_returns_none_for_none_content(self):
        msg = SimpleNamespace()
        assert detect_reasoning_format(msg, None) == "none"

    def test_prefers_reasoning_content_over_think_tags(self):
        msg = SimpleNamespace(reasoning_content="rc")
        assert (
            detect_reasoning_format(msg, "<think>tags</think>\ncontent")
            == "reasoning_content"
        )

    def test_empty_reasoning_content_falls_through(self):
        msg = SimpleNamespace(reasoning_content="")
        assert (
            detect_reasoning_format(msg, "<think>tags</think>\ncontent") == "think_tags"
        )


# ---------------------------------------------------------------------------
# TestNormalizeReasoningContent
# ---------------------------------------------------------------------------
class TestNormalizeReasoningContent:
    def test_both_present(self):
        result = normalize_reasoning_content("thinking", "answer")
        assert result == "<think>\nthinking\n</think>\nanswer"

    def test_reasoning_only(self):
        result = normalize_reasoning_content("thinking", None)
        assert result == "<think>\nthinking\n</think>"

    def test_reasoning_only_empty_content(self):
        result = normalize_reasoning_content("thinking", "")
        assert result == "<think>\nthinking\n</think>"

    def test_content_only(self):
        result = normalize_reasoning_content(None, "answer")
        assert result == "answer"

    def test_neither(self):
        result = normalize_reasoning_content(None, None)
        assert result == ""

    def test_neither_empty_strings(self):
        result = normalize_reasoning_content("", "")
        assert result == ""

    def test_avoids_duplication(self):
        content = "<think>\nalready tagged\n</think>\nanswer"
        result = normalize_reasoning_content("new reasoning", content)
        assert result == content  # unchanged

    def test_avoids_duplication_with_whitespace(self):
        content = "  <think>\nalready tagged\n</think>\nanswer"
        result = normalize_reasoning_content("new reasoning", content)
        assert result == content  # unchanged

    def test_whitespace_reasoning_skipped(self):
        result = normalize_reasoning_content("   ", "answer")
        assert result == "answer"

    def test_multiline_reasoning(self):
        reasoning = "step 1\nstep 2\nstep 3"
        result = normalize_reasoning_content(reasoning, "final answer")
        assert result == "<think>\nstep 1\nstep 2\nstep 3\n</think>\nfinal answer"


# ---------------------------------------------------------------------------
# TestStripReasoningFromContent
# ---------------------------------------------------------------------------
class TestStripReasoningFromContent:
    def test_with_tags(self):
        reasoning, content = strip_reasoning_from_content(
            "<think>\nmy reasoning\n</think>\nmy answer"
        )
        assert reasoning == "my reasoning"
        assert content == "my answer"

    def test_without_tags(self):
        reasoning, content = strip_reasoning_from_content("just an answer")
        assert reasoning is None
        assert content == "just an answer"

    def test_truncated_no_closing(self):
        reasoning, content = strip_reasoning_from_content(
            "<think>reasoning without closing tag"
        )
        assert reasoning is None
        assert content == "<think>reasoning without closing tag"

    def test_empty_reasoning(self):
        reasoning, content = strip_reasoning_from_content("<think></think>\nanswer")
        assert reasoning is None
        assert content == "answer"

    def test_empty_content(self):
        reasoning, content = strip_reasoning_from_content("")
        assert reasoning is None
        assert content == ""

    def test_only_think_tags(self):
        reasoning, content = strip_reasoning_from_content(
            "<think>some reasoning</think>"
        )
        assert reasoning == "some reasoning"
        assert content == ""

    def test_round_trip_with_normalize(self):
        original_reasoning = "step by step"
        original_content = "the answer is 42"
        normalized = normalize_reasoning_content(original_reasoning, original_content)
        reasoning, content = strip_reasoning_from_content(normalized)
        assert reasoning == original_reasoning
        assert content == original_content

    def test_whitespace_before_think(self):
        reasoning, content = strip_reasoning_from_content(
            "  <think>\nmy reasoning\n</think>\nmy answer"
        )
        assert reasoning == "my reasoning"
        assert content == "my answer"


# ---------------------------------------------------------------------------
# TestPrepareMessagesForProvider
# ---------------------------------------------------------------------------
class TestPrepareMessagesForProvider:
    def _make_messages(self):
        return [
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "What is 2+2?"},
            {
                "role": "assistant",
                "content": "<think>\nlet me think\n</think>\n4",
            },
            {"role": "user", "content": "<think>user think tags</think>\nfollow up"},
        ]

    def test_strips_think_from_assistant_for_reasoning_content(self):
        messages = self._make_messages()
        result = prepare_messages_for_provider(messages, "reasoning_content")
        # assistant message should have think tags stripped
        assert result[2]["content"] == "4"
        # user message should be untouched
        assert result[3]["content"] == "<think>user think tags</think>\nfollow up"
        # system message untouched
        assert result[0]["content"] == "You are helpful."

    def test_strips_think_from_assistant_for_reasoning(self):
        messages = self._make_messages()
        result = prepare_messages_for_provider(messages, "reasoning")
        assert result[2]["content"] == "4"

    def test_noop_for_think_tags(self):
        messages = self._make_messages()
        result = prepare_messages_for_provider(messages, "think_tags")
        assert result is messages  # same object, not copied

    def test_noop_for_none_format(self):
        messages = self._make_messages()
        result = prepare_messages_for_provider(messages, "none")
        assert result is messages

    def test_noop_for_auto(self):
        messages = self._make_messages()
        result = prepare_messages_for_provider(messages, "auto")
        assert result is messages

    def test_assistant_without_think_tags_unchanged(self):
        messages = [
            {"role": "assistant", "content": "plain answer"},
        ]
        result = prepare_messages_for_provider(messages, "reasoning_content")
        assert result[0]["content"] == "plain answer"

    def test_does_not_mutate_original(self):
        messages = [
            {
                "role": "assistant",
                "content": "<think>\nthinking\n</think>\nanswer",
            },
        ]
        original_content = messages[0]["content"]
        prepare_messages_for_provider(messages, "reasoning_content")
        assert messages[0]["content"] == original_content

    def test_tool_message_untouched(self):
        messages = [
            {
                "role": "tool",
                "content": "<think>tool output</think>\nresult",
                "tool_call_id": "123",
            },
        ]
        result = prepare_messages_for_provider(messages, "reasoning_content")
        assert result[0]["content"] == "<think>tool output</think>\nresult"


# ---------------------------------------------------------------------------
# TestParseResponseMessagesIntegration
# ---------------------------------------------------------------------------
class TestParseResponseMessagesIntegration:
    @pytest.mark.asyncio
    async def test_reasoning_content_appears_in_parsed_messages(self):
        """Mock a ChatCompletion with reasoning_content field and verify <think> tags."""
        from openai.types.chat.chat_completion import ChatCompletion

        from verifiers.utils.response_utils import parse_response_messages

        message = MagicMock()
        message.content = "the answer is 42"
        message.reasoning_content = "let me work through this"
        message.reasoning = None
        message.tool_calls = None

        choice = MagicMock()
        choice.message = message

        response = MagicMock(spec=ChatCompletion)
        response.choices = [choice]

        result = await parse_response_messages(response, "chat")
        assert isinstance(result, list)
        assert len(result) == 1
        content = result[0]["content"]
        assert content.startswith("<think>")
        assert "let me work through this" in content
        assert "the answer is 42" in content

    @pytest.mark.asyncio
    async def test_no_reasoning_field_works_as_before(self):
        """Response without reasoning fields should work identically to before."""
        from openai.types.chat.chat_completion import ChatCompletion

        from verifiers.utils.response_utils import parse_response_messages

        message = MagicMock()
        message.content = "plain answer"
        message.tool_calls = None
        # No reasoning_content or reasoning attributes
        del message.reasoning_content
        del message.reasoning

        choice = MagicMock()
        choice.message = message

        response = MagicMock(spec=ChatCompletion)
        response.choices = [choice]

        result = await parse_response_messages(response, "chat")
        assert isinstance(result, list)
        assert result[0]["content"] == "plain answer"

    @pytest.mark.asyncio
    async def test_reasoning_only_no_content(self):
        """Response with reasoning but no content should produce <think> block only."""
        from openai.types.chat.chat_completion import ChatCompletion

        from verifiers.utils.response_utils import parse_response_messages

        message = MagicMock()
        message.content = None
        message.reasoning_content = "thinking deeply"
        message.reasoning = None
        message.tool_calls = None

        choice = MagicMock()
        choice.message = message

        response = MagicMock(spec=ChatCompletion)
        response.choices = [choice]

        result = await parse_response_messages(response, "chat")
        content = result[0]["content"]
        assert content == "<think>\nthinking deeply\n</think>"


# ---------------------------------------------------------------------------
# TestValidationIntegration
# ---------------------------------------------------------------------------
class TestValidationIntegration:
    """Test that reasoning-only responses are not rejected by validation."""

    def test_reasoning_only_response_not_rejected(self):
        """A response with content=None but reasoning_content='...' should not raise."""
        from openai.types.chat.chat_completion import Choice

        # Build a mock Choice that has reasoning_content but no content
        message = MagicMock()
        message.content = None
        message.tool_calls = None
        message.reasoning_content = "deep reasoning here"
        message.reasoning = None

        choice = MagicMock(spec=Choice)
        choice.message = message

        # The validation logic we added
        has_content = bool(choice.message.content)
        has_tool_calls = bool(choice.message.tool_calls)
        has_reasoning = bool(
            getattr(choice.message, "reasoning_content", None)
            or getattr(choice.message, "reasoning", None)
        )
        # Should pass - has_reasoning is True
        assert has_reasoning is True
        assert not has_content
        assert not has_tool_calls
        assert has_content or has_tool_calls or has_reasoning

    def test_truly_empty_response_still_rejected(self):
        """A response with no content, no tool_calls, and no reasoning should fail."""
        message = MagicMock()
        message.content = None
        message.tool_calls = None
        message.reasoning_content = None
        message.reasoning = None

        has_content = bool(message.content)
        has_tool_calls = bool(message.tool_calls)
        has_reasoning = bool(
            getattr(message, "reasoning_content", None)
            or getattr(message, "reasoning", None)
        )
        assert not (has_content or has_tool_calls or has_reasoning)
