"""Tests for the ThinkParser class."""

from verifiers import MaybeThinkParser


class TestMaybeThinkParser:
    """Test cases for the MaybeThinkParser class."""

    def test_parser_initialization(self, maybe_think_parser):
        """Test that MaybeThinkParser initializes correctly."""
        assert isinstance(maybe_think_parser, MaybeThinkParser)
        assert hasattr(maybe_think_parser, "logger")

    def test_does_not_parse_non_reasoner_text(self, maybe_think_parser):
        """Test that parse method returns text unchanged."""
        text = "This is a test string without thinking tags"
        result = maybe_think_parser.parse(text)
        assert result == text

    def test_parses_reasoner_text(self, maybe_think_parser):
        """Test that parse method returns text after thinking tags."""
        text = "<think>This is a test string with thinking tags</think> This is the final answer"
        result = maybe_think_parser.parse(text)
        assert result == "This is the final answer"

    def test_does_not_parse_non_reasoner_messages(self, maybe_think_parser):
        """Test that parse method returns text unchanged."""
        messages = [
            {"role": "user", "content": "What is 2+2?"},
            {"role": "assistant", "content": "The answer is 4"},
        ]
        result = maybe_think_parser.parse_answer(messages)
        assert result == messages[-1]["content"]

    def test_parses_reasoner_messages(self, maybe_think_parser):
        """Test parse_answer with completion list."""
        messages = [
            {"role": "user", "content": "What is 2+2?"},
            {
                "role": "assistant",
                "content": "<think>This is a test string with thinking tags</think>The answer is 4",
            },
        ]
        result = maybe_think_parser.parse_answer(messages)
        assert result == "The answer is 4"
