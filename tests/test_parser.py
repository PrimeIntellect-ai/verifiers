"""Tests for the base Parser class."""

from verifiers import Parser


class TestParser:
    """Test cases for the base Parser class."""

    def test_parser_initialization(self, basic_parser):
        """Test that Parser initializes correctly."""
        assert isinstance(basic_parser, Parser)
        assert hasattr(basic_parser, "logger")

    def test_parse_returns_text_as_is(self, basic_parser):
        """Test that parse method returns text unchanged."""
        text = "This is a test string"
        result = basic_parser.parse(text)
        assert result == text

    def test_get_assistant_messages(self, basic_parser):
        """Test extraction of assistant messages from completion."""
        completion = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there"},
            {"role": "user", "content": "How are you?"},
            {"role": "assistant", "content": "I'm doing well"},
        ]
        assistant_messages = basic_parser.get_assistant_messages(completion)
        assert len(assistant_messages) == 2
        assert assistant_messages[0]["content"] == "Hi there"
        assert assistant_messages[1]["content"] == "I'm doing well"

    def test_parse_answer_with_string(self, basic_parser):
        """Test parse_answer with string input."""
        text = "This is an answer"
        result = basic_parser.parse_answer(text)
        assert result == text

    def test_parse_answer_with_completion(self, basic_parser):
        """Test parse_answer with completion list."""
        completion = [
            {"role": "user", "content": "What is 2+2?"},
            {"role": "assistant", "content": "The answer is 4"},
        ]
        result = basic_parser.parse_answer(completion)
        assert result == "The answer is 4"

    def test_parse_answer_with_empty_completion(self, basic_parser):
        """Test parse_answer with empty completion list."""
        result = basic_parser.parse_answer([])
        assert result is None

    def test_parse_answer_with_no_assistant_messages(self, basic_parser):
        """Test parse_answer with completion containing no assistant messages."""
        completion = [{"role": "user", "content": "Hello"}]
        result = basic_parser.parse_answer(completion)
        assert result is None

    def test_get_format_reward_func(self, basic_parser):
        """Test that format reward function returns 1.0 by default."""
        reward_func = basic_parser.get_format_reward_func()
        completion = [{"role": "assistant", "content": "test"}]
        reward = reward_func(completion)
        assert reward == 1.0
