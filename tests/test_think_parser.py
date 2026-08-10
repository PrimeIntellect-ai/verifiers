"""Tests for the ThinkParser class."""

from verifiers import ThinkParser


class TestThinkParser:
    """Test cases for the ThinkParser class."""

    def test_think_parser_initialization(self, think_parser):
        """Test that ThinkParser initializes correctly."""
        assert isinstance(think_parser, ThinkParser)
        assert hasattr(think_parser, "extract_fn")

    def test_think_parser_with_custom_extractor(self, think_parser_with_extractor):
        """Test ThinkParser with custom extraction function."""
        assert isinstance(think_parser_with_extractor, ThinkParser)

    def test_parse_with_think_tags(self, think_parser):
        """Test parsing text with think tags."""
        text = """<think>
        Let me think about this problem.
        I need to consider multiple factors.
        </think>
        The final answer is 42."""

        result = think_parser.parse(text)
        assert result == "The final answer is 42."

    def test_parse_without_think_tags(self, think_parser):
        """Test parsing text without think tags."""
        text = "Just a simple answer without thinking tags."
        result = think_parser.parse(text)
        assert result == ""

    def test_parse_with_multiple_think_blocks(self, think_parser):
        """Test parsing with multiple think blocks (should use content after last one)."""
        text = """<think>First thought</think>
        Some intermediate text.
        <think>Second thought</think>
        Final answer here."""

        result = think_parser.parse(text)
        assert result == "Final answer here."

    def test_parse_with_custom_extractor(self, think_parser_with_extractor):
        """Test parsing with custom extraction function."""
        text = """<think>
        I need to solve this step by step.
        </think>
        The answer is \\boxed{42}."""

        result = think_parser_with_extractor.parse(text)
        assert result == "42"

    def test_parse_with_custom_extractor_no_boxed(self, think_parser_with_extractor):
        """Test custom extractor when no boxed answer is found."""
        text = """<think>
        Thinking about the problem.
        </think>
        Just a plain answer."""

        result = think_parser_with_extractor.parse(text)
        assert result == "Just a plain answer."

    def test_parse_empty_after_think(self, think_parser):
        """Test parsing when content after think tags is empty."""
        text = "<think>Some thinking</think>"
        result = think_parser.parse(text)
        assert result == ""

    def test_parse_whitespace_handling(self, think_parser):
        """Test that whitespace is properly stripped."""
        text = """<think>
        Thinking process here.
        </think>
        
        Answer with spaces around it.
        
        """
        result = think_parser.parse(text)
        assert result == "Answer with spaces around it."

    def test_format_reward_function_good_format(self, think_parser):
        """Test format reward function with well-formatted content."""
        reward_func = think_parser.get_format_reward_func()

        completion = [
            {"role": "assistant", "content": "<think>Let me think</think>Final answer"}
        ]
        reward = reward_func(completion)
        assert reward == 1.0

    def test_format_reward_function_bad_format(self, think_parser):
        """Test format reward function with poorly formatted content."""
        reward_func = think_parser.get_format_reward_func()

        # Missing think tags
        bad_completion1 = [
            {"role": "assistant", "content": "Just an answer without thinking"}
        ]
        reward1 = reward_func(bad_completion1)
        assert reward1 == 0.0

        # Multiple think tags
        bad_completion2 = [
            {
                "role": "assistant",
                "content": "<think>First</think><think>Second</think>Answer",
            }
        ]
        reward2 = reward_func(bad_completion2)
        assert reward2 == 0.0

        # No content after think
        bad_completion3 = [
            {"role": "assistant", "content": "<think>Only thinking</think>"}
        ]
        reward3 = reward_func(bad_completion3)
        assert reward3 == 0.0

    def test_format_reward_function_mixed_messages(self, think_parser):
        """Test format reward function with mixed good and bad messages."""
        reward_func = think_parser.get_format_reward_func()

        completion = [
            {"role": "assistant", "content": "<think>Good thinking</think>Good answer"},
            {"role": "assistant", "content": "Bad answer without thinking"},
            {
                "role": "assistant",
                "content": "<think>More thinking</think>Another good answer",
            },
        ]
        reward = reward_func(completion)
        assert reward == 2.0 / 3.0  # 2 out of 3 messages are well-formatted

    def test_format_reward_function_no_assistant_messages(self, think_parser):
        """Test format reward function with no assistant messages."""
        reward_func = think_parser.get_format_reward_func()

        completion = [{"role": "user", "content": "Question"}]
        # Should handle gracefully, though the implementation might vary
        # This tests robustness of the reward function
        try:
            reward = reward_func(completion)
            # If it doesn't raise an error, the reward should be reasonable
            assert 0.0 <= reward <= 1.0
        except (ZeroDivisionError, IndexError):
            # If it raises an error, that's also acceptable behavior
            # since there are no assistant messages to evaluate
            pass

    def test_parse_answer_integration(self, think_parser):
        """Test parse_answer method inherited from Parser."""
        completion = [
            {"role": "user", "content": "What is 2+2?"},
            {
                "role": "assistant",
                "content": "<think>Let me calculate</think>The answer is 4",
            },
        ]
        result = think_parser.parse_answer(completion)
        assert result == "The answer is 4"

    def test_parse_answer_string_integration(self, think_parser):
        """Test parse_answer with string input."""
        text = "<think>Calculating...</think>Result: 42"
        result = think_parser.parse_answer(text)
        assert result == "Result: 42"
