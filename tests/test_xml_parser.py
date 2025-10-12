"""Tests for the XMLParser class."""

import pytest

from verifiers import XMLParser


class TestXMLParser:
    """Test cases for the XMLParser class."""

    def test_xml_parser_initialization(self, xml_parser):
        """Test that XMLParser initializes correctly."""
        assert isinstance(xml_parser, XMLParser)
        assert xml_parser.answer_field == "answer"

    def test_xml_parser_with_alternatives(self, xml_parser_with_alternatives):
        """Test XMLParser with alternative field names."""
        assert isinstance(xml_parser_with_alternatives, XMLParser)
        fields = xml_parser_with_alternatives.get_fields()
        assert "reasoning" in fields
        assert "code" in fields  # canonical name from ("code", "answer")

    def test_parse_simple_xml(self, xml_parser):
        """Test parsing simple XML with basic fields."""
        xml_text = """
        <reasoning>
        Let me think about this problem step by step.
        </reasoning>
        <answer>
        The final answer is 42.
        </answer>
        """
        result = xml_parser.parse(xml_text)
        assert result.reasoning == "Let me think about this problem step by step."
        assert result.answer == "The final answer is 42."

    def test_parse_xml_with_alternatives(self, xml_parser_with_alternatives):
        """Test parsing XML with alternative field names."""
        xml_text = """
        <reasoning>
        First, I need to understand the problem.
        </reasoning>
        <code>
        def solve(): return 42
        </code>
        """
        result = xml_parser_with_alternatives.parse(xml_text)
        assert result.reasoning == "First, I need to understand the problem."
        assert result.code == "def solve(): return 42"
        # Both alternatives should be accessible
        assert hasattr(result, "answer")
        assert result.answer is None

    def test_parse_missing_fields(self, xml_parser):
        """Test parsing XML with missing fields."""
        xml_text = "<reasoning>Only reasoning here</reasoning>"
        result = xml_parser.parse(xml_text)
        assert result.reasoning == "Only reasoning here"
        assert result.answer is None

    def test_parse_empty_fields(self, xml_parser):
        """Test parsing XML with empty fields."""
        xml_text = "<reasoning></reasoning><answer></answer>"
        result = xml_parser.parse(xml_text)
        assert result.reasoning == ""
        assert result.answer == ""

    def test_parse_no_strip(self, xml_parser):
        """Test parsing without stripping whitespace."""
        # Note: The regex pattern itself removes leading/trailing whitespace
        # from the capture group, so strip=False only affects the .strip() call
        xml_text = "<answer>  spaced content  </answer>"
        result_strip = xml_parser.parse(xml_text, strip=True)
        result_no_strip = xml_parser.parse(xml_text, strip=False)
        assert result_strip.answer == "spaced content"
        assert (
            result_no_strip.answer == "spaced content"
        )  # regex already strips whitespace

    def test_parse_answer_from_completion(self, xml_parser):
        """Test extracting answer from completion."""
        completion = [
            {"role": "user", "content": "Solve this problem"},
            {
                "role": "assistant",
                "content": "<reasoning>Let me think</reasoning><answer>42</answer>",
            },
            {
                "role": "assistant",
                "content": "<reasoning>Actually, let me reconsider</reasoning><answer>43</answer>",
            },
        ]
        result = xml_parser.parse_answer(completion)
        assert result == "43"  # Should get the last answer

    def test_parse_answer_from_string_completion(self, xml_parser):
        """Test extracting answer from string completion."""
        completion = "<answer>44</answer><reasoning>Actually, that's not right either.</reasoning><answer>45</answer>"
        result = xml_parser.parse_answer(completion)
        assert (
            result == "45"
        )  # Should return just the last answer, not a full parse() namespace

    def test_parse_answer_no_answer_field(self, xml_parser):
        """Test parse_answer when no answer field is found."""
        completion = [
            {
                "role": "assistant",
                "content": "<reasoning>Only reasoning here</reasoning>",
            }
        ]
        result = xml_parser.parse_answer(completion)
        assert result is None

    def test_get_format_str(self, xml_parser):
        """Test format string generation."""
        format_str = xml_parser.get_format_str()
        assert "<reasoning>" in format_str
        assert "</reasoning>" in format_str
        assert "<answer>" in format_str
        assert "</answer>" in format_str

    def test_get_format_str_with_alternatives(self, xml_parser_with_alternatives):
        """Test format string with alternatives."""
        format_str = xml_parser_with_alternatives.get_format_str()
        assert "code | answer" in format_str

    def test_format_method(self, xml_parser):
        """Test formatting keyword arguments into XML."""
        formatted = xml_parser.format(reasoning="My reasoning", answer="42")
        assert "<reasoning>\nMy reasoning\n</reasoning>" in formatted
        assert "<answer>\n42\n</answer>" in formatted

    def test_format_method_missing_field(self, xml_parser):
        """Test format method with missing required field."""
        with pytest.raises(ValueError, match="Missing value for field"):
            xml_parser.format(reasoning="Only reasoning")

    def test_format_method_with_alternatives(self, xml_parser_with_alternatives):
        """Test format method with alternative field names."""
        # Using canonical name
        formatted1 = xml_parser_with_alternatives.format(
            reasoning="test", code="print('hello')"
        )
        assert "<code>\nprint('hello')\n</code>" in formatted1

        # Using alternative name
        formatted2 = xml_parser_with_alternatives.format(
            reasoning="test", answer="print('hello')"
        )
        assert (
            "<code>\nprint('hello')\n</code>" in formatted2
        )  # Should use canonical tag

    def test_get_fields(self, xml_parser, xml_parser_with_alternatives):
        """Test getting field names."""
        fields1 = xml_parser.get_fields()
        assert fields1 == ["reasoning", "answer"]

        fields2 = xml_parser_with_alternatives.get_fields()
        assert fields2 == ["reasoning", "code"]

    def test_invalid_field_types(self):
        """Test XMLParser initialization with invalid field types."""
        from typing import Any, cast

        with pytest.raises(TypeError):
            bad_fields = cast(
                list[str | tuple[str, ...]], [cast(Any, "valid"), cast(Any, 123)]
            )
            XMLParser(bad_fields)  # Invalid field type mixed in

        # Empty fields is actually allowed - it just creates a parser with no fields
        empty_parser = XMLParser([])  # This works
        assert empty_parser.get_fields() == []

        with pytest.raises(ValueError):
            XMLParser(["field1", "field1"])  # Duplicate fields

    def test_format_reward_function(self, xml_parser):
        """Test the format reward function."""
        reward_func = xml_parser.get_format_reward_func()

        # Well-formatted completion
        good_completion = [
            {
                "role": "assistant",
                "content": "<reasoning>Good reasoning</reasoning><answer>42</answer>",
            }
        ]
        good_reward = reward_func(good_completion)
        assert 0.0 <= good_reward <= 1.0

        # Poorly formatted completion - gets partial credit for proper spacing
        bad_completion = [
            {"role": "assistant", "content": "Just plain text without XML"}
        ]
        bad_reward = reward_func(bad_completion)
        assert bad_reward == 0.2  # Gets 0.2 for proper spacing (no XML tags to mess up)
