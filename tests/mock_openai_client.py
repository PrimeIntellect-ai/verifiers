"""
Mock OpenAI client for testing purposes.
"""

from typing import Dict, List, Optional
from unittest.mock import Mock


class MockCompletion:
    def __init__(self, content: str, finish_reason: str = "stop"):
        self.message = Mock()
        self.message.content = content
        self.text = content
        self.finish_reason = finish_reason


class MockCompletionResponse:
    def __init__(self, content: str, finish_reason: str = "stop"):
        self.choices = [MockCompletion(content, finish_reason)]


class MockChatCompletions:
    def __init__(
        self,
        responses: Optional[Dict[str, str]] = None,
        default_response: str = "Test response",
    ):
        self.responses = responses or {}
        self.default_response = default_response
        self.call_count = 0
        self.last_messages = None
        self.last_model = None

    def create(
        self, model: str, messages: List[Dict[str, str]], **kwargs
    ) -> MockCompletionResponse:
        self.call_count += 1
        self.last_messages = messages
        self.last_model = model

        # Handle special error cases
        if kwargs.get("max_tokens", 0) == 1:
            return MockCompletionResponse("", "length")

        # Check for specific response patterns
        last_message = messages[-1]["content"] if messages else ""
        for pattern, response in self.responses.items():
            if pattern in last_message:
                return MockCompletionResponse(response)

        return MockCompletionResponse(self.default_response)


class MockCompletions:
    def __init__(
        self,
        responses: Optional[Dict[str, str]] = None,
        default_response: str = "Test response",
    ):
        self.responses = responses or {}
        self.default_response = default_response
        self.call_count = 0
        self.last_prompt = None
        self.last_model = None

    def create(self, model: str, prompt: str, **kwargs) -> MockCompletionResponse:
        self.call_count += 1
        self.last_prompt = prompt
        self.last_model = model

        # Handle special error cases
        if kwargs.get("max_tokens", 0) == 1:
            return MockCompletionResponse("", "length")

        # Check for specific response patterns
        for pattern, response in self.responses.items():
            if pattern in prompt:
                return MockCompletionResponse(response)

        return MockCompletionResponse(self.default_response)


class MockOpenAIClient:
    """
    Mock OpenAI client that simulates API responses for testing.
    """

    def __init__(
        self,
        chat_responses: Optional[Dict[str, str]] = None,
        completion_responses: Optional[Dict[str, str]] = None,
        default_chat_response: str = "Test chat response",
        default_completion_response: str = "Test completion response",
        base_url: str = "http://localhost:8000",
    ):
        self.base_url = base_url
        self.chat = Mock()
        self.chat.completions = MockChatCompletions(
            chat_responses, default_chat_response
        )
        self.completions = MockCompletions(
            completion_responses, default_completion_response
        )

    def reset_counters(self):
        """Reset call counters for testing."""
        self.chat.completions.call_count = 0
        self.completions.call_count = 0


def create_mock_math_client():
    """Create a mock client that responds appropriately to math problems."""
    math_responses = {
        "What is 2+2": "<think>\n2 + 2 = 4\n</think>\n<answer>4</answer>",
        "What is 5*3": "<think>\n5 * 3 = 15\n</think>\n<answer>15</answer>",
        "solve x+1=3": "<think>\nx + 1 = 3\nx = 3 - 1\nx = 2\n</think>\n<answer>2</answer>",
    }
    return MockOpenAIClient(chat_responses=math_responses)


def create_mock_tool_client():
    """Create a mock client that responds with tool usage."""
    tool_responses = {
        "calculate": """<reasoning>
I need to use the calculator tool to compute this.
</reasoning>
<tool>
{"name": "calculator", "args": {"expression": "2+2"}}
</tool>""",
        "search": """<reasoning>
I need to search for information about this topic.
</reasoning>
<tool>
{"name": "search", "args": {"query": "python programming"}}
</tool>""",
    }
    return MockOpenAIClient(chat_responses=tool_responses)


def create_mock_error_client():
    """Create a mock client that simulates various error conditions."""

    def create_error(*args, **kwargs):
        if "longer than the maximum" in str(kwargs):
            raise Exception(
                "This model's maximum context length is 4096 tokens. Your message was longer than the maximum."
            )
        if "exceeds the model" in str(kwargs):
            raise Exception("Input exceeds the model's context window.")
        raise Exception("Generic error")

    client = MockOpenAIClient()
    client.chat.completions.create = create_error  # type: ignore[assignment]
    client.completions.create = create_error  # type: ignore[assignment]
    return client
