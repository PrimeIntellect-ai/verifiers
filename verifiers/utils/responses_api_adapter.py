from openai.types.chat import ChatCompletionMessage
from openai.types.chat.chat_completion import Choice


class ResponsesAPIAdapter:
    """Adapter to normalize Responses API responses to ChatCompletion format."""

    def __init__(self, responses_response):
        self._response = responses_response
        self._text = getattr(responses_response, "output_text", "")
        self._tool_calls = self._extract_tool_calls()

    def _extract_tool_calls(self):
        tool_calls = []
        output = getattr(self._response, "output", [])

        for item in output:
            item_type = getattr(item, "type", None)
            if item_type == "function_call":
                tool_calls.append(
                    {
                        "id": getattr(item, "call_id", ""),
                        "type": "function",
                        "function": {
                            "name": getattr(item, "name", ""),
                            "arguments": str(getattr(item, "arguments", {})),
                        },
                    }
                )

        return tool_calls if tool_calls else None

    @property
    def choices(self):
        return [
            Choice(
                index=0,
                message=ChatCompletionMessage(role="assistant", content=self._text, tool_calls=self._tool_calls),
                finish_reason="stop",
            )
        ]

    @property
    def id(self):
        return getattr(self._response, "id", "responses-api-adapter")

    @property
    def model(self):
        return getattr(self._response, "model", "")

    @property
    def created(self):
        return getattr(self._response, "created_at", 0)

    @property
    def object(self):
        return "chat.completion"

    @property
    def usage(self):
        return getattr(self._response, "usage", None)
