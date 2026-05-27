from openai.types.chat import ChatCompletion

from verifiers.utils.client_utils import normalize_chat_completion_data


def test_normalize_chat_completion_data_fills_null_finish_reason_for_tool_calls():
    data = {
        "id": "cmpl-test",
        "object": "chat.completion",
        "created": 0,
        "model": "qwen-test",
        "choices": [
            {
                "index": 0,
                "finish_reason": None,
                "message": {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "call-1",
                            "type": "function",
                            "function": {"name": "read", "arguments": "{}"},
                        }
                    ],
                },
            }
        ],
    }

    normalized = normalize_chat_completion_data(data)

    assert normalized["choices"][0]["finish_reason"] == "tool_calls"
    ChatCompletion.model_validate(normalized)
